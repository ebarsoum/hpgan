import sys
import time
import os
import math
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging
import random as rnd
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Filter out all logs.

import tensorflow as tf

from braniac import nn as nn
from braniac.format import SourceFactory
from braniac.utils import DataPreprocessing, NormalizationMode
from braniac.viz import Skeleton2D
from braniac.readers.body import SequenceBodyReader
from braniac.models.body import RNNDiscriminator, NNDiscriminator, SequenceToSequenceGenerator

def main(args):
    '''
    Main entry point that drive GAN training for body and skeleton data.

    Args:
        args: arg parser object, contains all arguments provided by the user.
    '''

    # setting up paths and log information.
    base_folder = args.output_folder
    output_path = os.path.join(base_folder, R'output')
    output_folder = os.path.join(output_path, "")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_models_folder = os.path.join(output_folder, "models")
    if not os.path.exists(output_models_folder):
        os.makedirs(output_models_folder)
    
    output_videos_folder = os.path.join(output_folder, "videos")
    if not os.path.exists(output_videos_folder):
        os.makedirs(output_videos_folder)

    output_tensorboard_folder = os.path.join(output_folder, "tensorboard")
    if not os.path.exists(output_tensorboard_folder):
        os.makedirs(output_tensorboard_folder)

    # creating logging file
    logging.basicConfig(filename=os.path.join(output_folder, "train.log"), filemode='w', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    max_epochs = args.max_epochs
    dataset = args.dataset_name

    # Return sensor and body information for the specific dataset.
    source = SourceFactory(dataset, args.camera_calibration_file)
    sensor = source.create_sensor()
    body_inf = source.create_body()

    # Log training information
    logging.info("Dataset: {}, Generator: {}".format(dataset, args.gan_type))

    # create the visualizer
    skeleton2D = Skeleton2D(sensor, body_inf)

    # input and output information
    input_sequence_length = 10
    output_sequence_length = 20
    sequence_length = input_sequence_length + output_sequence_length
    inputs_depth = 128
    z_size = 128 # Latent value that control predicted poses.
    data_preprocessing = None

    # prepare the data.
    logging.info("Loading data...")
    if args.data_normalization_file is not None:
        data_preprocessing = DataPreprocessing(args.data_normalization_file,
                                               normalization_mode=NormalizationMode.MeanAndStd2)

    if dataset == 'nturgbd':
        train_data_reader = SequenceBodyReader(args.train_file, 
                                               sequence_length,
                                               dataset,
                                               skip_frame=0,
                                               data_preprocessing=data_preprocessing,
                                               random_sequence=False)
    elif dataset == 'human36m':
        train_data_reader = SequenceBodyReader(args.train_file, 
                                               sequence_length,
                                               dataset,
                                               skip_frame=1,
                                               data_preprocessing=data_preprocessing,
                                               random_sequence=True)
    else:
        raise ValueError("Invalid dataset value.")

    # setting up the model
    minibatch_size = 16
    lr_init = 5e-5

    d_lr = lr_init
    g_lr = lr_init    
    epoch = 0

    d_inputs = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length] + list(train_data_reader.element_shape), name="d_inputs")
    g_inputs = tf.placeholder(dtype=tf.float32, shape=[None, input_sequence_length] + list(train_data_reader.element_shape), name="g_inputs")
    g_z = tf.placeholder(dtype=tf.float32, shape=[None, z_size], name="g_z")

    # Defining the model.
    d_real = NNDiscriminator(d_inputs, inputs_depth, sequence_length)
    
    g      = SequenceToSequenceGenerator(g_inputs,
                                         inputs_depth,
                                         g_z, 
                                         input_sequence_length, 
                                         output_sequence_length,
                                         reverse_input=False)
    d_fake_inputs = tf.concat([g_inputs, g.output], axis=1)
    d_fake = NNDiscriminator(d_fake_inputs, inputs_depth, sequence_length, reuse=True)

    # The purpose of those is only to learn the probability in case of WGAN, LS-GAN and their families.
    d_real_prob = NNDiscriminator(d_inputs, inputs_depth, sequence_length, scope="prob")
    d_fake_prob = NNDiscriminator(d_fake_inputs, inputs_depth, sequence_length, reuse=True, scope="prob")

    # Skeleton specific loss
    g_prev = g_inputs[:, input_sequence_length-1:input_sequence_length, :, :]
    if output_sequence_length > 1:
        g_prev = tf.concat([g_prev, g.output[:, 0:output_sequence_length-1, :, :]], axis=1)
    g_next = g.output

    g_consistency_loss = tf.maximum(0.0001, tf.norm(g_next-g_prev, ord=2) / (minibatch_size*output_sequence_length))
    tf.summary.scalar("consistency_loss", g_consistency_loss)

    g_bone_loss = (nn.bone_loss(g_prev, g_next, body_inf) / (minibatch_size*output_sequence_length))
    tf.summary.scalar("bone_loss", g_bone_loss)

    # Gradient penalty
    def gradient_penalty():
        alpha = tf.random_uniform([], 0.0, 1.0)
        d_inputs_hat = alpha * d_inputs + (1 - alpha) * d_fake_inputs
        d_outputs_hat = NNDiscriminator(d_inputs_hat, inputs_depth, sequence_length, reuse=True).output
        gradients = tf.gradients(d_outputs_hat, d_inputs_hat)[0]
        gradients_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[2,3]))
        return tf.reduce_mean(tf.square(gradients_l2 - 1.))

    gradient_penalty_loss = 10.0 * gradient_penalty()
    tf.summary.scalar("gradient_penalty_loss", gradient_penalty_loss)

    # Discriminator and generative loss function.
    d_loss = tf.reduce_mean(d_fake.output - d_real.output) + gradient_penalty_loss
    g_gan_loss = -tf.reduce_mean(d_fake.output)

    d_loss_prob = -tf.reduce_mean(tf.log(d_real_prob.prob) + tf.log(1. - d_fake_prob.prob))

    d_loss += 0.001 * tf.add_n([tf.nn.l2_loss(p) for p in d_real.weights])
    tf.summary.scalar("discriminator_or_critic_loss", d_loss)
    tf.summary.scalar("gan_loss", g_gan_loss)

    g_loss = g_gan_loss + 0.001*g_consistency_loss + 0.01*g_bone_loss
    tf.summary.scalar("generator_loss", g_loss)

    d_loss_prob += 0.001 * tf.add_n([tf.nn.l2_loss(p) for p in d_real_prob.weights])
    tf.summary.scalar("discriminator_loss", d_loss_prob)

    # Optimizers
    d_op = tf.train.AdamOptimizer(learning_rate=d_lr).minimize(d_loss, var_list=d_real.parameters)
    g_op = tf.train.AdamOptimizer(learning_rate=g_lr).minimize(g_loss, var_list=g.parameters)
    d_op_prob = tf.train.AdamOptimizer(learning_rate=d_lr/2.0).minimize(d_loss_prob, var_list=d_real_prob.parameters)

    # tensorboard log
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(output_tensorboard_folder, graph=tf.get_default_graph())

    # Must be after the optimizer
    init_op = tf.global_variables_initializer()

    # Draws all z random vectors used in visualization
    z_rand_type = 'uniform'
    z_rand_params = {'low':-0.1, 'high':0.1, 'mean':0.0, 'std':0.2}
    z_data_p = []
    for _ in range(10):
        z_data_p.append(nn.generate_random(z_rand_type, z_rand_params, shape=[minibatch_size, z_size]))

    logging.info("Start training, training clip count {}.".format(train_data_reader.size()))
    g_best_loss = float('inf')
    g_best_epoch = -1
    g_best_pos_loss = float('inf')
    g_best_pos_epoch = -1
    g_best_prob = 0
    g_best_prob_epoch = -1

    d_losses = []
    g_losses = []
    d_losses_prob = []

    model_saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)

        d_loss_val = 0.
        g_loss_val = 0.
        d_loss_val_prob = 0.

        d_per_mb_iterations = 10
        g_per_mb_iterations = 2
        tensorboard_index = 0
        generate_skeletons = nn.generate_skeletons_with_prob(sess, g, d_real_prob, d_fake_prob, data_preprocessing, 
                                                             d_inputs, g_inputs, g_z)
        
        while epoch < max_epochs:
            train_data_reader.reset()

            # Training
            start_time = time.time()
            d_training_loss = 0.
            g_training_loss = 0.
            d_training_loss_prob = 0.
            d_is_sequence_probs = []

            k = 0
            while train_data_reader.has_more():
                input_data, _, current_batch_size, activities, subjects = train_data_reader.next_minibatch(minibatch_size)
                input_past_data = input_data[:, 0:input_sequence_length, :, :]

                if minibatch_size != current_batch_size:
                    continue

                if k == 0:
                    subject_id = subjects[0]
                    skeleton_data, d_is_sequence_probs = \
                        generate_skeletons(input_data, input_past_data, z_data_p)

                    skeleton2D.draw_to_file(skeleton_data, subject_id, os.path.join(output_folder, "pred_{}.png".format(epoch)))

                z_data = nn.generate_random(z_rand_type, z_rand_params, shape=[minibatch_size, z_size])
                for _ in range(int(d_per_mb_iterations-1)):
                    _, d_loss_val = sess.run([d_op, d_loss], feed_dict={d_inputs:input_data, g_inputs:input_past_data, g_z:z_data})
                
                _, d_loss_val_prob = sess.run([d_op_prob, d_loss_prob], feed_dict={d_inputs:input_data, g_inputs:input_past_data, g_z:z_data})

                for _ in range(int(g_per_mb_iterations-1)):
                    _, g_loss_val = sess.run([g_op, g_loss], feed_dict={g_inputs:input_past_data, d_inputs:input_data, g_z:z_data})
                
                summary, _, g_loss_val = sess.run([summary_op, g_op, g_loss], feed_dict={g_inputs:input_past_data, d_inputs:input_data, g_z:z_data})
                writer.add_summary(summary, tensorboard_index)

                d_training_loss += d_loss_val
                g_training_loss += g_loss_val
                d_training_loss_prob += d_loss_val_prob
                tensorboard_index += 1
                k += 1

            if k > 0:
                d_losses.append(d_training_loss / current_batch_size)
                g_losses.append(g_training_loss / current_batch_size)
                d_losses_prob.append(d_training_loss_prob / current_batch_size)                
            
            # Keep track of best epoch.
            if epoch > 20: # Ignore first couple of epochs.
                save_model_and_video = False
                prob_count = -1. # Don't count ground truth.
                for z_prob in d_is_sequence_probs:
                    if z_prob >= 0.5:
                        prob_count += 1.
                current_prob = prob_count / (len(d_is_sequence_probs) - 1)
                
                if (current_prob >= g_best_prob) and (current_prob > 0.):
                    save_model_and_video = True
                    g_best_prob = current_prob
                    g_best_prob_epoch = epoch
                
                if g_training_loss < g_best_loss:
                    save_model_and_video = True
                    g_best_loss = g_training_loss
                    g_best_epoch = epoch

                if (g_training_loss > 0) and (g_training_loss < g_best_pos_loss):
                    save_model_and_video = True
                    g_best_pos_loss = g_training_loss
                    g_best_pos_epoch = epoch

                # Save current model trained parameters and a video per z value.
                if save_model_and_video:
                    model_saver.save(sess, os.path.join(output_models_folder, "models"), global_step=epoch+1)
                    video_index = 0
                    if args.record_clip:
                        for sequence in skeleton_data:
                            skeleton2D.draw_to_video_file(sequence, os.path.join(output_videos_folder, "pred_{}_z{}.mp4".format(epoch, video_index)))
                            video_index += 1

            logging.info("Epoch {}: took {:.3f}s".format(epoch, time.time() - start_time))
            logging.info("  discriminator training loss:\t{:e}".format(d_training_loss))
            logging.info("  generative training loss:\t{:e}".format(g_training_loss))
            logging.info("  discriminator prob training loss:\t{:e}".format(d_training_loss_prob))
            logging.info("  is sequence: {}".format(d_is_sequence_probs))
            if epoch > 20:
                logging.info("  generative best loss:\t{:e}, for epoch {}".format(g_best_loss, g_best_epoch))
                logging.info("  generative best pos loss:\t{:e}, for epoch {}".format(g_best_pos_loss, g_best_pos_epoch))
                logging.info("  best motion prob:\t{:.1%}, for epoch {}".format(g_best_prob, g_best_prob_epoch))

            epoch += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train",
                        "--train_file",
                        type=str,
                        help="Provide the path of your train file.",
                        required=True)

    parser.add_argument("-test",
                        "--test_file",
                        type=str,
                        help="Provide the path of your test file.")

    parser.add_argument("-ccf",
                        "--camera_calibration_file",
                        type=str,
                        help="Provide the path to your camera calibration file.")

    parser.add_argument("-dnf",
                        "--data_normalization_file",
                        type=str,
                        help="Provide the path to your data normalization file. Which should contain mean and standard deviation of the input dataset.")

    parser.add_argument("-out",
                        "--output_folder",
                        type=str,
                        help="Provide the path of your output folder.",
                        required=True)

    parser.add_argument("-dataset",
                        "--dataset_name",
                        type=str,
                        help="Provide the name of the dataset (nturgbd or human36m).",
                        default="nturgbd")

    parser.add_argument("-epochs",
                        "--max_epochs",
                        type=int,
                        help="Maximum number of epochs (default 300).",
                        default=300)

    parser.add_argument("-clip",
                        "--record_clip", 
                        help="Record a video clip of the predicted action.",
                        action="store_true")

    args = parser.parse_args()
    main(args)
