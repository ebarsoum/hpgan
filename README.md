# HP-GAN: Probabilistic 3D human motion prediction via GAN
This repo implements an updated version of the code behind HP-GAN paper (https://arxiv.org/abs/1711.09561).

## Dependencies

* Tensorflow 1.8
* h5py
* Pillow
* numpy
* moviepy

## Dataset
We used the 3D skeleton data from NTU-RGBD and Human 3.6m dataset to train HP-GAN:
* NTU-RGBD: http://rose1.ntu.edu.sg/datasets/actionrecognition.asp
* Human 3.6m: http://vision.imar.ro/human3.6m/description.php

For Human 3.6m, we used the h5 format and parsing code from https://github.com/una-dinosauria/3d-pose-baseline

### Prepare the data
The reader take a CSV file that contain the actual path to the skeleton file, activity ID and subject ID. 

To generate those CSV files call the following for ntu dataset:
```
python split_ntu_data.py -i <path>/nturgb+d_skeletons/ -o <path to your output>
```
And `split_h36m_data.py` for human3.6m. Feel free to update the script for your need.

## Train
For training simply call `train_hpgan.py`, the needed parameters are documented.

Here an example:
```
python train_gan.py -train <path>/train_map.csv -out <path>/results -epochs 10000 -dataset human36m -ccf <path>/cameras.h5 -dnf <path>/data_statistics.h5
```
Here part of the spewed output during training:
```
Epoch 9985: took 12.548s
  discriminator training loss:	6.805864e+00
  generative training loss:	3.923600e+01
  discriminator prob training loss:	4.793965e+01
  discriminator category training loss:	5.555983e-02
  is sequence: [0.9999447, 0.0040896684, 0.004284089, 0.0003978344, 0.034581296, 0.010833021, 0.035049524, 0.013850356, 0.10408278, 0.051567502, 0.027076172]
  generative best loss:	3.736670e+01, for epoch 9930
  generative best pos loss:	3.736670e+01, for epoch 9930
  best motion prob:	90.0%, for epoch 9978
Epoch 9986: took 12.150s
  discriminator training loss:	6.597710e+00
  generative training loss:	4.036868e+01
  discriminator prob training loss:	4.712248e+01
  discriminator category training loss:	2.735711e-02
  is sequence: [0.9881322, 0.00035418675, 0.00031916617, 9.020698e-05, 0.0053915004, 0.0031668958, 0.0021527985, 0.004252481, 0.005165949, 0.026035802, 0.002512865]
  generative best loss:	3.736670e+01, for epoch 9930
  generative best pos loss:	3.736670e+01, for epoch 9930
  best motion prob:	90.0%, for epoch 9978
```
## Results
First raw is the ground truth, input is the first 10 poses and the network predict 20 poses. Each row after the first one correspond to a new `z` value.

![Alt text](/img/ntu_pred.png)

## Citation
If you use the provided code or part of it in your research, please cite the following:

```
@article{BarsoumCVPRW2018,
  author = {Emad Barsoum and John Kender and Zicheng Liu},
  title = {{HP-GAN:} Probabilistic 3D human motion prediction via {GAN}},
  journal = {CoRR},
  year = {2017},
}
```
