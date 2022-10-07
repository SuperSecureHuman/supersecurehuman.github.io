---
title: "U Net architure for Image Segmentation"
tags: 
    - pytorch
    - image
    - deep learning
---
## Introduction

Image segmentation is the process of labeling each and every pixel in an image. Its one of the major tasks in computer vision, majorly helping in tasks such as bio medical image segmentation, self driving cars and more.

In this article, we will go on exploring U-Net architure. Along with this, we will also be implementing the same using PyTorch, with help of carvana data set from kaggle.

## U-Net Architure

U-Net architure is a completely built on convolution layers. One thing with all conv layers is that, its independent of input image resolution. We can train this model on say 224x224 images and then use it on 512x512 images. This is one of the major advantages of using only conv layers.

The architure is as follows:

![U-Net](https://i.imgur.com/i1zgbgu.png)

It consists of 3 parts -

* Encoder

* Bottleneck

* Decoder

![Arch Breakdown](https://i.imgur.com/5ZCNdwQ.png)

### Encoder

Encoder path is the path where we are going to downsample the image. We will be using maxpooling layers to downsample the image. We will be using 2 conv layers for each downsampling. The first conv layer will have 64 filters and the second will have 128 filters and so onn (Of course the filter size is our wish, but here I am following the paper). We will be using 3x3 kernel size for all conv layers. In the above image, each conv layer in the encoder path is followed by a ReLU activation function.

### Bottleneck

This is the tranisition part of the architure. Here the input image is passed to the decoder block, where the image is upsampled and the segmentation map is achieveed.

### Decoder

Decoder path is the path where we are going to upsample the image. This part of the network completely consists of transpose conv layers. In order to preserve the features from the input, we have skip connection from the same shaped layers from the encoder path. 

