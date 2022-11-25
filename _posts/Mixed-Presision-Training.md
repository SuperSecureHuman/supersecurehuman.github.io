---
title: "Mixed Precision Training"
tags: 
    - pytorch
    - tensorflow
    - deep learning
---


## Introduction

These days models go deeper and deeper. And it has also been prover that deeper models tend to perform better. But larger the model is, more resources it needs to train. In order to come across this, a technique called mixed precision is used. In a nutshell, it casts some parts of the network into lower precision, while keeping the rest in higher precision. This allows us to train larger models with less memory and compute resources.


## Note on Mixed Precision

Group of researchers from Nvidia released a [paper](https://arxiv.org/pdf/1710.03740.pdf) showing how to reduce memory usage during training. 

```
We introduce methodology for training deep neural networks using half-precision floating point numbers, without losing model accuracy or having to modify hyperparameters. This nearly halves memory requirements and, on recent GPUs, speeds up arithmetic.
```


### What is Mixed Precision?

