---
title: "Distributed Training With Horovod on Tensorflow and PyTorch"
excerpt: "Horovod is a distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. It was developed by Uber AI Labs. Horovod is easy to use, and it provides performance optimizations for commonly used deep learning frameworks. In this post, I'll show you how to use Horovod to train a deep learning model on a cluster of machines."
tags:
    - deep learning
    - tensorflow
    - pytorch
    - horovod
    - cluster
---

Horovod is an open-source software framework for distributed deep-learning training. It was developed by Uber and is designed to make it easier and more efficient to train deep learning models across multiple GPUs and machines. Horovod is built on top of popular deep learning frameworks such as TensorFlow, Keras, and PyTorch, and supports various communication protocols such as MPI, NCCL, and Gloo. It uses techniques such as gradient averaging, message passing, and compression to minimize the communication overhead and maximize the scalability of the training process. Horovod is widely used in industry and academia, especially in large-scale deep-learning applications that require high performance and parallelism.

In this post, I'll show how you can convert your existing TensorFlow or PyTorch code to use Horovod and train your model on a cluster of machines.

## Installing Horovod

Detailed instructions on installing Horovod are available in my other post. Do check it out before proceeding to the next step.

[Horovod Setup](https://supersecurehuman.github.io/Horovod-Setup/)

## Keras Example
