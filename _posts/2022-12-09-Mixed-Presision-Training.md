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

### Floating Point Precision

According to IEEE standards, there are various levels of precision that can be used to represent floating point numbers, ranging from binary16 (half-precision) to binary256 (octuple-precision). Here the number after binary represents the number of bits available to represent the number.

For a long time, deep learning has used FP32 (usually reffered as single-precision) to represent all the parameters.

## Mixed Precision Training

Note: Here, I'll be trying to provide a jist of this topic from the implementation part of the paper. For more details, please refer to the paper.

The idea is to store the weights, activations and gradients in FP16. But doing so, will narrow down the range than single precision. To overcome this, the following is done:

1. Maintaining a single precision copy of the weights which accumulates the gradients after every iteration (this copy is rounded off to FP16 during forward and backward pass).
2. A method called `loss-scaling` to preserve gradients with very small magnitudes.
3. Finally, half precision arithmetic that accumlates the gradients in FP32, that is rounded off to FP16 before storing to memory.


### FP32 Master Copy of Weights

To match the accuracy of the single precision model, an FP32 is stored, and parameters are updated during optimizer step. 

In each iteration, an FP16 copy of the master weights is used to compute the forward and backward pass, halving the storage and bandwidth requrements during the training.

From nvidia's paper

![Half Pres Flow](https://i.imgur.com/a0LPdAO.jpg)

Reason to maintain a master copy of the weights is to use them, incase the gradient becomes too small to be represented in FP16 (basically, it becomes 0 in FP16, but in FP32, it has some value), and when this small value is multiplied with the learning rate, it becomes even smaller. In order to account this, master copy is updated in such cases at single precision. Ignoring this might adversely affect the model performace.

Even though maintaing a master copy would mean that we need more memory, the impact on overall memory usage, which is usually dominated by saving of activations for reuse duing backward pass. Processing these in FP32 would mean that we have roughly halved the memory reqirements.


### Scaling Loss

In FP16, the representable range of values are 10 power [-14, 15]. In practice, the gradient values tend to be smaller than that, mainly when we consider the effect of learning rate too. In the example that nvidia took in their paper (training multibox SSD detector network)

![Gradient Histogram](https://i.imgur.com/sASSXwc.png)


Here, most of the FP16 range was mostly unused. Scaling up the gradients, can preserve these values, which otherwise will be 0 when converted into FP16. Some network will diverge if the lower gradients are ignored/zeroed out. 

One way to handle this, is to scale the loss into FP16 range during forward pass. By doing this the resultant backward pass, will be scaled by the same factor. By doing this, there is no need of extra opertaions applied to gradients and maintains the relevant gradient values from becomming zero. After all this, gradients must be unscaled to preserve the FP32 master copy. Its recommended to do the unscaling just after the backward pass, before any gradient clipping or any gradient related functions.

We have to be careful when choosing the scaling factor, to avoid overflow during back-propagation - this will result in infinities and NaNs. One option to skip the updates when there is overflow and move on to next iteration.

### Arthimetic Precision

Most of the arthimetic operations that happen in neural nets fall under these 3 categories:

1. Dot Product - To maintain model accuracy, nvidia found that FP16 vector dot-product accumulates the partial products into FP32. Now this is important to maintain the model accuracy. After these operations, the final result is saved into memory in FP16. Newer generation Nvidia GPUs have support to multiply in FP16, and accumulate the result in FP32 or FP16.

2. Reduction Operations - Operations that include sums across elements of vectors. These operations read the values in FP16, and does the operation in FP32. These are not slowed down, because of the available hadware acclerations.

3. Pointwise Operations - Non-linear stuff, element wise matrix products, etc. These can be done in FP16/FP32.

## Normal Training with Pytorch

```python
import torch
from tqdm import tqdm
from torchvision import datasets, transforms

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=False)
model.fc = torch.nn.Linear(2048, 10)
model = model.cuda()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1,6):
    model.train()
    with tqdm(trainloader, unit="batch") as tepoch:
        for data, target in tepoch:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())
```

Output of the training loop:

```
100%|██████████| 782/782 [00:54<00:00, 14.22batch/s, loss=2.05]
100%|██████████| 782/782 [00:55<00:00, 14.18batch/s, loss=1.47]
100%|██████████| 782/782 [00:55<00:00, 14.15batch/s, loss=1.52]
100%|██████████| 782/782 [00:55<00:00, 14.13batch/s, loss=1.76]
100%|██████████| 782/782 [00:55<00:00, 14.12batch/s, loss=1.42]
```


## Using Torch amp

Pytorch has automatic mixed precision support, which can be used to train the model in FP16. This is done by using the `torch.cuda.amp` module. This module provides convenience methods for mixed precision and autocasting. 

```python
scaler = torch.cuda.amp.GradScaler() # Gradient scaler for amp (Mixed Precision)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 6):
    model.train()
    with tqdm(trainloader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(): # Automatic Mixed Precision
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward() # Scale the loss
            scaler.step(optimizer) # Unscales the gradients of optimizer's assigned params in-place
            scaler.update() # Updates the scale for next iteration
            tepoch.set_postfix(loss=loss.item())
```

After making the changes, the output of the training loop is:

```
Epoch 1: 100%|██████████| 782/782 [00:47<00:00, 16.57batch/s, loss=2.53]
Epoch 2: 100%|██████████| 782/782 [00:47<00:00, 16.57batch/s, loss=1.76]
Epoch 3: 100%|██████████| 782/782 [00:46<00:00, 16.68batch/s, loss=2.08]
Epoch 4: 100%|██████████| 782/782 [00:46<00:00, 16.77batch/s, loss=1.22]
Epoch 5: 100%|██████████| 782/782 [00:46<00:00, 16.79batch/s, loss=1.2] 
```

There is approx ~20% speedup on this simple model.

### Note

As of me writing this post, I hit a issue when using amp. I fixed it by editing the file - apex/amp/utils.py

```diff
- if cached_x.grad_fn.next_functions[1][0].variable is not x:
# into this
+ if cached_x.grad_fn.next_functions[0][0].variable is not x:
```

I am not sure why I got this. I'll update if I get any heads up on this

## Conclusion

You can find the notebook [here](https://github.com/SuperSecureHuman/ML-Experiments/blob/main/Half-Precision-Training/Pytorch_amp_post.ipynb)

Here we took a look at mixed precision training, and trained a simple CIFAR 10 model with torch's amp scaler. Later on, I will make a post on Nvidia Apex, trying out its various features, which is not available on torch's amp yet. See yaa!

## References

1. [Nvidia's Mixed Precision Training Paper](https://arxiv.org/abs/1710.03740)
2. [Pytorch's Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
3. [Pytorch's Mixed Precision Training Examples](https://pytorch.org/docs/stable/notes/amp_examples.html)