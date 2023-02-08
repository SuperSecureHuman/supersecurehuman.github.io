---

title: "Setting up Horovod on cluster"
excerpt: "Setting and running Horovod on a PBS managed cluster"
tags:
    - pytorch
    - deep learning
    - horovod
    - cluster
    - tensorflow
---

I have access to a cluster with five nodes, each with a V100 GPUs. It uses PBS for job scheduling. Here is how I set up Horovod on the cluster.

Other options for multi-worker training include using PyTorch's DistributedDataParallel (DDP)/TorchRun, Tensorflow/Keras, and DeepSpeed. I'll Cover those in a separate post and link them here.

This post only deals with installing and checking if Horovod is working. I'll cover the actual training in a separate post.

## 0. Initial ENV setup

Make sure to have pytorch (for CUDA version 11.6) in a conda environment. Everything will be in the conda environment.

Python Version Tested = 3.8.10

You have to load CUDA 10.1 first then CUDA 11.6 for tensorflow. This is to ensure that Tensorflow can find CuBlasLT from CUDA 10.1. If you load CUDA 11.6 first, then Tensorflow will use CuBlasLT from CUDA 11.6, which is not compatible Tensorflow.

Evem in the training phase, you have to load the modules in the given order, otherwise, you'll get an error.

```bash
module load cuda10.1/toolkit/10.1.243
module load cuda11.6/toolkit/11.6.2
module load cudnn8.4/8.4

export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=TRACE
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/home/user/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/home/user/include
```

The following flags can be ignored

```bash
export NCCL_DEBUG=TRACE
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

I recommend maintaining `lib` and `include` folders in your home directory and adding all the custom stuff there.

## 1. Get NCCL

Use NCCL Version 2.16.2.1

The Nvidia website states that it is for CUDA 11, but the official build for CUDA 11.6 has stub library loading issues, which will have to be solved by manually compiling NCCL, which may cause problems. Currently, 2.16.2.1 works without any problems for CUDA 11.6.2

Save the NCCL libraries in the home's `lib` and `include` folders.

## 2. Horovod Install

If you wish to build for Tensorflow, then make sure you have g++ version 8 or higher for Tensorflow version 2.10 and higher. If you have a lower version of g++, you can stick with an older version of Tensorflow. I'll be using Tensorflow 2.8 here.

Additionally, install protobuf 3.20.0 with `pip install protobuf==3.20.0`

For pytorch, the requirement is g++ version 5 or higher.

Mxnet - Ignored

MPI needs some additional checking because it kept throwing errors.

GLOO backend will be used for communications

```bash
HOROVOD_NCCL_INCLUDE=/home/user/include/ HOROVOD_NCCL_LIB=/home/user/lib/ HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITHOUT_MPI=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir horovod
```

Make sure to modify the flags based on your requirements. For example, if you want to use only pytorch and not with Tensorflow, then replace the `HOROVOD_WITH_TENSORFLOW=1` flag with `HOROVOD_WITHOUT_TENSORFLOW=1`.

More info on the flags used here can be found in the official docs

[Horovod - Install](https://github.com/horovod/horovod/blob/master/docs/install.rst#environment-variables)

## 3. Checking the install

`horovodrun --check-build`

```bash
horovodrun --check-build
[I debug.cpp:49] [c10d] The debug level is set to DETAIL.
[I debug.cpp:49] [c10d] The debug level is set to DETAIL.
[I debug.cpp:49] [c10d] The debug level is set to DETAIL.
[I debug.cpp:49] [c10d] The debug level is set to DETAIL.
[I debug.cpp:49] [c10d] The debug level is set to DETAIL.
[I debug.cpp:49] [c10d] The debug level is set to DETAIL.
Horovod v0.27.0:

Available Frameworks:
    [X] TensorFlow
    [X] PyTorch
    [ ] MXNet

Available Controllers:
    [ ] MPI
    [X] Gloo

Available Tensor Operations:
    [X] NCCL
    [ ] DDL
    [ ] CCL
    [ ] MPI
    [X] Gloo
```

## 4. Testing a Trial Run

Take this example - [https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py](https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py)

Running this on a single node

```bash
horovodrun --verbose  --gloo --gloo-timeout-seconds=90  --network-interface="ib0"  -np {N} -H localhost:N  NCCL_SOCKET_IFNAME=ib0  python  script.py {args}

```

Here, N = numGpu

Running this on multiple nodes

```bash
horovodrun --verbose  --gloo --gloo-timeout-seconds=90  --network-interface="ib0"  -np {N} -H node01:n1,node02:n2,node03:n3 python  script.py {args}
```

n1,n2,n3 = num of GPUs in each node

N = n1+n2+n3

## 5. Horovod Flags

`--verbose` - Verbose output

`--gloo` - Use GLOO backend

`--gloo-timeout-seconds` - Timeout for GLOO backend (default 30, it kept timing out for me, so I increased it to 90)

`--network-interface` - Network interface to use (Using ib0 here, but you can use eth0 as well)

`-H` - Hosts (node01:4,node02:4,node03:4)

`-np` - Number of processes (Total Number of GPUs)

## 6. PBS

Running a horovod job on PBS is similar to mpi.

Horovod relies on having passwordless SSH into the worker nodes.

If you try SSH into the other nodes, you might not be able to get shell access because of the way PBS works.

However, when you request a node for a job and the node gets allocated to you, you can SSH into the allocated node.

Here is a sample PBS file for horovod

```bash
# !/bin/bash -x

# PBS -l mem=16gb
# PBS -l nodes=node01:ppn=8+node03:ppn=8
# PBS -q workq

# EXECUTION SEQUENCE

cd $PBS_O_WORKDIR

# module purge
# module load cuda10.1/toolkit/10.1.243
# module load cuda11.6
# module load cudnn8.4

# activate the conda environment

source /home/user/miniconda3/bin/activate
conda activate env

export NCCL_SOCKET_IFNAME=ib0
export TERM=xterm-xfree86
export NCCL_DEBUG=TRACE
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/home/user/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/home/user/include

horovodrun --verbose  --gloo --gloo-timeout-seconds=90  --network-interface="ib0"  -np 2 -H node01:1,node03:1  python pytorch_mnist.py 2>&1 | tee -a log.gloo.txt

```

Here, I have piped the output and the error to a file.

The training script used is from official horovod examples.

Pytorch - <https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py>

Tensorflow - <https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_keras_mnist.py>

## 7. Output

Since the NCCL was set to TRACE, the output is very verbose.

We can see the stdout and stderr of each process, from each node.

```bash
Filtering local host names.
Remote host found: node03
Checking ssh on all remote hosts.
SSH was successful into all the remote hosts.
[I debug.cpp:49] [c10d] The debug level is set to DETAIL.
.
.
. Horovod SSH log (this step where horovod does ssh into nodes to run the script)
.
.
[0]<stderr>:[I debug.cpp:49] [c10d] The debug level is set to DETAIL.
[1]<stderr>:[I debug.cpp:49] [c10d] The debug level is set to DETAIL.
[0]<stdout>:node01:150770:150790 [0] NCCL INFO Bootstrap : Using ib0:10.149.0.1<0>
[0]<stdout>:node01:150770:150790 [0] NCCL INFO NET/Plugin : No plugin found (libnccl-net.so), using internal implementation
[0]<stdout>:node01:150770:150790 [0] NCCL INFO cudaDriverVersion 11060
[0]<stdout>:NCCL version 2.16.2+cuda11.0
[1]<stdout>:node03:34537:34681 [0] NCCL INFO cudaDriverVersion 11060
[1]<stdout>:node03:34537:34681 [0] NCCL INFO Bootstrap : Using ib0:10.149.0.3<0>
[0]<stdout>:node01:150770:150790 [0] NCCL INFO NET/IB : Using [0]mlx4_0:1/IB [1]mlx4_0:2/IB ; OOB ib0:10.149.0.1<0>
[0]<stdout>:node01:150770:150790 [0] NCCL INFO Using network IB
[1]<stdout>:node03:34537:34681 [0] NCCL INFO NET/Plugin : No plugin found (libnccl-net.so), using internal implementation
[1]<stdout>:node03:34537:34681 [0] NCCL INFO NET/IB : Using [0]mlx4_0:1/IB [1]mlx4_0:2/IB ; OOB ib0:10.149.0.3<0>
[1]<stdout>:node03:34537:34681 [0] NCCL INFO Using network IB
[0]<stdout>:node01:150770:150790 [0] NCCL INFO Setting affinity for GPU 0 to 550000
[1]<stdout>:node03:34537:34681 [0] NCCL INFO Setting affinity for GPU 0 to 05555555
[0]<stdout>:node01:150770:150790 [0] NCCL INFO Channel 00/02 :    0   1
[0]<stdout>:node01:150770:150790 [0] NCCL INFO Channel 01/02 :    0   1
[0]<stdout>:node01:150770:150790 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] -1/-1/-1->0->1
[0]<stdout>:node01:150770:150790 [0] NCCL INFO P2P Chunksize set to 131072
[1]<stdout>:node03:34537:34681 [0] NCCL INFO Trees [0] -1/-1/-1->1->0 [1] 0/-1/-1->1->-1
[1]<stdout>:node03:34537:34681 [0] NCCL INFO P2P Chunksize set to 131072
[0]<stdout>:node01:150770:150790 [0] NCCL INFO Channel 00/0 : 1[3b000] -> 0[3b000] [receive] via NET/IB/1
[1]<stdout>:node03:34537:34681 [0] NCCL INFO Channel 00/0 : 0[3b000] -> 1[3b000] [receive] via NET/IB/1
[0]<stdout>:node01:150770:150790 [0] NCCL INFO Channel 01/0 : 1[3b000] -> 0[3b000] [receive] via NET/IB/1
[1]<stdout>:node03:34537:34681 [0] NCCL INFO Channel 01/0 : 0[3b000] -> 1[3b000] [receive] via NET/IB/1
[0]<stdout>:node01:150770:150790 [0] NCCL INFO Channel 00/0 : 0[3b000] -> 1[3b000] [send] via NET/IB/1
[1]<stdout>:node03:34537:34681 [0] NCCL INFO Channel 00/0 : 1[3b000] -> 0[3b000] [send] via NET/IB/1
[0]<stdout>:node01:150770:150790 [0] NCCL INFO Channel 01/0 : 0[3b000] -> 1[3b000] [send] via NET/IB/1
[1]<stdout>:node03:34537:34681 [0] NCCL INFO Channel 01/0 : 1[3b000] -> 0[3b000] [send] via NET/IB/1
[0]<stdout>:node01:150770:150790 [0] NCCL INFO Connected all rings
[0]<stdout>:node01:150770:150790 [0] NCCL INFO Connected all trees
[0]<stdout>:node01:150770:150790 [0] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
[0]<stdout>:node01:150770:150790 [0] NCCL INFO 2 coll channels, 2 p2p channels, 2 p2p channels per peer
[1]<stdout>:node03:34537:34681 [0] NCCL INFO Connected all rings
[1]<stdout>:node03:34537:34681 [0] NCCL INFO Connected all trees
[1]<stdout>:node03:34537:34681 [0] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
[1]<stdout>:node03:34537:34681 [0] NCCL INFO 2 coll channels, 2 p2p channels, 2 p2p channels per peer
[0]<stdout>:node01:150770:150790 [0] NCCL INFO comm 0x2aab7000d010 rank 0 nranks 2 cudaDev 0 busId 3b000 commId 0x25baa10b1e55271b - Init COMPLETE
[1]<stdout>:node03:34537:34681 [0] NCCL INFO comm 0x2aab7000d010 rank 1 nranks 2 cudaDev 0 busId 3b000 commId 0x25baa10b1e55271b - Init COMPLETE
[1]<stderr>:/home/user/superSecureHuman/horovod/pytorch_mnist.py:67: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
[1]<stderr>:  return F.log_softmax(x)
[0]<stderr>:/home/user/superSecureHuman/horovod/pytorch_mnist.py:67: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
[0]<stderr>:  return F.log_softmax(x)
[0]<stdout>:Train Epoch: 1 [0/30000 (0%)] Loss: 2.312604
[1]<stdout>:Train Epoch: 1 [0/30000 (0%)] Loss: 2.291532
[1]<stdout>:Train Epoch: 1 [640/30000 (2%)] Loss: 2.315194
   :
   :
   : Remaining Epochs
   :
   :
[1]<stdout>:Train Epoch: 1 [29440/30000 (98%)] Loss: 0.444777
[0]<stdout>:Train Epoch: 1 [29440/30000 (98%)] Loss: 0.295054
[0]<stderr>:/home/user/miniconda3/envs/env/lib/python3.8/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
[0]<stderr>:  warnings.warn(warning.format(ret))
[1]<stderr>:/home/user/miniconda3/envs/env/lib/python3.8/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
[1]<stderr>:  warnings.warn(warning.format(ret))
[0]<stderr>:/home/user/superSecureHuman/horovod/pytorch_mnist.py:120: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
[0]<stderr>:  tensor = torch.tensor(val)
[1]<stderr>:/home/user/superSecureHuman/horovod/pytorch_mnist.py:120: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
[1]<stderr>:  tensor = torch.tensor(val)
[0]<stdout>:
[0]<stdout>:Test set: Average loss: 0.2124, Accuracy: 93.70%
[0]<stdout>:
   :
   :
   : Remaining Epochs
   :
   :
[1]<stdout>:Train Epoch: 10 [29440/30000 (98%)] Loss: 0.093691
[0]<stdout>:
[0]<stdout>:Test set: Average loss: 0.0538, Accuracy: 98.26%
[0]<stdout>:
[0]<stdout>:node01:150770:150790 [0] NCCL INFO comm 0x2aab7000d010 rank 0 nranks 2 cudaDev 0 busId 3b000 - Destroy COMPLETE
[1]<stdout>:node03:34537:34681 [0] NCCL INFO comm 0x2aab7000d010 rank 1 nranks 2 cudaDev 0 busId 3b000 - Destroy COMPLETE
```
