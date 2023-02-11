---
title: "Optimizing Deep Learning Models for Inference with Speedster"
excerpt: "Nebuly-AI's Speedster is a tool that can optimize deep learning models for inference on CPUs and GPUs."
tags:
    - pytorch
    - deep learning
    - tensorflow
    - onnx
    - speedster
    - optimization
    - inference
---

Now you have your super advanced fabulous model ready. Now what?

Deep learning in production doesn't stop with your jupyter notebook. There are equally complex challenges involved in deploying your model to end users.

All that matters to the end user is the result and the speed of whatever they want to get from your deployment.

Several strategies can be used to optimize the latency of deep learning models, including:

1. Model architecture: Choosing a model architecture optimized for latency, such as a compact network with fewer layers, can reduce the time it takes to make inferences.

2. Quantization: Converting the model from floating-point to fixed-point representation can significantly reduce the computational cost of inferencing, leading to lower latency.

3. Pruning: Removing redundant or less essential neurons and connections from the model can reduce the complexity of the model and lower latency.

4. Batch inference: Running inferences on a batch of inputs rather than a single input can take advantage of parallelism and reduce overall latency.

5. Hardware acceleration: Using specialized hardware, such as GPUs or TPUs, can significantly speed up inferencing and reduce latency.

6. Model compression: Techniques such as weight sharing, parameter quantization, and knowledge distillation can be used to reduce the size of the model, which can lead to lower latency.

It is important to note that while optimizing latency is essential, it is not the only consideration in deep learning model design. Accurately and effectively addressing the problem being solved and maintaining an appropriate balance between latency, accuracy, and complexity are crucial factors in successful model design.

Every hardware manufacturer has a set of libraries to aid in this aspect. Nvidia has TensorRT, and Intel has OpenVINO. Both of these were made to optimize the model for fast inference times.

Apart from these, some models perform better in other frameworks. For example, a Tensorflow model might have lower latency in ONNX. When you try to include all these combinations of frameworks and optimization methods each provides, manually doing this needs a steep learning curve and manually going through each process.

## Welcome to Speedster from Nebuly-AI

Speedster is a framework designed to streamline the optimization of deep learning models. It automates the application of state-of-the-art optimization techniques to improve the performance of deep learning models for inferencing. Speedster provides a unified interface that incorporates the best practices and techniques from popular deep learning frameworks like TensorFlow, PyTorch, and ONNX. The goal of Speedster is to make the process of optimizing deep learning models for high-performance inference more accessible and straightforward for practitioners and researchers.

With Speedster, deep-learning practitioners and researchers can focus on the core aspects of their work rather than spending time and effort on optimizing their models. Speedster automates the optimization process, making it easier to create high-performance models that meet the specific needs of different hardware configurations.

The optimization techniques used by Speedster are designed to reduce latency and improve inference performance. These techniques include quantization, pruning, and hardware-specific optimizations.

## Key Concepts

![Key Concepts](https://i.imgur.com/0FvP1Dq.png)

Speedster is shaped around four building blocks and leverages a modular design to foster scalability and integration of new acceleration components across the software-to-hardware stack.

* Converter: converts the input model from its original framework to the framework backends supported by Speedster, namely PyTorch, ONNX, and TensorFlow. This allows the Compressor and Compiler modules to apply any optimization technique to the model.

* Compressor: applies various compression techniques to the model, such as pruning, knowledge distillation, or quantization-aware training.

* Compiler: converts the compressed models to the intermediate representation (IR) of the supported deep learning compilers. The compilers apply post-training quantization techniques and graph optimizations to produce compiled binary files.

* Inference Learner: takes the best-performing compiled model and converts it back into the same interface as the original input model.

## Claims of Speedster

From their [GitHub Page](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/speedster)

![Speedster Claims](https://i.imgur.com/VBM1QGu.png)

My Experience with Speedster

I've run tests on speedster locally with CPU and on Google Colab for its GPU.

CPU used  - Intel i7 11800H
Model used - Resnet50

Result

```py
[Speedster results on 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz]
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Metric      ┃ Original Model   ┃ Optimized Model   ┃ Improvement   ┃
┣━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━┫
┃ backend     ┃ PYTORCH          ┃ DeepSparse        ┃               ┃
┃ latency     ┃ 0.0444 sec/batch ┃ 0.0155 sec/batch  ┃ 2.87x         ┃
┃ throughput  ┃ 22.55 data/sec   ┃ 64.72 data/sec    ┃ 2.87x         ┃
┃ model size  ┃ 102.55 MB        ┃ 102.06 MB         ┃ 0%            ┃
┃ metric drop ┃                  ┃ 0                 ┃               ┃
┃ techniques  ┃                  ┃ fp32              ┃               ┃
┗━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━┛
```

According to the results, the model was optimized by 2.87x, using DeepSparse as the backend.

Comparing the original model and the optimized model with Nebullvm's benchmarking tool, I got the following results:

![Benchmark-CPU](https://i.imgur.com/z0T4zlz.png)

Original model - 48ms per image
Optimized model - 20ms per image

That's a 2.4x improvement in latency. Good.

Furthermore, you can apply quantization, pruning, etc., which further improves the latency at the cost of accuracy.

Here are the results of allowing Speedster with some degree of accuracy drop:

```py
[Speedster results on 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz]
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Metric      ┃ Original Model   ┃ Optimized Model   ┃ Improvement   ┃
┣━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━┫
┃ backend     ┃ PYTORCH          ┃ OpenVINO          ┃               ┃
┃ latency     ┃ 0.0409 sec/batch ┃ 0.0053 sec/batch  ┃ 7.71x         ┃
┃ throughput  ┃ 24.44 data/sec   ┃ 188.47 data/sec   ┃ 7.71x         ┃
┃ model size  ┃ 102.55 MB        ┃ 25.98 MB          ┃ -74%          ┃
┃ metric drop ┃                  ┃ 0                 ┃               ┃
┃ techniques  ┃                  ┃ int8              ┃               ┃
┗━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━┛
```

The model was optimized by 7.71x, using OpenVINO as the backend.

Benchmarking the same model with Nebullvm's benchmarking tool, I got the following results:

![Image](https://i.imgur.com/pD5pAxc.png)

That's a 6x improvement in latency!

Here are some more results:

| Model | Device | Original Latency | Optimized Latency | Improvement | Method | Metric Drop |
| --- | --- | --- | --- | --- | --- | --- |
| Resnet50 | CPU | 48ms | 20ms | 2.4x | DeepSparse | 0% |
| Resnet50 | CPU | 42ms | 6ms | 6x | OpenVINO(Quantised INT8) | 0% |
| Resnet50 | Tesla T4 | 13ms | 4ms | 3.25x | TensorRT | 0% |
| Resnet50 | Tesla T4 | 13ms | 1ms | 13x | TensorRT(Quantised INT8) | 0% |
| BERT | CPU | 60ms | 57ms | 1.05x | ONNXRuntime | 0% |
| BERT | CPU | 57ms | 41ms | 1.4x | ONNXRuntime(Quantised INT8) | 0.4% |
| BERT | Tesla T4 | 9.3ms | 7ms | 1.33x | ONNXRuntime(fp16) | 0% |
| BERT | Tesla T4 | 9.3ms | 3.9ms | 2.4x | ONNXRuntime(fp16) | 0.8% |

The results speaks for themselves :)

## How does it do whatever it does?

Generally, you would try each optimization step one by one and store all the results.

Speedster takes this a step further by creating a pipeline for any model to various optimized models, resulting in a set of pipelines your model can go through. Then your model is optimized with each pipeline, and the best model is stored.

The workflow of Speedster consists of 3 steps:

1. Select - Input your model in your desired framework, and give your preferences about the time taken to optimize and max accuracy drop.

2. Search - The library automatically optimizes your model with various SOTA techniques that are compatible with your hardware.

3. Serve - The library chooses the best model returned from the search and returned the accelerated version of your model.

## Installation and Usage

### Installation

First, install speedster:

```bash
pip install speedster
```

Then install all the compliers so that Speedster can use them:

```bash
python -m nebullvm.installers.auto_installer --compilers all
```

### Usage

Using Speedster is very simple.

```py

# load your model
model = ...

# provide some input data
input_data = [((torch.randn(1, 3, 256, 256), ), torch.tensor([0])) for _ in range(100)]

# Run Speedster optimization
from speedster import optimize_model
optimized_model = optimize_model(
    model, 
    input_data=input_data, 
    optimization_time="constrained",
    metric_drop_ths=0.05
)
```

Thats it! You can now use the optimized model.

Further documentation - [Speedster Docs](https://docs.nebuly.com/Speedster/).

## Conclusion

Speedster is a valuable tool for deep-learning practitioners and researchers looking to optimize their models for high-performance inference. By automating the optimization process and incorporating the best practices and techniques from popular deep-learning frameworks, Speedster makes it easier and more efficient to create high-performance models for specific hardware configurations.

## Further Exploring

* [Nebullvm Docs](https://docs.nebuly.com/)

* [Nebullvm Github](https://github.com/nebuly-ai/nebullvm)
