---
title: "Serving Fastchat - Personal Journey"
excerpt: "Serving fastchat for people to experiment with various LLMs. This guide also incluides setting up Vllm to serve multiple models on a single GPU."
tags:
    - deep learning
    - NLP
    - transformers

header:
    overlay_image: "https://i.imgur.com/56wqrSe.png"
    overlay_filter: 0.5
---


FastChat is a tool for working with large chatbot models. It helps in setting up, running, and checking how well chatbots perform. Below, I'll explain how to get FastChat ready for use, especially focusing on using models (not training).



## Env setup

The system I am using contains 2xA100 80GB. This setup can handle models as big as 70 billion parameters at 16bit precision.

### Base

I choose to use `nvcr.io/nvidia/pytorch:24.01-py3` image for this for no spefic reason. Actually the reason was I already had downloaded the container.


### Installation

Create a new env and install python in it (going with 3.11). Do not install any other dependencies yet.

```bash
mamba create -n fastchat
mamba activate fastchat
mamba install python==3.11
```

#### Compiling VLLM from source

Since VLLM has tighter requirements than fastchat, we can first install VLLM, then install fastchat.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
# making sure to compile against local cuda
CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install .  # Takes a while....
pip install flash-attn --no-build-isolation
```

#### Installing fastchat

Now, we go bleeding edge! Install from source


```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
mamba install rust cmake
pip3 install -e ".[model_worker,webui]" -vv # verbose because u can see what is happening
```


## Using Fastchat

FastChat operates by connecting workers (the models) to a controller.


1. Launch controller

```bash
python3 -m fastchat.serve.controller
```

2. Launch worker(s)

You can run multiple models depending on your GPU capacity. There are options to restrict GPU usage per model, allowing you to load multiple models concurrently. For instance, a 7-billion-parameter model needs about 20GB of VRAM to run efficiently. Here's how to run a few models:


```bash
python3 -m fastchat.serve.vllm_worker \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --model-names llama3-8b-instruct
```
Note: VLLM's flags enable you to optimize the setup, including limiting VRAM usage per model. In this setup, your chosen models will remain loaded in VRAM and ready for use.


Pro tip: Use hf_transfer to download models faster than traditional methods. Make sure to cache the models before launching FastChat.


3. Serve the WebUI

```bash
python3 -m fastchat.serve.gradio_web_server
```

You should now have all the models you 'served' via the webui


## Experiments

### 1. Llama 3 8b + Phi 3 + Gemma 7b + DeciLM 7B + StableLM 1.6B

I'll be running these models on VLLM directly instead of using the fastchat VLLM worker. This allows me to export the metrics from each models. I can then register each of these VLLM models as openAI workers under fastchat.

My models for this experiment and my launch args. I would like to use only 1 of my GPU, because, I need to test how much I can squeeze of it!

```bash

# Llama 3 8B (8k)
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --device cuda --gpu-memory-utilization 0.25 --dtype bfloat16  --disable-log-requests --tensor-parallel-size=1 --trust-remote-code --enforce-eager --port 8001

# http://0.0.0.0:8001/v1

# Gemma 7b (8K)
python -m vllm.entrypoints.openai.api_server --model google/gemma-1.1-7b-it --device cuda --gpu-memory-utilization 0.27 --dtype bfloat16  --disable-log-requests --tensor-parallel-size=1 --trust-remote-code --enforce-eager --kv-cache-dtype fp8 --port 8002

# DeciLM 7B (8k)
python -m vllm.entrypoints.openai.api_server --model Deci/DeciLM-7B-instruct --device cuda --gpu-memory-utilization 0.23 --dtype bfloat16  --disable-log-requests --tensor-parallel-size=1 --trust-remote-code --enforce-eager --kv-cache-dtype fp8 --port 8003

# Phi 3 128k (18k)
python -m vllm.entrypoints.openai.api_server --model microsoft/Phi-3-mini-128k-instruct --device cuda --gpu-memory-utilization 0.17 --dtype bfloat16  --disable-log-requests --tensor-parallel-size=1 --trust-remote-code --enforce-eager --kv-cache-dtype fp8 --max-model-len 18000 --port 8004

# Stable LM 1.6B (4k)
python -m vllm.entrypoints.openai.api_server --model stabilityai/stablelm-2-1_6b-chat --device cuda --gpu-memory-utilization 0.07 --dtype float16  --disable-log-requests --tensor-parallel-size=1 --trust-remote-code --enforce-eager --kv-cache-dtype fp8 --port 8005
```

This is how my GPU looks like after loading these models...

![](https://i.imgur.com/3krX70C.png)


Following [this](https://github.com/lm-sys/FastChat/blob/main/docs/model_support.md#api-based-models), my fastchat setup would be

```json
{
"Llama 3 8B 8K":{
  "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
  "api_key": "Empty",
  "api_type": "openai",
  "api_base": "http://0.0.0.0:8001/v1/",
  "anony_only": false
},

"Gemma 7B 8K":{
  "model_name": "google/gemma-1.1-7b-it",
  "api_key": "Empty",
  "api_type": "openai",
  "api_base": "http://0.0.0.0:8002/v1/",
  "anony_only": false
},

"DeciLM 7B 8K":{
  "model_name": "Deci/DeciLM-7B-instruct",
  "api_key": "Empty",
  "api_type": "openai",
  "api_base": "http://0.0.0.0:8003/v1/",
  "anony_only": false
},

"Phi 3 18K":{
  "model_name": "microsoft/Phi-3-mini-128k-instruct",
  "api_key": "Empty",
  "api_type": "openai",
  "api_base": "http://0.0.0.0:8004/v1/",
  "anony_only": false
},

"StableLM 1.6B":{
  "model_name": "stabilityai/stablelm-2-1_6b-chat",
  "api_key": "Empty",
  "api_type": "openai",
  "api_base": "http://0.0.0.0:8005/v1/",
  "anony_only": false
}
}

```

Launch the webui (make sure to `pip install openai`)

```bash
python3 -m fastchat.serve.gradio_web_server --controller "" --share --register api_endpoints.json
```

Umm, it works!

![](https://i.imgur.com/1zldpGs.png)


### 2. Getting the arena mode - It has to infer on 2 LM at the same time

There seems to be some bug in arena battle mode. But side-by-side mode works as expected!

```bash
python3 -m fastchat.serve.gradio_web_server_multi  --controller "" --share --register api_endpoints.json
```

![](https://i.imgur.com/WBYg7Og.png)

## So whats the conclusion?

The purpose of this article was to offer a straightforward method for anyone seeking to identify the most suitable Large Language Model (LLM) for their specific use case. By deploying five models on a single GPU, I demonstrated a cost-effective approach to testing these models. In a future article, I plan to explore on a comprehensive, user-friendly UI to facilitate experimentation.