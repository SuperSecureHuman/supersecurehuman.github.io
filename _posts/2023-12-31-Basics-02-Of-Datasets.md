---
title: "Basics of Transformers and Huggingface - Datasets"
excerpt: "A take on trying to help understand LLMs and Transformers - Now the dataset!"
tags:
    - deep learning
    - NLP
    - transformers

header:
    overlay_image: "https://i.imgur.com/6ERmPRK.jpg"
    overlay_filter: 0.5
---

It's important to note that these AI Models, their ability to recognize patterns and behave in certain ways, all trace back to the "Dataset" they were trained on.

To put it simply, the datasets are like the textbooks from which these models learn, understand patterns, and form intuition about the inputs they receive. The quality of the dataset directly impacts the performance of the model - an excellent dataset will result in an excellent model. For example, if you need a model that summarizes large articles, but your dataset consists of only small articles, the model's performance on large articles will be poor.

![Image](https://i.imgur.com/6ERmPRK.jpg)

Creating a dataset that perfectly fits the task at hand is a complex art form. It requires numerous failed experiments, specific knowledge in the field, an understanding of the model, and more.

Because the dataset is so crucial to the function of Large Language Models, it's important to understand how datasets are stored, shared, and structured. Later, we will dive into more complex uses of datasets and provide tips on creating the perfect one for your task.

## Back to Hugging face! Datasets Library

The Hugging Face Datasets library offers easy access to and sharing of datasets for Audio, Computer Vision, and Natural Language Processing tasks.

Hugging Face's Datasets make it incredibly easy to structurally organize, share, host, and process datasets in the most efficient manner for AI tasks. Combined with the HF hub, it's a treasure trove of data for Machine Learning.

![Image](https://i.imgur.com/ktFQjbc.jpg)

In the previous post (Assuming you saw my previous post - pls see, gib support), we explored an example model for summarization. Those with sharp eyes would have noticed the "Summarize" keyword. Give it another look! Remeber this, will be useful later on.

## Looking at Datasets @HF

![Image](https://i.imgur.com/d8ylGX6.png)

You have a lot of tasks at hand - From Vision to being a future visonary :)

![Image](https://i.imgur.com/72BY4eG.png)

## Why should I care?

There comes a time when you need to be organized and adhere to certain standards in your work. Using the Datasets framework provided by Hugging Face does much of the hard work for you, offers great flexibility, and integrates seamlessly with all other Hugging Face tools. Getting familiar with managing datasets would be an incredibly useful skill to acquire as you navigate the world of AI models.

## What's Next?

Your next step is to select a dataset from Hugging Face that piques your interest or pertains to your field, and get a feel for how to prepare datasets for AI tasks. In the following sections, we will cover how to build your own model suited to your specific needs. Let's keep exploring!
We will look into using these datasets to make our onw models in the next post!

## Read more

[https://huggingface.co/docs/datasets/](https://huggingface.co/docs/datasets/)
