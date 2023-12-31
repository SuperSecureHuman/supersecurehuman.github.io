---
title: "Basics of Transformers and Huggingface - Inference"
excerpt: "A take on trying to help understand LLMs and Transformers - In a code first approach!"
tags:
    - deep learning

header:
    overlay_image: "<https://i.imgur.com/gIryYT6.png>"
    overlay_filter: 0.5
---


So, we hear these “LLMs (Large Language Models)” terms too much put around everywhere. Here is my attempt in providing a code first approach in exploring them, from nothing to something.
While math of it is really interesting, getting to see it in action is even more fun! Let's get started!

### What are LLMs?

A language model (LM) is a kind of AI model that produces text that appears like it was written by a human. Large Language Models, or LLMs, are essentially larger versions of these models with billions of parameters trained on a vast array of text data. The term 'large' refers to the size of the underlying AI model and the computer resources needed to train it.

LLMs predict the next word in a sentence using the context from the preceding words. This lets them generate unique and sensible sentences. As they learn from diverse text data covering a wide range of subjects and styles, their predictions become more accurate and widely applicable.

![An DALLE-3 Image for LLM](https://i.imgur.com/qYZeM5Q.jpg){: .align-center}

There are several kinds of LLMs. One example is OpenAI’s GPT-3, which is based on the Transformer model and uses a method called self-attention to determine the importance of each piece of input. Google’s BERT is another example which revolutionized language tasks by considering the context of a word from both sides.

Though LLMs are incredibly powerful, they have their drawbacks. They can unintentionally produce harmful or biased text and are sensitive to the way the input is phrased. Also, they lack understanding or awareness, so their outputs should be evaluated carefully. Ongoing research aims to make these models safer, more respectful of guidelines, and more helpful for more tasks.

[Here is a useful video on LLMs](https://www.youtube.com/watch?v=5sLYAQS9sWQ)

## What are Transformers?

Transformer architecture is a deep learning model used mainly in natural language processing. It was introduced to solve issues like complex cross dependencies and slow calculations in the paper "Attention is All You Need". It handles sequential data and uses an 'Attention' mechanism to decide the importance of different pieces of input. This architecture abandons recurrent mechanisms, which leads to a large increase in speed. Models like BERT, GPT-2, and T5 are built on Transformer architecture.

![Image](https://i.imgur.com/sFqUWR5.jpg){: .align-center}

Now that we understand the basics, let's understand how to use them!

## Introduction to Huggingface!

Huggingface, also known as HF, is a company that is paving the way in open-source AI model ecosystems. Regardless of your AI needs, HF has something to offer. They have created a range of libraries like transformers, diffusers, and gradio, which greatly simplify the creation and use of AI models. Their accelerate library makes it easy to scale up your AI training capability. Here's the link to check them out: **[Huggingface](https://huggingface.co/)**

## Show me the LLM in working (or LM for now)

HF’s transformers library makes it too easy to start using LLMs. Let’s now try out an Text Summary model

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""
print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))
>>> [{'summary_text': 'Liana Barrientos, 39, is charged with two counts of "offering a false instrument for filing in the first degree" In total, she has been married 10 times, with nine of her marriages occurring between 1999 and 2002. She is believed to still be married to four men.'}]
```

Thats it!

Behind the scenes, you have downloaded a model from huggingface hub, loaded it, took the article, “tokenized” it, passed it through the model and printed the output. Cool right!

## Read More

Remember that HF is not just for NLP, check their docs to see other cool stuff you can do with them!

[https://huggingface.co/tasks#:~:text=Natural Language Processing](https://huggingface.co/tasks#:~:text=Natural%20Language%20Processing)

<https://huggingface.co/docs/transformers/pipeline_tutorial>

<https://huggingface.co/docs/transformers/multilingual>

<https://huggingface.co/docs/transformers/task_summary>
