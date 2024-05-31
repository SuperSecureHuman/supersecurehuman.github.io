---
title: "Basics of Transformers and Huggingface - Training"
excerpt: "A take on trying to help understand LLMs and Transformers - Now training them!"
tags:
    - deep learning
    - NLP
    - transformers

header:
    overlay_image: ""
    overlay_filter: 0.5
---

Now you know what transformer does, how it knows what it knows. We combine these both now. Let's train a model!

## What you need

### Dataset

The whole point of training something is to do something we want. Here, we assume that we the model to give good summaries, or even better, good summaries of scientific articles. No matter that is your task, you create an appropriate dataset for it.

### The model

Now that we have the data, we need to train the model. I'll be using Bart for this. Bart is an ~400M parameter model and is by ~Facebook~ Meta.

The model by itself is not standalone. It needs something called a tokenizer (will be explained in detail in later posts) which converts the natural language into numbers that computers understand.

### Some compute

While these models can be trained on your personal machine, chances are slim that most will be rocking an high performant PC that can crunch through these numbers. Thankfully, there is Google Colab and Kaggle (which is also by google) that provide free GPU that we can use to train our models.

### Lot of patience

Training time is directly propotional to the model size and the dataset size. The bigger the model, the longer it takes to train. The bigger the dataset, the longer it takes to train. The bigger the model and the dataset, the longer it takes to train. Also, one more thing is that, we need to make sure that the model we are using can be training on the GPU we have. For example, if you have a GPU with 12GB of VRAM, you can't train a 16GB model on it. You need to use a smaller model. Colab provides 16GB T4 GPUs, so we will work accordingly.

### Coffee

Most important

## What you do

We take dataset, we take model, we take compute, we take patience, we take coffee, we mix them all together and we get a trained model. Simple.

Ok, getting serious now. We need to do a few things before we can train our model.

### Install the libraries

Colab usually does have most of the libraries we need, but its just a good idea to be specific about what we need. We need to install the transformers library and the datasets library.

```bash
pip install transformers datasets
```

### Import the libraries, load the model, load the dataset

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn") # This is the tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn") # This is the model

dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]") # This is the train dataset
dataset_test = load_dataset("cnn_dailymail", "3.0.0", split="validation[:10%]") # This is the validation dataset

# Move the model to GPU for faster training
model = model.to("cuda")
```

### Check out the dataset, and initial model performance

```python
print(dataset[0]) # Print the first example in the dataset

# Check model perforamnce on the first example
input = tokenizer(dataset[0]["article"], return_tensors="pt").to('cuda') # Tokenize the input
output = model.generate(**input) # Generate the output
print(tokenizer.decode(output[0])) # Decode the output
```

Here is the input and output of the model

- Input - LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won\'t cast a spell on him. Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. "I don\'t plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month. "I don\'t think I\'ll be particularly extravagant. "The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," currently six places below his number one movie on the UK box office chart. Details of how he\'ll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "I\'ll definitely have some sort of party," he said in an interview. "Hopefully none of you will be reading about it." Radcliffe\'s earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. "People are always looking to say \'kid star goes off the rails,\'" he told reporters last month. "But I try very hard not to go that way because it would be too easy for them." His latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films.  Watch I-Reporter give her review of Potter\'s latest » . There is life beyond Potter, however. The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. He will also appear in "December Boys," an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer\'s "Equus." Meanwhile, he is braced for even closer media scrutiny now that he\'s legally an adult: "I just think I\'m going to be more sort of fair game," he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed.', 'highlights': "Harry Potter star Daniel Radcliffe gets £20M fortune as he turns 18 Monday .\nYoung actor says he has no plans to fritter his cash away .\nRadcliffe's earnings from first five Potter films have been held in trust fund.

- Output of model - Harry Potter star Daniel Radcliffe turns 18 on Monday. He gains access to a reported £20 million ($41.1 million) fortune. Radcliffe's earnings from the first five Potter films have been held in a trust fund. Details of how he'll mark his landmark birthday are under wraps.

### Preprocess the dataset

As noted earlier, the model needs the inputs to be tokenized. We need to tokenize the dataset. We also need to remove the examples that are too long for the model to handle. Thankfully, HF tokenizers does that!

```python
def preprocess_function(examples):
    inputs = [doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples["highlights"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

Now that we have defined the preprocessing function, we can apply it to the dataset.

```python
tokenized_ds = dataset.map(preprocess_function, batched=True)
tokenized_eval_df = dataset_test.map(preprocess_function, batched=True)
```

### Lets now get to training (almost)

We need to define a few things before we can train the model. We need to define the optimizer, the learning rate, the batch size, the number of epochs, and the evaluation metric.

```python
from transformers import DataCollatorForSeq2Seq 
import evaluate
import numpy as np

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model) # Handles all the data part to feed to model
rouge = evaluate.load("rouge") # Evaluation metric used in summarization tasks
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}
```

### Now we are training, for real!

```python
from transformers import  Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="my_awesome_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    eval_dataset=tokenized_eval_df,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

Now you may have your coffee, and wait for the model to train. It will take a while, so you can go and do something else. Once the model is trained, you can use it in the same way as before. Whilte training, the trainer outputs useful info, which will start to make sense as you train more models.
