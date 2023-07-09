---
title: "Unleashing the Power of AI in Data Analysis with Sketch"
excerpt: "Explore and Analyze Your Data with Sketch: In this blog post, we will explore Sketch - an AI-powered DataFrame assistant for Python that uses data sketching to quickly summarize large amounts of data. With Sketch, users can explore their data and receive code-writing prompts without the need for complex coding. "
tags:
    - pandas
header:
    overlay_image: "https://i.imgur.com/MsqyzCy.jpg"
    overlay_filter: 0.7
---
Have you heard about the latest buzz in the AI world? Everyone is talking about the power of AI and large language models to create cool interactive chatbots like ChatGPT. Now, you can use these models directly from a Python library within a Jupyter Notebook!

Enter Sketch, a new Python library that brings an AI coding assistant directly to Python and is super easy to use within Jupyter notebooks and IDEs. Sketch is aimed at making it easier for anyone to understand and explore data stored within a Pandas DataFrame without the need for additional plugins.

Sketch uses a powerful technique called data sketching, which is a type of approximation algorithm that can quickly summarize large amounts of data. With Sketch, you can ask for a summary of the data in your DataFrame and it will generate summary statistics such as mean, median, mode, standard deviation, and other statistical measures. Sketch then feeds this summarized data into a large language model to provide code-writing prompts that can be used to analyze the data.

Currently, Sketch summarizes the columns of data and provides these summary statistics as additional context to the code-writing prompt. However, in the future, Sketch hopes to use the sketches directly in custom-made "data + language" foundation models to get even more accurate results.

By using Sketch, you can quickly and easily explore your data without the need for complex coding. Sketch's use of data sketching allows for efficient summarization of large datasets, making it a valuable tool for any data analysis or data science project.

Sketch has three main functions - `ask`, `howto`, and `apply`. In this blog post, we'll be focusing on `ask` and `howto` functions. Unfortunately, we won't be discussing examples on the apply function as it requires the use of a custom OpenAI API key, and my free credits have run out :(

With Sketch's ask function, you can ask your DataFrame to perform operations such as finding the mean or finding the datatype of columns. Sketch will then return a code snippet that shows you how to do what you've asked.

On the other hand, with Sketch's howto function, you can ask Sketch to provide you with a code snippet that shows you how to perform a particular task. For example, you can ask Sketch to show you how to create a histogram or a scatter plot.

## Imports and Data

Importing sketch is pretty straightforward. You can install it using `pip install sketch`.

```python
import sketch
import pandas as pd
```

After Sketch has been imported, three new functions attached to the DataFrame object will become available to us. These are `df.ask()` , `df.howto()`, and `df.apply()`.

Some data is of course required to play around with. I am using the `wine` dataset from `sklearn` for this blog post.

```python
# Load wine data from sklearn
from sklearn.datasets import load_wine
wine = load_wine()

# Create a dataframe from the wine data
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Create a new column in the dataframe with the wine target
df['target'] = wine.target
```

Here is the loaded DataFrame

![The Loaded Wine Dataset](https://i.imgur.com/B35J7J0.png)

## `df.ask()`

One of the functions available in Sketch is the ability to ask the DataFrame to perform operations on the data, such as finding the mean, median, mode, standard deviation, variance, and other statistical measures. Users can also ask the DataFrame about the data types of the columns, which can help identify potential data type mismatches that could lead to errors in analysis.

```python
df.sketch.ask('What is the mean of alcohol?')
df.sketch.ask('Which columns have only integers?')
```

![Ask The DataFrame](https://i.imgur.com/W2nfdVE.png)

It can also give summary of the DataFrame.

```python
df.sketch.ask("Give me a summary of the data")
```

![Summary of the DataFrame](https://i.imgur.com/dwGXkci.png)

## `df.howto()`

One of the functions available in Sketch is the df.howto() function, which allows users to ask the DataFrame for a code snippet that shows how to perform a particular operation.

The code generated usually works with just a little modification, usually a great starting point. 

```python
df.sketch.howto("Get a plot so that I can find the outliers in 'alcohol'")
```

![HowTo in Sketch](https://i.imgur.com/13K3kOc.png)

Here is the generated code. Remember to replace the '_' with your DataFrame name.

```py
import matplotlib.pyplot as plt

# Get the 'alcohol' column from the dataframe
alcohol_data = df['alcohol']

# Plot a boxplot of the 'alcohol' column
plt.boxplot(alcohol_data)
plt.title('Alcohol Outliers')
plt.xlabel('Alcohol')
plt.ylabel('Values')
plt.show()
```

![The Plot](https://i.imgur.com/nRKgpMC.png)

## `df.apply()`

`df.apply()` is a function that allows users to apply a function to the DataFrame. This is an advanced prompt that generates data from the existing data. This is build on [LambdaPrompt](https://github.com/approximatelabs/lambdaprompt) which directly interfaces with the OpenAI API.

## Conclusion

In conclusion, with the rise of AI-powered tools like Sketch, the future of data analysis and data science is looking brighter than ever before. By automating complex data summarization tasks, Sketch makes it easier for users to explore their data and generate insights quickly and efficiently.

As AI technology continues to evolve and improve, we can expect to see even more powerful tools that will make data analysis and data science more accessible to people from all walks of life. With the help of these tools, we can unlock new insights and understandings that will drive innovation and change in countless industries.

So, whether you're a seasoned data scientist or just getting started with data analysis, it's a great time to be alive. With AI-powered tools like Sketch at our disposal, we have the power to unlock the full potential of our data and create a better future for ourselves and for generations to come. So, let's embrace the future of AI and data science and see what amazing things we can achieve!

## References

- [Sketch - PyPI](https://pypi.org/project/sketch/)
- [Sketch - GitHub](https://github.com/approximatelabs/sketch)
- [LambdaPrompt - GitHub](https://github.com/approximatelabs/lambdaprompt)
