---
title: "Stuff I learnt doing an end to end project"
excerpt: "Deep Learning Lessons: Insights from an End-to-End Project. Gain valuable tips from my personal experiences in deep learning. Discover the power of thorough logging, structured scripting, and effective checkpoints. Ensure fairness in datasets and optimize for success. Embark on your own deep learning journey armed with these invaluable insights."
tags:
    - deep learning
    - projects
header:
    overlay_image: "https://i.imgur.com/qTjzvpH.png"
    overlay_filter: 0.5
---

Imagine this scenario: I find myself deeply immersed in an incredibly fascinating research project, fully immersing myself in the mind-boggling realm of deep learning. Can you believe it? I've trained a deep learning model well over a hundred times! It's mind-blowing, isn't it? But let me tell you, it wasn't a walk in the park by any means. I had to experiment with an abundance of hyperparameters, more than you can imagine, and explore various architectures like a passionate scientist on a quest.

Take a look at this snapshot I captured from my Weights & Biases Dashboard. It offers a glimpse into the excitement and intensity of the journey: ![Screenshot from Weights & Biases Dashboard](https://i.imgur.com/RQJtVuu.png)

However, amidst this rollercoaster ride, I stumbled upon some invaluable workflow tips that have truly revolutionized my experience. These little gems of wisdom have become a game-changer, streamlining my life during these exhilarating times. Believe me, they possess the potential to elevate your research endeavors into a thrilling adventure of discovery. So fasten your seatbelts, my friends, and prepare to enhance your deep learning skills with these remarkable tricks!

## Tip 1 - Log Everything: Keeping Track of Your Deep Learning Journey

First things first: log everything! And I mean every single detail that's worth noting during your training sessions. It's a total game-changer, I promise!

So, how do you log like a pro? Well, there are some fantastic tools out there that make it super easy. Let me introduce you to a few favorites: TensorBoard, Weights and Biases, and MLflow. These bad boys will become your new best friends in no time!

Personally, I used TensorBoard to keep track of my model's progress. It allowed me to see how my model's parameters evolved over time in a visual way. It's like having a map of your model's journey, with all the twists, turns, and surprises.

But wait, there's more! When it came to logging metrics, dataset samples, and predictions, I relied on Weights and Biases. This awesome tool made it a breeze to log and track all the important numbers. And the best part? It let me show off some eye-catching samples and predictions from my dataset. Your research buddies will be seriously impressed!

And let's not forget about MLflow, another superstar in the logging game. With MLflow, I could effortlessly track my experiments, manage their settings, and keep everything organized. It's like having a personal assistant dedicated to keeping your deep learning projects in order.

By logging everything using these cool tools, you'll never have to worry about losing important information again. It's like leaving a trail of breadcrumbs that leads you straight to success! So embrace the power of logging, my friends, and get ready for a well-documented and thrilling research journey.

![My Project's Runs](https://i.imgur.com/ospuuad.png)

## Tip 2 - Scripting is the Key: Mastering Your Deep Learning Pipeline

Alright, my fellow deep learning enthusiasts, let's move on to our second tip, and it's truly a game-changer: scripting is the key! If you want to take complete control of your deep learning pipeline and unlock its full potential, pay close attention.

Here's the deal: rather than cramming all your code into a single, overwhelming file, it's time to break it down into smaller, more manageable pieces. Think of it as assembling a well-structured team for deep learning, where each file handles a specific task in your pipeline.

But wait, there's more! Once you've organized your code into these modular files, it's time to create a central script that brings them all together. Picture this script as the conductor of your deep learning orchestra, effectively coordinating the entire process.

With this unified script, you gain the power to control essential aspects of your training. How, you may wonder? It's all thanks to command line parameters. By incorporating command line flags into your script, you open up a world of possibilities. You can effortlessly adjust hyperparameters, switch between different models, and make crucial decisions on the fly.

In my project, I had seperate files for dataset loader, model builder, callback functions, utility scripts to hold training and evaluation loops, checkpoint manager, and main entry.

Consider the freedom and flexibility this approach brings to your research! No longer will you need to tediously modify your code each time you want to experiment with new hyperparameters or models. With just a few command line arguments, you can boldly venture into unexplored territories and explore a range of options without breaking a sweat.

This scripting approach not only streamlines your workflow but also promotes collaboration. Your fellow researchers can easily join your project, execute the script with their preferred configurations, and contribute to the collective knowledge.

## Tip 3 - Checkpoint Everything: Safeguarding Your Deep Learning Journey

Alright, my esteemed colleagues in the realm of deep learning, let's delve into our third tip that guarantees your efforts won't vanish into the void: *checkpoint everything*! Trust me, you'll appreciate the value of these invaluable checkpoints down the line.

When training your models, it's vital to save checkpoints at strategic intervals. Consider these checkpoints as snapshots of your model's progress, capturing its state at specific epochs. Not only do these checkpoints serve as backups, but they also open up a multitude of possibilities.

So, what's the plan? Here's my recommendation: save two types of checkpoints. First, preserve the checkpoint of the *best performing* model. This snapshot captures the model's state when it achieved its highest performance during training. It's like preserving a golden moment in your deep learning journey.

Secondly, save a checkpoint at the *final epoch* of training. This allows you to pick up where you left off if you need to continue training or fine-tune your model in the future.

When it comes to managing and tracking these checkpoints, you have some excellent options at your disposal. For smaller model sizes, consider uploading them to hosted services like Weights and Biases or utilize MLflow for efficient tracking. These platforms offer a centralized and convenient approach to store, visualize, and compare your checkpoints.

![Checkpointing in Action [Keras]](https://i.imgur.com/61mmRlx.png)

Another handy option is to maintain a single TensorBoard that displays your training progress and performance metrics. Combine it with model uploads to platforms like Hugging Face, and you have a powerful combination for sharing and reproducibility.

A vital lesson I've learned is the importance of naming your checkpoint files properly. Choose a naming convention that includes essential information such as the model name, learning rate, optimizer, weight decay, dataset name and ID (which I will discuss further in the next tip), image size, and accuracy. Customize the components of the file name according to your preference, ensuring easy filtering and identification of the precise model needed for reproducing or analyzing specific results.

By diligently checkpointing, organizing, and appropriately naming your models, you guarantee that your deep learning journey remains accessible, reproducible, and filled with thrilling possibilities. So, my friends, safeguard your hard work, and let your checkpoints pave the way for deep learning success!

## Tip 4 - Dataset: Building a Solid Foundation for Fair Evaluation

Alright, my fellow data enthusiasts, let's dive into our fourth tip, which is crucial for maintaining fairness and reproducibility when working with datasets: *consistently use a random seed*! Trust me, it's the secret ingredient to ensure a level playing field during your test train split.

When dividing your dataset into training and testing subsets, it's vital to consistently apply the same random seed. This means using the same seed value each time you split the data, ensuring that the same data points end up in the same sets. By doing this, you eliminate potential biases that may arise from different random splits and create a fair evaluation environment.

Now, let's discuss situations where you need to combine different data sources to create a comprehensive training set. It's crucial to establish clear naming conventions for each dataset you create. These names should accurately reflect the source, type, or other relevant information about the data. By logging these names and associated details, you establish a robust system that enables you to reproduce your dataset precisely and track its origins effectively.

With consistent and meaningful dataset naming and logging, you establish a solid foundation for reproducibility. Whether you revisit your research or collaborate with others, you can confidently recreate the exact dataset you used before, ensuring consistent results and avoiding any confusion or misinterpretation.

So, my friends, remember the importance of maintaining a consistent random seed and creating meaningful dataset names. By doing so, you not only ensure fair evaluation but also build a strong framework for transparent and reproducible research. Let's keep our datasets organized and pave the way for data-driven discoveries!

## Tip 5 - Find the Heaviest Model: Optimizing Hyperparameters for Limited GPU Memory

Alright, my fellow GPU enthusiasts, let's dive into our fifth tip, specially designed to help you navigate the challenges of limited GPU memory: *identify the most demanding model*! Not all of us have access to GPUs with abundant memory capacities, but fret not, for we have a strategy to optimize your hyperparameters and conquer this obstacle.

When working with GPUs that have limited memory, it's essential to identify the model that requires the most memory. This model pushes the limits of your GPU's capacity and serves as a benchmark for optimization. Once you've identified this heavyweight contender, it's time to find the optimal batch size that allows it to fit comfortably within your GPU memory.

Now, here's the trick: once you determine the ideal batch size for your most demanding model, stick with it for all your experiments. Yes, you heard it right! Maintain a consistent batch size throughout your hyperparameter tuning process. By doing so, you ensure a fair and consistent comparison of different configurations, as they are all evaluated under the same memory constraints.

![Batchsize](https://i.imgur.com/UcyhC0X.jpg)

Why is this important? Well, if you were to vary the batch size for each experiment, you would introduce an additional variable that affects the model's performance. It becomes difficult to isolate the impact of other hyperparameters when the batch size keeps changing. By keeping the batch size consistent, you can confidently attribute any performance differences to the specific hyperparameters you are tuning.

So, when faced with limited GPU memory, identify that most demanding model, determine the optimal batch size, and stick with it throughout your experiments. This approach allows for fair comparisons and ensures that your hyperparameter tuning is conducted on a level playing field. Conquer the memory limitations and let your models shine, even on GPUs with more modest capacities!

## Final Tip - Embrace the Journey and Share Your Knowledge

As we come to the end of this deep learning expedition, I want to share one final tip that goes beyond technicalities and speaks to the essence of your journey: *embrace the journey and share your knowledge*!

Deep learning is a dynamic field, constantly evolving with each stride you take. Embrace the challenges, victories, and even the setbacks. Each experience holds valuable lessons that shape you as a researcher and practitioner.

But don't keep those lessons to yourself! Share your insights, discoveries, and tips with the community. Whether it's through blog posts, tutorials, contributing to open-source projects, or engaging in discussions, your knowledge has the power to inspire and empower others.

Collaboration and knowledge-sharing are the driving forces behind the advancement of deep learning. So be generous with your wisdom, celebrate your successes, and don't hesitate to seek help when needed. Remember, the journey is just as important as the destination.

So, my fellow deep learning enthusiasts, armed with your newfound tips and tricks, embark on your adventure. Embrace every moment, learn from each step, and make your mark on the ever-changing landscape of deep learning. Together, we can push the boundaries of what's possible and shape a future driven by AI innovation. Happy deep learning!

## References

Links to ~~stolen~~ borrowed images:

<https://towardsdatascience.com/checkpointing-deep-learning-models-in-keras-a652570b8de6>
<https://towardsdatascience.com/how-to-break-gpu-memory-boundaries-even-with-large-batch-sizes-7a9c27a400ce>
