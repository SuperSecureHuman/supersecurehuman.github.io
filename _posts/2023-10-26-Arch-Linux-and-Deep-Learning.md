---
title: "Arch Linux is the Best for Deep Learning (for intermediate users)"
excerpt: "Arch linux makes it better to manage deep learning system, and understand the system better."
tags:
    - deep learning

header:
    overlay_image: "https://i.imgur.com/FywH5yZ.png"
    overlay_filter: 0.5
---

You know what? It's been more than 2 years since I jumped into the Arch-based world, using it mainly for deep learning. I said goodbye to Ubuntu and its siblings and found my happy place in Arch.

![I Use Arch BTW](https://i.imgur.com/HN6EPJr.png){: .align-center}

## The Deep Learning Dilemma

When you start diving deeper into the world of deep learning, managing packages, updating libraries, and maintaining a clean system can devolve into a complex endeavor. What's worse is that your operating system (OS) often hides these complexities from you, which can lead to a wild goose chase on the internet for a solution.

![CUDA to Could I.....](https://i.imgur.com/coKMTIa.png){: .align-center}

Updating TensorFlow may suddenly require a CUDA update. This typically means visiting NVIDIA's website, downloading the local installer, and crossing your fingers nothing breaks during the installation. Manually updating NVIDIA drivers can make things even more complicated.

![All is Well](https://i.imgur.com/J63csdH.png){: .align-center}

Alright, so you've leveled up and learned to deploy your models like a champ. But wait, there's more! You hear all the cool kids on the block talking about this swanky thing called Docker. So, naturally, you dive in, learn everything you can, and before you know it - bam! You're a pro at deploying models in Docker containers!

But hold on! You might think that using Docker would make your code work on any host, like a charm. But think again, my friend! What if - and I mean, just what if - you run into an OS-level dependency error right inside that shiny Docker container of yours? How do you go on resolving it. Thats where experience in Arch truly helps!

## Why Arch?

When you use Arch, the Arch way, you learn how the system components go with each other.

For instance, an average Google Colab/Ubuntu user may not even know that there is something called an envoironment path, where binaries need to be placed to be run. Or `LD_LIBRARY_PATH` where shared paths of shared libraries are stored. Most folks don't need to bother with these stuff, but hey - we're a different breed, aren't we? 😉

![Is it even possible you ask?](https://i.imgur.com/rKMeVnL.png){: .align-center}

Guess what happens when you're an Arch user? You get to pick and choose your own driver versions, learn how binaries and libraries interact together, and get your hands dirty with compilations. What's the prize, you ask?

I manage multiple CUDA versions! You get better at compiling and positioning things just right!

![I was once blind, but no longer](https://i.imgur.com/w8PoTMc.png){: .align-center}

## Major takeaways form me, using Arch

### Managing Multiple CUDA versions

There are times when I tangle with libraries that chat up CUDA directly, and guess what? They don't always play nice with the same version. Sure, they might claim to be compatible, but I like to stick with the version the library was built and tested against - call me picky!

So I installed each CUDA version in its own special folder and give it a brand-new home at `/usr/local/cuda`. Then, I set up a symlink for the version I need (and that definitely includes the matching cuDNN version as well).

Now, thanks to always having CUDA in my path, switching between versions is just a one-liner for me!

### TensorRT

In case of TensorRT, the CUDA + TensorRT + Pytorch/Tensorflow verison matching became very strict. Having a switchable envoironment makes it again easy to even try out the bleeding edge features, while having the confidence of rolling back to what you know worked best.

![The production fear of TensorRT](https://i.imgur.com/l7bi6zQ.png){: .align-center}

### Caffe

Don't call me an outdated grandfather for even speaking about this library. I wanted a pretrained model from a research paper, which used Caffe, and weights were shared, but in Caffe. The part which made it worse, is that it used custom layers. What made it even worse is that it used older version of OpenCV, Protobuf and related libraries. Now trying to get this up and running on an ubuntu based system is complicated hell. First of all, you will need to go to an older version of ubuntu for these things to be installed right (even on an docker, you need to go to older versions of ubuntu base image). Then hope that your `apt install` installs the packages which are compitatible with the custom versions of packge you are looking for.

![Image](https://i.imgur.com/OXmOqbv.png){: .align-center}

## Has it helped me outside my own PC?

Now, after using arch, and setting up the system in the way I want, I know exactly where and what stuff is placed. It helps me understand error that originate due to the system config. It helps me understand the env variables that affect a certain library.

In case of Arch, I can setup a seperate shell envroinment to do any sort experiments, without even breaking my main system. Now, I know what exactly goes into building a piece of code work, I can easily package this with a barebones docker container, and I can tell for sure that it wil work!

I have learnt about low level optimized packages like OneDNN/OneAPI, Different Memory Allocation libraries and much more. Its purely because of Arch, I was able to understand how stuff works.

![You will choose the red side, once you get used to Arch](https://i.imgur.com/RpHjvZP.png){: .align-center}

![Back to my work, byeeee](https://i.imgur.com/5UMuu5s.png){: .align-center}