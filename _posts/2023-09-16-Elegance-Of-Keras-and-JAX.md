---
title: "Elegance of Keras Core and JAX combo"
excerpt: "Combining Keras and JAX as a backend, makes JAX to be meant for Humans"
tags:
    - deep learning

header:
    overlay_image: "https://i.imgur.com/FywH5yZ.png"
    overlay_filter: 0.5
---

So, I gave a talk on JAX with Keras for Keras Community day, Coimbatore. I am putting the essence of my talk here.

The codes and the presentation can be found here - <https://github.com/SuperSecureHuman/keras-day>

## Keras Core: The Basics

Keras Core is a newer version of Keras. It can work with different systems like JAX, TensorFlow, and PyTorch. This means developers can easily change the system they're using without starting from scratch.

One cool thing about Keras Core is that it can run a training loop that works everywhere. Sometimes, it can even make things run up to 20% faster. And if you have data from different places like PyTorch or TensorFlow, Keras Core can handle it.

If you know how to use Keras, you can use Keras Core in the same way. Plus, if you want to, you can get deeper into the system you're using. Keras Core lets you do that.

## JAX: A Quick Look

Google and DeepMind made JAX. It's like a better version of NumPy. You can use it on different machines, whether they are CPUs, GPUs, or TPUs. One of the best things about JAX is how it deals with automatic changes, which is very important for high-quality machine learning work.

Even though JAX is new, many people are starting to use it. They like the results they get, and it's getting better all the time.

## Why JAX is Different

Machine Learning needs a lot of computer power, especially for things like matrix work and figuring out gradients. JAX is really good at this.

JAX is fast when it does matrix work, which a lot of algorithms use. But it's not just about being fast. JAX makes hard things like gradient calculations easy. If you have a function in JAX, you can quickly give it the ability to compute gradients. Because JAX is both fast and easy to use, many see it as a top choice for heavy computer work.

## Keras: A Simple History

Keras began with a simple idea: "Deep Learning for Everyone." It wanted to make deep learning easy for all people, no matter their skill.

Started in 2015, Keras was different because it could work with many systems from the start. But things change fast in technology. Some systems became less popular. Then, a big thing happened: Keras became part of TensorFlow and became known as `tf.keras`.

But Keras still had its old spirit. So, Keras Core was made. This new version went back to the old way, working with many systems. Today, Keras Core works on its own, without needing TensorFlow. It has come back to its original idea.

## JAX: How It Works

JAX is built for fast computing. Here's how it does it:

1. **Fast NumPy**: JAX makes the usual NumPy stuff faster. It changes them to work better with fast machines, making calculations quick and right.

2. **Quick Matrix Work**: Matrices are important for deep learning. JAX does this work really quickly. It uses special tools and tricks for different machines to save time.

3. **Using `jax.jit`**: This tool in JAX changes Python work into machine language. This makes things run much faster, especially if you do the same thing many times.

4. **Special Tools - `jax.pmap` and `jax.vmap`**: These tools help do many tasks at once. `jax.pmap` splits the work between devices like TPUs and GPUs. `jax.vmap` makes the tasks run in a line. Using them together makes things very fast.

5. **Working with XLA and MLIR**: XLA helps with math work for fast machines. MLIR is a way to show many types of computer tasks. Both are very important for JAX. They make sure JAX works really well with specific machines.

6. **Easy Math with AutoDiff**: One great thing about JAX is how it does math. It can find out how things change on its own, making hard calculations easy. This means fewer mistakes and better results.

## Stateless Nature of JAX: A Blessing for Distribution, JIT, and Parallelization

JAX's stateless architecture stands as one of its most defining and advantageous features, particularly when discussing distribution, Just-In-Time (JIT) compilation, and parallelization.

1. **Distribution**: In the realm of distributed computing, managing and synchronizing state across multiple devices or nodes can be a significant challenge. A stateless design, like JAX's, simplifies this. Without the need to constantly synchronize state or manage shared memory across devices, distributing computations becomes far more straightforward. Each computation becomes an isolated event, free from external dependencies, ensuring that distributed tasks can be executed without entangling complexities.

2. **JIT Compilation**: The JIT compiler's job is to translate high-level code into machine-level instructions that can be executed efficiently on a target hardware. In the presence of mutable state, the compiler must make conservative assumptions to ensure correctness, which can limit optimization opportunities. JAX's stateless nature ensures that functions are pure, meaning their outputs are solely determined by their inputs. This purity allows the `jax.jit` compiler to make aggressive optimizations without worrying about unforeseen side-effects or external state changes, leading to significantly faster code execution.

3. **Parallelization**: When parallelizing computations, one of the most challenging aspects is managing concurrent access to shared state. Such access can lead to race conditions, deadlocks, or other synchronization issues. JAX's stateless design inherently sidesteps these challenges. Since each operation is self-contained and doesn't rely on an external state, parallelizing them using tools like `jax.pmap` or `jax.vmap` becomes a seamless endeavor. This design choice ensures that functions can be distributed across multiple cores or devices without the typical hazards of parallel programming.

## JAX vs. C with MPI: A Data Scientist’s Perspective

For data scientists, the choice of tools can greatly influence their productivity, the efficiency of their algorithms, and ultimately, the impact of their work. When comparing JAX to the combination of C with the Message Passing Interface (MPI), there are clear advantages in favor of JAX, even if it comes with its own learning curve.

1. **Abstraction and Simplicity**: JAX provides a higher level of abstraction compared to C with MPI. This means that data scientists can focus more on algorithm design and less on the intricacies of parallelization, memory management, and inter-process communication. While C with MPI offers granular control over these aspects, it also demands a deep understanding of parallel programming, which might not be the primary expertise of many data scientists.

2. **Automatic Differentiation**: One of JAX's standout features is its capability for automatic differentiation. In the realm of machine learning, where gradient computations are ubiquitous, this feature alone can save vast amounts of time and reduce potential sources of error.

3. **Optimized Matrix Operations**: For data scientists, especially those working on deep learning tasks, optimized matrix operations are crucial. While C with MPI can be fine-tuned for performance, JAX inherently provides accelerated matrix operations, removing the onus of manual optimization.

4. **Statelessness**: As previously discussed, JAX's stateless nature simplifies many tasks like JIT compilation, distribution, and parallelization. In contrast, managing state in C with MPI can be cumbersome and error-prone.

5. **Learning Curve**: While JAX offers numerous benefits, it's not without its challenges. The shift from traditional imperative programming paradigms to JAX’s more functional approach can be daunting. However, this learning curve is often outweighed by the benefits, especially when considering the steep learning curve and intricacies involved in mastering C with MPI for high-performance parallel computations.

## Keras Core's Integration with JAX: A Symbiotic Fusion

The amalgamation of Keras Core with JAX forms a powerful alliance that brings together the best of both worlds. This union makes deep learning more intuitive while retaining the computational prowess JAX offers.

1. **Unified Framework with Extended Support**: Keras Core, known for its user-friendly interface and adaptability, has now embraced JAX as one of its backends. This means practitioners can continue to define models with the familiar elegance of Keras while capitalizing on the computational speed and efficiency of JAX.

2. **Harnessing JAX's Benefits Within Keras**: With this integration, when you define a model in Keras, you're not just getting the simplicity of Keras; you're also reaping all the advantages JAX brings to the table. From automatic differentiation to lightning-fast matrix operations, the marriage of Keras and JAX ensures that your models are both easy to define and quick to train.

3. **Simplified Multi-Device Distribution**: One of the challenges with core JAX is managing computations across multiple devices. With Keras Core’s integration, this process is streamlined. Distributing your deep learning models across GPUs or TPUs becomes more intuitive, removing much of the manual overhead associated with setting up multi-device computations in core JAX.

## Conclusion

The interplay between user-centric design and powerful computation has often been a balancing act in the world of deep learning. While some tools have sacrificed one for the other, Keras Core and JAX stand as exemplars of how the two can coexist harmoniously.

Keras, with its motto of "Deep Learning for Humans," has consistently strived to simplify the complexity of neural networks, making them more accessible to a wider audience. Its evolution, particularly the reintroduction of its multi-backend nature, shows a commitment to versatility without compromising on its core philosophy.

JAX, meanwhile, is a testament to what is achievable when there's a focus on raw computational power, optimization, and flexibility. Its stateless design, ability to leverage hardware accelerators, and seamless parallelization are features that make it a formidable force in the realm of deep learning.

Their integration is a watershed moment. It embodies the potential of bringing together the best of both worlds: the user-friendliness of Keras and the computational might of JAX. This symbiotic fusion is a boon for the deep learning community, making advanced techniques and tools more attainable.
