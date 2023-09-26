---
title: "Elegance of Keras Core and JAX combo"
excerpt: "Combining Keras and JAX as a backend, makes JAX to be meant for Humans"
tags:
    - deep learning

header:
    overlay_image: "https://i.imgur.com/FywH5yZ.png"
    overlay_filter: 0.5
---

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

## Choosing JAX or C with MPI: What's Best for Data Work?

People who work with data need good tools. When looking at JAX and C with MPI, JAX has some clear good points. But there are things to learn, too.

1. **Keeping Things Simple**: JAX makes things simple. With C and MPI, you need to know a lot about making many machines work together. But with JAX, you can focus on your main work and let JAX handle the rest.

2. **Doing Math Easily**: JAX can figure out changes in numbers on its own. This is great for machine learning where you need this a lot.

3. **Fast Number Work**: People working with deep learning need to do math fast. JAX does this quickly, while with C and MPI, you might have to do more work to get the same speed.

4. **Not Remembering Too Much**: JAX doesn’t need to remember a lot, making some tasks simpler. But with C and MPI, you might have to manage a lot of details.

5. **Learning New Things**: Even though JAX has many good points, it's a bit different. Some people might need time to learn it. But compared to learning C with MPI, which can be hard, many find JAX easier.

## Keras Core and JAX Working Together: The Best of Both

When Keras Core and JAX come together, it's like two good friends teaming up. They both bring something special, making deep learning easy and fast.

1. **Working Together Well**: Keras Core is easy to use, and now with JAX's power, it can work even better. People can make models easily and get the speed of JAX.

2. **Getting All the Good Stuff**: When you make a model in Keras, you get the simple side of Keras and all the fast, smart work from JAX.

3. **Using Many Devices Easily**: Sometimes, using many GPUs or TPUs with JAX can be a bit tricky. But with Keras Core, it's much easier. This means you can make your models work on many devices without too much trouble.

## Conclusion

Absolutely, the interplay between user-centric design and powerful computation has often been a balancing act in the world of deep learning. While some tools have sacrificed one for the other, Keras Core and JAX stand as exemplars of how the two can coexist harmoniously.

Keras, with its mantra of "Deep Learning for Humans," has consistently strived to simplify the complexity of neural networks, making them more accessible to a wider audience. Its evolution, particularly the reintroduction of its multi-backend nature, shows a commitment to versatility without compromising on its core philosophy.

JAX, meanwhile, is a testament to what is achievable when there's a focus on raw computational power, optimization, and flexibility. Its stateless design, ability to leverage hardware accelerators, and seamless parallelization are features that make it a formidable force in the realm of deep learning.

Their integration is a watershed moment. It embodies the potential of bringing together the best of both worlds: the user-friendliness of Keras and the computational might of JAX. This symbiotic fusion is a boon for the deep learning community, making advanced techniques and tools more attainable.
