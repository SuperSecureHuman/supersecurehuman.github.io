---
title: "Distributed Training With Tensorflow"
excerpt: "Leveraging Multi Worker Mirrored Strategy in TensorFlow to train models across multiple workers using data parallelism."
tags:
    - deep learning
    - tensorflow
    - cluster
---

## Tensorflow MultiWorkerMirroredStrategy

MultiWorkerMirroredStrategy is a distributed training strategy in TensorFlow that is designed to train large-scale models across multiple machines in a cluster. This strategy allows for synchronous training across multiple workers, where each worker trains a copy of the model on a subset of the data. During training, the gradients of each worker are aggregated and applied to the model weights, which helps to speed up the training process and improve the model's accuracy.

MultiWorkerMirroredStrategy employs data parallelism to distribute the training across multiple workers in a cluster. In data parallelism, each worker trains a copy of the model on a subset of the training data, and the gradients from each worker are averaged to update the model weights. This approach is particularly effective when dealing with large datasets that cannot fit into the memory of a single machine.

During training, the training data is divided into equal-sized shards, with each worker processing a unique shard. The model and its variables are replicated on each worker, and the updates made by each worker are aggregated to ensure consistency across the model's replicas. By using this approach, the workers can train the model simultaneously, which speeds up the training process and enables the efficient use of resources.

### Multi-Worker Configuration

When using MultiWorkerMirroredStrategy in TensorFlow to train a model across multiple workers, several environment variables need to be set to configure the training job. These environment variables provide information about the cluster and the specific task that is being performed by each worker.

The most important environment variable that needs to be set is `TF_CONFIG`. This variable is used to specify the cluster configuration and the role of each worker in the cluster. The `TF_CONFIG` variable should be set to a JSON string that contains information about the cluster, such as the IP addresses and port numbers of each worker, and the task index of the worker within the cluster. Here is an example of a `TF_CONFIG` JSON string:

```json
{
  "cluster": {
    "worker": ["worker0.example.com:12345", "worker1.example.com:23456"]
  },
  "task": {"type": "worker", "index": 0}
}
```

In this example, the `TF_CONFIG` variable specifies a cluster with two workers and the current worker's task index is 0. It is important to note that the `TF_CONFIG` variable should be set on each worker before the training job is started.

In the other node, set the `TF_CONFIG` variable to the following JSON string:

```json
{
  "cluster": {
    "worker": ["worker0.example.com:12345", "worker1.example.com:23456"]
  },
  "task": {"type": "worker", "index": 1}
}
```

Here the index is 1. This means that the current worker is the second worker in the cluster.

You can set the `TF_CONFIG` variable in bash using the following command:

```bash
export TF_CONFIG='{"cluster": {"worker": ["worker0.example.com:12345", "worker1.example.com:23456"]}, "task": {"type": "worker", "index": 0}}'
```

### Some notes

#### Dataset loading

Take a look at this example:

```python

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
# The `x` arrays are in uint8 and have values in the [0, 255] range.
# You need to convert them to float32 with values in the [0, 1] range.
x_train = x_train / np.float32(255)
y_train = y_train.astype(np.int64)
train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)

```

When using `MultiWorkerMirroredStrategy` in TensorFlow to train a model across multiple workers, it is important to use the `.repeat()` method on the input dataset. The `.repeat()` method is used to repeat the dataset indefinitely, which is necessary to ensure that each worker processes a unique subset of the data during training.

Without using the `.repeat()` method, each worker would process the same subset of the data during each epoch of training, which could result in overfitting and poor model performance. By using the `.repeat()` method, each worker processes a unique subset of the data during each epoch, which helps to prevent overfitting and ensures that the model learns from a diverse set of examples.

When using the `.repeat()` method, it is important to set the steps_per_epoch parameter to the number of training examples divided by the batch size multiplied by the number of workers. This ensures that each worker processes an equal number of examples during each epoch of training.

Steps per epoch can be computed by the following logic

```python
steps_per_epoch = len(x_train) // batch_size*num_workers
```

#### Defining the model

When using `MultiWorkerMirroredStrategy` in TensorFlow to train a model across multiple workers, it is important to define the model inside the `strategy.scope()` context manager. This is because the strategy needs to know how to replicate the model across multiple workers and how to aggregate the gradients during training.

To define the model within the strategy scope, you can use the strategy.scope() context manager. This ensures that the model is created within the scope of the strategy and will be replicated across all workers.

#### Learning Rate and Batch Size

By experimentation, find the most optimal batch size per node in your case. Then in the distributed script, scale the batch size by the number of nodes. For example, if you have 4 nodes and the optimal batch size per node is 32, then the batch size in the distributed script should be 32*4=128.

The learning rate needs some tuning as well. As a general rule of thumb, scale your learning rate same as the batch size.

### Code Changes

#### 1. Initilize `MultiWorkerMirroredStrategy`

```python
communication_options = tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)

strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)
```

We set the communication backend to be NCCL and create a strategy

In case NCCL fails, try using `implementation=tf.distribute.experimental.CommunicationImplementation.RING`

RING uses gRPC-based communication, which might result in performance bottlenecks

#### 2. Creating your model

Usually, the model is built in this way

```python
.
.
.
.
model = ......
model.compile(......)

```

To make this work in a distributed setup, wrap them in a function and return the model, and load the model within the scope of the strategy

```python
def create_model():
 model = ....
 model.compile(....)
 return model

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = create_model()
```

#### 3. Dataset

This is a sample data loading. Double Check with your dataset

```python
multi_worker_train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(dataset_length).repeat().batch(global_batch_size)
```

.repeat() is necessary to fill the batch_size. Another option is to use `drop_remainder=True`, which will drop data from epochs. You can use it if it's desirable.

When using data_augmentation, I would suggest using `.repeat`

#### 4. Setting some Parameters

You increase your batch size by running distributed training in a higher view. It would help if you scaled your learning rate appropriately. A rule of thumb

`learning_rate = learning_rate * num_workers`

```
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])
```

And your batch size needs to be scaled too

`global_batch_size = batch_size*num_workers`

Note: Do a test run without distributed training to find the optimal batch size per node, then scale it.

`steps_per_epoch = (dataLength // global_batch_size)`

Setting steps_per_epoch is necessary if you use `.repeat()` on your dataset. It would be best if you experimented with steps_per_epoch and your dataset function to find an optimal config.

#### 5. Train

If using the validation dataset, load like how to train data was loaded, and use steps for validation similar to what was done for train set.

Callbacks can also be used. All the callbacks will be executed only once on the master node

```python
model.fit(multi_worker_train_dataset, epochs=10, steps_per_epoch=steps_per_epoch)
```

### An end-to-end example

This is an end-to-end example of distributed training using `MultiWorkerMirroredStrategy` in TensorFlow. The example uses the MNIST dataset to train a simple CNN model.

```python
import tensorflow as tf
import numpy as np
import json
import os

def mnist_dataset(batch_size):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the [0, 255] range.
  # You need to convert them to float32 with values in the [0, 1] range.
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
  return train_dataset

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model



# laod the env variable
tf_config = json.loads(os.environ['TF_CONFIG'])
# get number of workers
num_workers = len(tf_config['cluster']['worker'])

##### Set Strategy
communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=communication_options)
#####


batch_size = 64
global_batch_size = batch_size * num_workers


#dataset = mnist_dataset(batch_size)
multi_worker_dataset = mnist_dataset(global_batch_size)

#model = build_and_compile_cnn_model()

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = build_and_compile_cnn_model()

# Here, set steps according to your dataset
# Steps = len(data)/global_batch_size
#model.fit(dataset, epochs=3, steps_per_epoch=70)
multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=35)
```

### Running

Just set the environment variable `TF_CONFIG` and run the script in each node.

More robust ways to launch training will be updated later.