from aotoencoder import VariationalAutoencoder
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(0)
tf.set_random_seed(0)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples
print(n_samples)

network_architecture = dict(n_hidden_recog_1 =500, n_hidden_recog_2=500, n_hidden_gener_1=500, n_hidden_gener_2=500, n_input=784, n_z=20)

batch_size = 100
vae = VariationalAutoencoder(network_architecture, learning_rate=0.001, batch_size=batch_size)

training_epochs = 75
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        cost = vae.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size

    if epoch % 5 == 0:
        print("Epoch:", "%04d" %(epoch + 1), "cost=", "{:.0f}".format(avg_cost))

