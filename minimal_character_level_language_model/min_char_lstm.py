# --*- coding:utf-8 -*--

from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
from utils import *
class SimpleModel(object):


    def __init__(self, vocabulary_size, num_steps, data_generator, num_nodes=64, batch_size=16, summary_frequency=100):
        self.vocabulary_size = vocabulary_size
        self.num_nodes = num_nodes
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.data_generator = data_generator
        self.summary_frequency = summary_frequency
        self.buildGraph()

    def buildGraph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.ifcox = tf.Variable(tf.truncated_normal([self.vocabulary_size, 4 * self.num_nodes], -0.1, 0.1))
            self.ifcom = tf.Variable(tf.truncated_normal([self.num_nodes, 4 * self.num_nodes], -0.1, 0.1))
            self.ifcob = tf.Variable(tf.zeros([1, 4 * self.num_nodes]))
            self.saved_output = tf.Variable(tf.zeros([self.batch_size, self.num_nodes]), trainable=False)
            self.saved_state = tf.Variable(tf.zeros([self.batch_size, self.num_nodes]), trainable=False)

            self.w = tf.Variable(tf.truncated_normal([self.num_nodes, self.vocabulary_size], -0.1, 0.1))
            self.b = tf.Variable(tf.zeros([self.vocabulary_size]))

    def lstm_cell(self, i, o, state):
        all_gates_state = tf.matmul(i, self.ifcox) + tf.matmul(o, self.ifcom) +self.ifcob
        input_gate = tf.sigmoid(all_gates_state[:, 0:self.num_nodes])
        forget_gate = tf.sigmoid(all_gates_state[:, self.num_nodes:2 *self.num_nodes])
        update = all_gates_state[:, 2 * self.num_nodes:3 * self.num_nodes]
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(all_gates_state[:, 3 * self.num_nodes:])
        return output_gate * tf.tanh(state), state

    def logprob(self, predictions, labels):
        """Log-probability of the true labels in a predicted batch."""
        predictions[predictions < 1e-10] = 1e-10
        return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

    def sample(self, prediction):
        """Turn a (column) prediction into 1-hot encoded samples."""
        p = np.zeros(shape=[1, self.vocabulary_size], dtype=np.float)
        p[0, self.sample_distribution(prediction[0])] = 1.0
        return p

    def sample_distribution(self, distribution):
        """Sample one element from a distribution assumed to be an array of normalized
        probabilities.
        """

        r = random.uniform(0, 1)
        s = 0
        for i in range(len(distribution)):
            s += distribution[i]
            if s >= r:
                return i
        return len(distribution) - 1

    def random_distribution(self):
        """Generate a random column of probabilities."""
        b = np.random.uniform(0.0, 1.0, size=[1, self.vocabulary_size])
        return b / np.sum(b, 1)[:, None]

    def run(self):

        ## 需要保证是同一个图
        with self.graph.as_default():
            train_data = list()
            for _ in range(self.data_generator._num_unrollings + 1):
                train_data.append(tf.placeholder(tf.float32, shape=[self.batch_size, self.vocabulary_size]))
            train_inputs = train_data[:self.data_generator._num_unrollings]
            train_labels = train_data[1:]
            print(train_data)
            outputs = list()
            output = self.saved_output
            state = self.saved_state
            for i in train_inputs:
                output, state = self.lstm_cell(i, output, state)
                outputs.append(output)

            with tf.control_dependencies([self.saved_output.assign(output),
                                          self.saved_state.assign(state)]):
                logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), self.w, self.b)
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits, labels=tf.concat( train_labels, 0)))
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(
                10.0, global_step, 5000, 0.1, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            gradients, v = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
            optimizer = optimizer.apply_gradients(
                zip(gradients, v), global_step=global_step)

            train_prediction = tf.nn.softmax(logits)
            sample_input = tf.placeholder(tf.float32, shape=[1, self.vocabulary_size])
            saved_sample_output = tf.Variable(tf.zeros([1, self.num_nodes]))
            saved_sample_state = tf.Variable(tf.zeros([1, self.num_nodes]))
            reset_sample_state = tf.group(
                saved_sample_output.assign(tf.zeros([1, self.num_nodes])),
                saved_sample_state.assign(tf.zeros([1, self.num_nodes])))
            sample_output, sample_state = self.lstm_cell(
                sample_input, saved_sample_output, saved_sample_state)

            with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                          saved_sample_state.assign(sample_state)]):
                sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, self.w, self.b))

        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            print("Initialized")
            mean_loss = 0


            for step in range(self.num_steps):
                batches = self.data_generator.next()
                feed_dict = dict()

                for i in range(self.data_generator._num_unrollings + 1):
                    feed_dict[train_data[i]] = batches[i]
                _, l, predictions, lr = session.run([optimizer, loss,
                                                     train_prediction, learning_rate], feed_dict=feed_dict)

                mean_loss += l
                if step % self.summary_frequency == 0:
                    if step > 0:
                        mean_loss = mean_loss / self.summary_frequency
                    print('Average loss at step', step, ':', mean_loss, 'learning rate:', lr)
                    mean_loss = 0
                    labels = np.concatenate(list(batches)[1:])
                    print('Minibatch perplexity: %.2f' % float(np.exp(self.logprob(predictions, labels))))
                    if step % (self.summary_frequency * 10) == 0:

                        print('=' * 80)
                        for _ in range(5):
                            feed = self.sample(self.random_distribution())
                            sentence = characters(feed, self.data_generator._id2char)[0]
                            reset_sample_state.run()
                            for _ in range(79):
                                prediction = sample_prediction.eval({sample_input: feed})
                                feed = self.sample(prediction)
                                sentence += characters(feed, self.data_generator._id2char)[0]
                            print(sentence)
                        print('=' * 80)


