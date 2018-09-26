# -*-- coding:utf-8 -*--
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import os

class Item2Vec(object):
    """
        implement item2vec by skipgram
    """
    def __init__(self, dataset, vocab_size, embed_size, batch_size, num_sample, learning_rate, steps = 10):
        """
        :param dataset:
        :param vocab_size:
        :param embed_size:
        :param batch_size:
        :param num_sample:
        :param learning_rate:
        :param steps:
        """
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sample = num_sample
        self.learning_rate = learning_rate
        self.steps = steps
        self.global_step = tf.get_variable("global_step", initializer=tf.constant(0), trainable=False)


    def _create_embedding(self):
        with tf.name_scope("embedding"):
            self.embed_matrix = tf.get_variable(shape=[self.vocab_size, self.embed_size], name="embed_matrix",
                                                initializer=tf.random_uniform_initializer())
            self.embedding = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name="embedding")

    def _create_data(self):
        with tf.name_scope("data"):
            self.iterator = self.dataset.make_initializable_iterator()
            self.center_words, self.target_words = self.iterator.get_next()

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.nce_weight = tf.get_variable("nce_weight", shape=[self.vocab_size, self.embed_size],
                                              initializer=tf.truncated_normal_initializer(
                                                  stddev=1.0 / (self.embed_size ** 0.5)))
            self.nce_bias = tf.get_variable("nce_bias", initializer=tf.zeros(shape=[self.vocab_size]))
            nce_loss = tf.nn.nce_loss(weights=self.nce_weight, biases=self.nce_bias,
                                      labels=self.target_words, inputs=self.embedding,
                                      num_classes=self.vocab_size,
                                      num_sampled=self.num_sample)
            self.loss = tf.reduce_mean(nce_loss, name="nce_loss")

    def _create_optimizer(self):
            with tf.name_scope("optimizer"):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def _create_summary(self):
        with tf.name_scope("summary"):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)

            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_data()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()

    def train(self, epoches=100):
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(self.iterator.initializer)
            sess.run(tf.global_variables_initializer())
            # make sure that continue to train no matter what happened
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoint/checkpoint'))
            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            total_loss = 0.0
            writer = tf.summary.FileWriter('graphs/item2vec/learning_rate' + str(self.learning_rate), sess.graph)
            initial_step = self.global_step.eval()
            print("initial_step is {}".format(initial_step))
            for i in range(initial_step, initial_step + epoches):
                try:
                    single_loss, _, summary = sess.run([self.loss, self.optimizer, self.summary_op])
                    writer.add_summary(summary, global_step=i)
                    if (i % self.steps) == 0:
                        total_loss += single_loss
                        print("Average loss is {}".format(total_loss / (i + 1)))
                        saver.save(sess, "checkpoint/skip-gram", i)
                except tf.errors.OutOfRangeError:
                    # move cursor to first
                    sess.run(self.iterator.initializer)

    def load_vector(self, model_save_path, embed_matrix_save_path):
        ckpt = tf.train.get_checkpoint_state(model_save_path)
        if ckpt:
            best_path = ckpt.model_checkpoint_path
        else:
            raise Exception("there is no model found!")

        g = tf.Graph()
        with g.as_default():
            with tf.Session() as sess:
                Saver = tf.train.import_meta_graph(best_path + ".meta")
                Saver.restore(sess, tf.train.latest_checkpoint(model_save_path))
                embed_matrix = tf.get_default_graph().get_tensor_by_name("embed_matrix:0")
                embed_matrix_out = sess.run(embed_matrix)
                with open(embed_matrix_save_path, 'w') as out:
                    for i in range(embed_matrix_out.shape[0]):

                        out_line = [str(_) for _  in embed_matrix_out[i]]
                        out.write(" ".join(out_line) + "\n")



