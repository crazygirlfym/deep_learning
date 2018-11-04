import tensorflow as tf
from initialize import xavier_init
import numpy as np
class VariationalAutoencoder(object):

    ## 推导过程可以参考 http://blog.csdn.net/ustbfym/article/details/78870990
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.x = tf.placeholder(tf.float32, [None, network_architecture['n_input']])
        self._create_network()
        self._create_loss_optimizer()
        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)


    def _create_network(self):
        network_weights = self._initialize_weights(**self.network_architecture)
        self.z_mean, self.z_log_sigma_sq = self._recognition_network(network_weights['weights_recog'], network_weights['biases_recog'])

        n_z = self.network_architecture['n_z']
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32)
        print("batch_size: " + str(self.batch_size))

        # z = mu + sigma * epsilon  高斯分布的重参数化
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        self.x_reconstr_mean = self._generator_network(network_weights['weights_gener'], network_weights['biases_gener'])


    def _recognition_network(self, weights, biases):
        """
        Generate probabilistic encoder, which maps input into a normal distribution in latent space
        :param weights:
        :param biases:
        :return:
        """
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))

        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
        z_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])  #[?, 20]
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        """
        Generate probabilistic decoder, which maps points in latent space into a Bernoulli distribution in data space
        :param weights:
        :param biases:
        :return:
        """
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        x_recon_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean']))
        return x_recon_mean

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1, n_hidden_gener_2, n_input, n_z):
        all_weights = {}

        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))
        }

        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))
        }

        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))
        }

        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))
        }
        return all_weights

    def _create_loss_optimizer(self):
        ## 1) reconstruction loss: the negative log probability of the input under the reconstructed distribution
        ## E_z (log(p_theta(x|z)))
        reconstr_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean) + (1-self.x) * tf.log(1e-10 + 1-self.x_reconstr_mean))

        ## 2) latent loss, which is defined as the kullback Leibler divergence
        ## D = 0.5 \sum(1 + log(\sigma^2) - \mu^2 -\sigma^2)
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)

        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(self.cost)


    def partial_fit(self, x):
        """
        Train model based on mini-batch of input data
        :param x:
        :return:  return the cost of mini-batch
        """

        opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: x})
        return cost

    def transform(self, x):
        return self.sess.run(self.z_mean, feed_dict={self.x: x})


    def generate(self, z_mu = None):
        """
        Generate data by sampling from latent space
        :param z_mu:
        :return:
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        return self.sess.run(self.x_reconstr_mean, feed_dict={self.z: z_mu })

    def reconstruct(self, x):
        """
        Use VAE to reconstruct given data
        :param x:
        :return:
        """
        return self.sess.run(self.x_reconstr_mean, feed_dict={self.x, x})





