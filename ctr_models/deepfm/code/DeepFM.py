#!/usr/bin/env python
#-*- coding: utf-8 -*-

# File Name: DeepFM.py
# Author: Yanmei Fu
# mail: yanmei2016@iscas.ac.cn
# Created Time: 2018-11-02


'''
Tensorflow implementation of DeepFM
Reference:
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.

'''
import sys, os
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import math
from sklearn.metrics import mean_squared_error
import argparse
import LoadData as DATA
import numpy as np
from tensorflow.contrib.layers.python import layers
### -------------------Arguments ------------------------ ####

def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepFM.")
    parser.add_argument('--process', nargs='?', default='train',
                        help='Process type: train, evaluate.')
    parser.add_argument('--path', nargs='?', default="../data/",
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-tag',
                        help='Choose a dataset')
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of epoches')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='flag or pretrain. 1:initialize from pretrain;0:randomly initialize')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    #parser.add_argument('--hidden_factor', nargs='?', default='[16,16]',
    #                    help='Number of hidden factors.')

    parser.add_argument('--hidden_factor', type=int,  default=100,
                        help='Number of embedding_size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
                        help='Specify an optimizer type')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to show the perfomance of each epoch(0 or 1)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perfom batch normalization (0 or 1)')
    parser.add_argument('--decay', type=float, default=0.999,
                        help='Decay value for batch norm.')

    parser.add_argument('--keep', nargs='?', default='[1.0,0.5]',
                        help='dropout ratio list')

    parser.add_argument('--activation', nargs='?', default='relu',
                    help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    return parser.parse_args()


class DeepFm(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size, embedding_size, pretrain_flag, save_file,
                 deep_layer_activation, epoch, batch_size, learning_rate, optimizer_type, batch_norm,
                 batch_norm_decay, keep, loss_type="mse", deep_layers=[32, 32], use_fm=True, use_deep=True, verbose=True, random_seed=2018,
                 greater_is_better=False):


        assert (use_fm or use_deep)

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size=embedding_size
        self.pretrain_flag = pretrain_flag
        self.save_file = save_file
        self.use_fm = use_fm
        self.use_deep= use_deep
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.optimizer_type=optimizer_type
        self.batch_norm=batch_norm
        self.decay = batch_norm_decay
        self.greater_is_better = greater_is_better
        self.verbose=verbose
        self.random_seed = random_seed
        self.deep_layers = deep_layers
        self.deep_layer_activation = deep_layer_activation
        self.loss_type = loss_type
        self.keep = keep
        self.batch_size = batch_size
        self.train_rmse, self.valid_rmse, self.test_tmse=[], [], []

        ## init all variables in tensorflow graph

        self._init_graph()


    def _init_graph(self):
        """
        Init a tensorflow graph containing: input data, variables, model, loss and optimizer

        """
        self.graph = tf.Graph()

        with self.graph.as_default():

            ## TODO: Add summary

            ## set graph lebel random seed
            tf.set_random_seed(self.random_seed)

            ## Input data
            self.train_features = tf.placeholder(tf.int32, shape=[None, None], name='train_feature')
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1], name='train_labels')
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None], name='dropout_keep')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

            # Variables
            self.weights = self._initialize_weights()

            # Model
            ## ----- embeddding features -----
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features) #None * F' * K

            ## ----- first order term ----
            self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features)  # None * F' * 1
            self.y_first_order = tf.reduce_sum(self.y_first_order, 1) ## None * 1

            ## ----- second order term ---
            ## sum_square part
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)   ## None * K
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  ## None * K

            ## square_sum part
            self.squared_features_emb = tf.square(self.embeddings) ## None * F * K
            self.squared_features_emb_sum = tf.reduce_sum(self.squared_features_emb, 1) ## None * K

            ## second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_features_emb_sum)  ## None * K

            # ----- deep component ----
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])  ## None * (F * K)
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep[0])


            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" %i]), self.weights['bias_%d' %i])  ## None * layer[i]
                if self.batch_norm:
                    self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i)

                self.y_deep = self.deep_layer_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep[1])

            ## TODO regulazation


            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], 1)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], 1)
            else:
                concat_input = self.y_deep

            ## TODO regulazation

            self.identity_out = tf.identity(concat_input, name='concat_out')
            self.linear_out = self._linear(self.identity_out, 1, -1, None)



            ## loss
            if self.loss_type == "logloss":
                self.out = tf.sigmoid(self.linear_out)
                self.loss = tf.losses.log_loss(self.train_labels, self.out)
            elif self.loss_type == "mse":
                self.out = self.linear_out
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))
            else:
                raise NotImplementedError
            self.out = tf.identity(self.out, name="out")
            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape() # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print ("#params: %d" %total_parameters)


    def _initialize_weights(self):

        ## TODO pretrain initialize
        all_weights = dict()
        # embeddings
        all_weights["feature_embeddings"] = tf.Variable(
                        tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
                                                                           name="feature_embeddings")  # feature_size * K
        all_weights["feature_bias"] = tf.Variable(
                        tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name="feature_bias")  # feature_size

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        all_weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        all_weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
            dtype=np.float32)  # 1 * layers[0]

        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            all_weights["layer_%d" % i] = tf.Variable(
                                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                                dtype=np.float32)  # layers[i-1] * layers[i]
            all_weights["bias_%d" % i] = tf.Variable(
                                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                                dtype=np.float32)  # 1 * layer[i]
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.decay, center=True, scale=True, updates_collections=None,
            is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.decay, center=True, scale=True, updates_collections=None,
            is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def _init_session(self):
        # adaptively growing video memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def partial_fit(self, data):
        ## fit a batch
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y'],
                     self.dropout_keep: self.keep, self.train_phase: True}

        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X , Y = [], []
        # forward get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i - 1
            else:
                break
        return {'X': X, 'Y': Y}

    def get_ordered_block_from_data(self, data, batch_size, index):  # generate a ordered block of data
        start_index = index*batch_size
        X , Y = [], []
        # get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append(data['Y'][i])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        return {'X': X, 'Y': Y}

    def shuffle_in_unison_scary(self, a, b): # shuffle two lists simutaneously
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def _linear(self, input_tensor, output_nums, l2_reg, activation_fn=None):
        if l2_reg <= 0:
            return layers.fully_connected(input_tensor, output_nums, activation_fn=activation_fn,
                        weights_initializer=layers.xavier_initializer(),
                        biases_initializer=layers.xavier_initializer(),)
        else:
            return layers.fully_connected(input_tensor, output_nums, activation_fn=activation_fn,
                    weights_initializer=layers.xavier_initializer(),
                    biases_initializer=layers.xavier_initializer(),
                    weights_regularizer=layers.l2_regularizer(l2_reg), biases_regularizer=layers.l2_regularizer(l2_reg))

    def train(self, train_data, valid_data, test_data):
        #TODO check init performance

        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(train_data['X'], train_data['Y'])
            total_batch = int(len(train_data['Y']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(train_data, self.batch_size)
                # Fit training
                self.partial_fit(batch_xs)
            t2 = time()

            # evaluate training and validation datasets
            train_result = self.evaluate(train_data)
            valid_result = self.evaluate(valid_data)
            self.train_rmse.append(train_result)
            self.valid_rmse.append(valid_result)
            if self.verbose > 0 and epoch%self.verbose == 0:
                print("Epoch %d [%.1f s]\ttrain=%.4f, validation=%.4f [%.1f s]"
                      %(epoch+1, t2-t1, train_result, valid_result, time()-t2))

            # test_result = self.evaluate(Test_data)
            # print("Epoch %d [%.1f s]\ttest=%.4f [%.1f s]"
            #       %(epoch+1, t2-t1, test_result, time()-t2))
            if self.eva_termination(self.valid_rmse):
                break

        if self.pretrain_flag < 0 or self.pretrain_flag == 2:
            print ("Save model to file as pretrain.")
            self.saver.save(self.sess, self.save_file)

    def get_ordered_block_from_data(self, data, batch_size, index):  # generate a ordered block of data
        start_index = index*batch_size
        X , Y = [], []
        # get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append(data['Y'][i])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        return {'X': X, 'Y': Y}


    def eva_termination(self, valid):
        if len(valid) > 5:
            if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                return True
        return False


    def evaluate(self, data):
        num_example = len(data['Y'])

        # fetch the first batch
        batch_index = 0
        batch_xs = self.get_ordered_block_from_data(data, self.batch_size, batch_index)
        y_pred = None

        while len(batch_xs['X']) > 0:
            num_batch = len(batch_xs['Y'])
            feed_dict = {self.train_features: batch_xs['X'], self.train_labels: [[y] for y in batch_xs['Y']], self.dropout_keep: list(1.0 for i in range(len(self.keep))), self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)
            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch, ))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch, ))))

            ## fetch the next batch
            batch_index += 1
            batch_xs = self.get_ordered_block_from_data(data, self.batch_size, batch_index)
        y_true = np.reshape(data['Y'], (num_example, ))


        predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
        predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # bound the higher values
        RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
        return RMSE


def make_save_file(args):
    pretrain_path = '../pretrain/%s_%d' %(args.dataset, args.hidden_factor)
    if not os.path.exists(pretrain_path):
        os.makedirs(pretrain_path)
    save_file = pretrain_path+'/%s_%d' %(args.dataset, args.hidden_factor)
    return save_file

def train(args):
    data = DATA.LoadData(args.path, args.dataset)
    if args.verbose > 0:
        print("DeepFm: dataset=%s, factors=%s, epoch=%d, batch_size=%d, lr=%.4f, keep=%s,\
          optimizer=%s, batch_norm=%s, decay=%f, activation=%s" %(args.dataset, args.hidden_factor, \
          args.epoch, args.batch_size, args.lr, eval(args.keep), args.optimizer, args.batch_norm, args.decay, \
          args.activation))


    activation_function = tf.nn.relu
    if args.activation == 'sigmoid':
        activation_function = tf.sigmoid
    elif args.activation == 'tanh':
        activation_function = tf.tanh
    elif args.activation == 'identity':
        activation_function = tf.identity

    save_file = make_save_file(args)

    ## training
    t1 = time()
    model = DeepFm(data.features_M, data.field, args.hidden_factor, args.pretrain, save_file,
                   activation_function, args.epoch, args.batch_size, args.lr, args.optimizer,
                   args.batch_norm, args.decay, eval(args.keep))
    model.train(data.Train_data, data.Validation_data, data.Test_data)

    ## find the best validation result across iterations

    best_valid_score = 0
    if model.greater_is_better:
        best_valid_score = max(model.valid_rmse)
    else:
        best_valid_score = min(model.valid_rmse)

    best_epoch = model.valid_rmse.index(best_valid_score)
    print ("Best Iter(validation)=%d\t train = %.4f, valid = %.4f [%.1f s]"
           %(best_epoch+1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], time()-t1))

def get_ordered_block_from_data(data, batch_size, index):  # generate a ordered block of data
    start_index = index*batch_size
    X , Y = [], []
    # get sample
    i = start_index
    while len(X) < batch_size and i < len(data['X']):
        if len(data['X'][i]) == len(data['X'][start_index]):
            Y.append(data['Y'][i])
            X.append(data['X'][i])
            i = i + 1
        else:
            break
    return {'X': X, 'Y': Y}


def evaluate(args):
    data = Data.LoadData(args.path, args.dataset).Test_data
    save_file = make_save_file(args)


    ## load the graph
    weight_saver = tf.train.import_meta_graph(save_file + '.meta')
    pretrain_graph = tf.get_default_graph()

    train_features = pretrain_graph.get_tensor_by_name('train_feature:0')
    train_labels = pretrain_graph.get_tensor_by_name('train_labels:0')
    dropout_keep = pretrain_graph.get_tensor_by_name('dropout_keep:0')
    train_phase = pretrain_graph.get_tensor_by_name('train_phase:0')
    out = pretrain_graph.get_tensor_by_name("out:0")

    ## restore session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    weight_saver.restore(sess, save_file)


    # start evaluation

    num_example = len(data['Y'])

    # fetch the first batch
    batch_index = 0
    batch_xs = get_ordered_block_from_data(data, args.batch_size, batch_index)
    y_pred = None

    while len(batch_xs['X']) > 0:
        num_batch = len(batch_xs['Y'])
        feed_dict = {train_features: batch_xs['X'], train_labels: [[y] for y in batch_xs['Y']], dropout_keep: list(1.0 for i in range(len(self.keep))), train_phase: False}
        batch_out = sess.run(out, feed_dict=feed_dict)

        if batch_index == 0:
            y_pred = np.reshape(batch_out, (num_batch, ))
        else:
            y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch, ))))

        ## fetch the next batch
        batch_index += 1
        batch_xs = get_ordered_block_from_data(data, args.batch_size, batch_index)
    y_true = np.reshape(data['Y'], (num_example, ))


    predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
    predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # bound the higher values
    RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
    print("Test RMSE: %.4f"%(RMSE))


def main():
    args = parse_args()
    if args.process == 'train':
        train(args)
    elif args.process == 'evaluate':
        evaluate(args)


if __name__ == "__main__":
    main()
