import tensorflow as tf
import numpy as np

d = 2
k = 2

class LogisticClassifier(object):
    # Compute output of the model
    def predict(self, dataset_pl):
        weights = tf.Variable( tf.truncated_normal( [d, k], dtype=tf.float64) )
        biases = tf.Variable( tf.zeros( [k], dtype=tf.float64 ) )
        logits_op = tf.add( tf.matmul( dataset_pl, weights ), biases)
        return logits_op


    # Compute the loss
    def loss(self, logits_op, labels_pl):
        loss_op  = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logits_op, labels=labels_pl) )
        return loss_op


    # Training operation
    def training(self, loss_op):
        optimize_op = tf.train.GradientDescentOptimizer(1).minimize(loss_op)
        return optimize_op


    # Get softmax probabilities
    def evaluate(self, logits_op):
        return tf.nn.softmax( logits_op )




class SingleHiddenLayerNN(object):

    def __init__(self, nHidden = 3):
        self.nhidden = 3  # number of neurons in the hidden layer

    def predict(self, dataset_pl):
        i2h_weights = tf.Variable(tf.truncated_normal([d, self.nhidden], dtype=tf.float64))
        i2h_biases = tf.Variable(tf.zeros([self.nhidden], dtype=tf.float64))
        i2h_net = tf.nn.relu(tf.matmul(dataset_pl, i2h_weights) + i2h_biases)

        h2o_weights = tf.Variable(tf.truncated_normal([self.nhidden, k], dtype=tf.float64))
        h2o_biases = tf.Variable(tf.zeros([k], dtype=tf.float64))
        logits_op = tf.matmul(i2h_net, h2o_weights) + h2o_biases

        return logits_op

    # LOSS OPERATION
    def loss(self, logits_op, labels_pl):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_op, labels=labels_pl))
        return loss_op

    # Training operation
    def training(self, loss_op):
        optimize_op = tf.train.GradientDescentOptimizer(1).minimize(loss_op)
        return optimize_op

    # Get softmax probabilities
    def evaluate(self, logits_op):
        return tf.nn.softmax(logits_op)


class DoubleHiddenLayerNN(object):

    def __init__(self, nhidden1 = 3, nhidden2 = 3):
        self.nhidden1 = nhidden1
        self.nhidden2 = nhidden2

    def predict(self, dataset_pl):
        # input to hidden 1
        i2h1_weights = tf.Variable(tf.truncated_normal([d, self.nhidden1], dtype=tf.float64))
        i2h1_biases = tf.Variable(tf.zeros([self.nhidden1], dtype=tf.float64))
        i2h1_net = tf.nn.relu(tf.matmul(dataset_pl, i2h1_weights) + i2h1_biases)
        # hidden 1 to hidden 2
        h12h2_weights = tf.Variable(tf.truncated_normal([self.nhidden1, self.nhidden2], dtype=tf.float64))
        h12h2_biases = tf.Variable(tf.zeros([self.nhidden2], dtype=tf.float64))
        h12h2_net = tf.nn.relu(tf.add(tf.matmul(i2h1_net, h12h2_weights),h12h2_biases))
        # hidden 2 to output
        h22o_weights = tf.Variable(tf.truncated_normal([self.nhidden2, k], dtype=tf.float64))
        h22o_biases = tf.Variable(tf.zeros([k], dtype=tf.float64))
        logits_op = tf.add(tf.matmul(h12h2_net, h22o_weights), h22o_biases)

        return logits_op

    # LOSS OPERATION
    def loss(self, logits_op, labels_pl):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_op, labels=labels_pl))
        return loss_op

    # Training operation
    def training(self, loss_op):
        optimize_op = tf.train.GradientDescentOptimizer(.1).minimize(loss_op)
        return optimize_op

    def evaluate(self, logits_op):
        return tf.nn.softmax(logits_op)