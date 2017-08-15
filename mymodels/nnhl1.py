import tensorflow as tf

# TODO: How could I automate these?
d = 2
k = 2
nhidden = 2 # number of neurons in the hidden layer

# THE MODEL
def inference(dataset_pl):
    i2h_weights    = tf.Variable( tf.truncated_normal( [d, nhidden], dtype=tf.float64) )
    i2h_biases     = tf.Variable( tf.zeros( [nhidden], dtype=tf.float64 ) )
    i2h_net        = tf.nn.relu( tf.matmul( dataset_pl, i2h_weights ) + i2h_biases )

    h2o_weights    = tf.Variable( tf.truncated_normal( [nhidden, k], dtype=tf.float64) )
    h2o_biases     = tf.Variable( tf.zeros( [k], dtype=tf.float64 ) )
    logits_op      = tf.matmul( i2h_net, h2o_weights) + h2o_biases


    return logits_op

# LOSS OPERATION
def loss(logits_op, labels_pl):
    loss_op  = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits_op, labels_pl) )
    return loss_op

# TRAINING OPERATION
def training(loss_op):
    optimize_op = tf.train.GradientDescentOptimizer(.05).minimize(loss_op)
    return optimize_op

# EVALUATION OPERATION
def evaluate(logits_op):
    return tf.nn.softmax( logits_op )