import tensorflow as tf

# TODO: How could I automate these?
d = 2
k = 2
nhidden1 = 2
nhidden2 = 2

# THE MODEL
def inference(dataset_pl):
    # input to hidden 1
    i2h1_weights    = tf.Variable( tf.truncated_normal( [d, nhidden1], dtype=tf.float64) )
    i2h1_biases     = tf.Variable( tf.zeros( [nhidden1], dtype=tf.float64 ) )
    i2h1_net        = tf.nn.relu( tf.matmul( dataset_pl, i2h1_weights ) + i2h1_biases )
    # hidden 1 to hidden 2
    h12h2_weights    = tf.Variable( tf.truncated_normal( [nhidden1, nhidden2], dtype=tf.float64) )
    h12h2_biases     = tf.Variable( tf.zeros( [nhidden2], dtype=tf.float64 ) )
    h12h2_net        = tf.nn.relu( tf.matmul(i2h1_net , h12h2_weights ) + h12h2_biases )
    # hidden 2 to output
    h22o_weights    = tf.Variable( tf.truncated_normal( [nhidden2, k], dtype=tf.float64) )
    h22o_biases     = tf.Variable( tf.zeros( [k], dtype=tf.float64 ) )
    logits_op      = tf.matmul( h12h2_net, h22o_weights) + h22o_biases


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