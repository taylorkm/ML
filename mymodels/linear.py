import tensorflow as tf

# TODO: How could I automate these?
d = 2
k = 2

# THE MODEL
def inference(dataset_pl):
    weights    = tf.Variable( tf.truncated_normal( [d, k], dtype=tf.float64) )
    biases     = tf.Variable( tf.zeros( [k], dtype=tf.float64 ) )
    logits_op  = tf.matmul( dataset_pl, weights ) + biases 
    return logits_op

# LOSS OPERATION
def loss(logits_op, labels_pl):
    loss_op  = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits_op, labels_pl) )
    return loss_op

# TRAINING OPERATION
def training(loss_op):
    optimize_op = tf.train.GradientDescentOptimizer(5).minimize(loss_op)
    return optimize_op

# EVALUATION OPERATION
def evaluate(logits_op):
    return tf.nn.softmax( logits_op )