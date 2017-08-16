import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import utilities.fakedata as fakedata
import utilities.inspector as inspector
import mymodels.nnhl1 as nnhl1


""" 
This code implements a neural network with a single hidden layer
comprising only two neurons.

"""

def accuracy(predictions, labels):
    numcorrect  = np.sum( np.argmax(predictions, 1) == np.argmax(labels, 1), dtype=float )
    fraccorrect = numcorrect / predictions.shape[0]
    return( 100.0 * fraccorrect )



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                        
# RUN TRAINING
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def runtraining(
    train_dataset, 
    train_labels, 
    valid_dataset, 
    valid_labels, 
    test_dataset, 
    test_labels):
        
    num_training_steps = 2001

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # CONSTRUCT COMPUTATION GRAPH 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    graph = tf.Graph()
    with graph.as_default():


        # DEFINE INPUT OPS
        tf_train_dataset = tf.constant( train_dataset )
        tf_train_labels  = tf.constant( train_labels  )
        tf_valid_dataset = tf.constant( valid_dataset )
        tf_test_dataset  = tf.constant( test_dataset  )

        # DEFINE GRAPH OPS
        logits_op       = nnhl1.inference( tf_train_dataset )
        loss_op         = nnhl1.loss(      logits_op, tf_train_labels )    
        optimize_op     = nnhl1.training(  loss_op   )
        eval_op         = nnhl1.evaluate(  logits_op )




    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # EXECUTE COMPUTATION GRAPH #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    with tf.Session(graph=graph) as session:
        
        tf.initialize_all_variables().run()
                
        for step in range(num_training_steps):            
            _, l, predictions = session.run( [optimize_op, loss_op, eval_op] )
            
            # Track performance
            if (step % (num_training_steps / 10) == 0):
                # print( 'Loss at step %d: %f' % (step, l) )
                print( 'Training accuracy at step %d: %.1f%%' % (step, accuracy( predictions, train_labels) ))


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # INSPECT RESULTS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    inspector.showData(train_dataset, predictions, train_labels)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

if __name__=="__main__":

    # # # # # # # #
    # CREATE DATA #
    # # # # # # # #
    # data, labels = fakedata.generate2BlobData()
    data, labels = fakedata.generateXORData()
    n = data.shape[0] 

    # # # # # # # # # # # # # # #
    # TRAINING/VALIDATION/TEST  #
    # # # # # # # # # # # # # # # 
    ntrain = int(1.0*n)
    nvalid = int(0.0*n)
    ntest  = n - ntrain - nvalid

    train_dataset = data[:ntrain,:]
    valid_dataset = data[ntrain:ntrain+nvalid,:]
    test_dataset  = data[-ntest:,:]

    train_labels = labels[:ntrain,:]
    valid_labels = labels[ntrain:ntrain+nvalid,:]
    test_labels  = labels[-ntest:,:]

    runtraining( train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)



