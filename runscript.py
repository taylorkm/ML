import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import utilities.fakedata as fakedata
import utilities.inspector as inspector
import mymodels.mymodels as mm


"""
This code creates several classification models for two-class
classification problem. 
"""

def run_experiment(data, labels, model):
    # Get training/validation/test
    n = data.shape[0]
    ntrain = int(0.45*n)
    nvalid = int(0.1*n)
    ntest = n - ntrain - nvalid

    train_dataset = data[:ntrain,:]
    valid_dataset = data[ntrain:ntrain+nvalid,:]
    test_dataset = data[-ntest:,:]

    train_labels = labels[:ntrain,:]
    valid_labels = labels[ntrain:ntrain+nvalid,:]
    test_labels = labels[-ntest:,:]
    num_training_steps = 1001


    # Construct flow graph    
    with tf.Session() as session:

        # Inputs
        X = tf.placeholder(dtype = np.float64)
        y = tf.placeholder(dtype = np.float64)

        # Graph operations
        logits_op = model.predict(X)
        loss_op = model.loss(logits_op, y)
        optimize_op = model.training(loss_op)
        eval_op = model.evaluate(logits_op)

        session.run(tf.global_variables_initializer())

        # Train
        for step in range(num_training_steps):
            _, l, predictions = session.run([optimize_op, loss_op, eval_op],feed_dict={X: train_dataset, y: train_labels})
            test_predictions = session.run(eval_op,feed_dict={X: test_dataset, y: test_labels})

            # Track performance
            if (step % (num_training_steps / 10) == 0):
                print('Training/Test accuracy at step %d: %.3f / %.3f' %
                      (step, inspector.accuracy(predictions, train_labels), inspector.accuracy(test_predictions, test_labels) ) )


        # Test on validation set
        valid_predictions = session.run(eval_op,feed_dict={X: valid_dataset, y: valid_labels})
        print('Accuracy on validation set: %.1f%%' % ( inspector.accuracy(valid_predictions,valid_labels) ) )

    # Inspect results
    inspector.showData(train_dataset, predictions, train_labels)
    # inspector.showData(valid_dataset, valid_predictions, valid_labels)
    # inspector.showLabels(valid_predictions, valid_labels)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__=="__main__":

    # Create data
    # data, labels = fakedata.generate2BlobData()
    data, labels = fakedata.generateXORData()

    model = [mm.LogisticClassifier(), mm.SingleHiddenLayerNN(), mm.DoubleHiddenLayerNN()]
    # model = [mm.LogisticClassifier()]

    for m in model:
        run_experiment(data, labels, m)



