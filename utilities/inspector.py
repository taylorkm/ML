import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def accuracy(predictions, labels):

    numcorrect  = np.sum( np.argmax(predictions, 1) == np.argmax(labels, 1), dtype=float )
    fraccorrect = numcorrect / len(predictions)
    return( 100.0 * fraccorrect )


def isgoodprediction(predictions, labels):
    return ( np.argmax(predictions) == np.argmax(labels) )


def showData(data, labels, truelabels=None):
    """Visualization of data and their classes.

    Arguments:
      data is a n-by-d numpy array

      labels is a n-by-k array of 1-hot encodings

      When provided, truelabels is a n-by-k array
      of the actual 1-hot encodings of the data."""

    n = data.shape[0]
    colors = np.dot(labels,np.arange(2)).reshape([-1]) # for color-coding labels

    plt.figure()
    plt.scatter(data[:,0],data[:,1], c=colors, s=40)


    # identify incorrectly labeled examples with an x colored with the correct class
    if truelabels is not None:
        incorrect_idx = []
        truecolors = np.dot(truelabels,np.arange(2)).reshape([-1])
        for i in range(n):
            if not isgoodprediction(labels[i,:], truelabels[i,:]):
                incorrect_idx.append(i)
        plt.scatter( data[incorrect_idx,0], data[incorrect_idx,1],s=50, c='k', marker='x',lw=5 ,label='misclassified')

    plt.legend()
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()



def showLabels(labels, truelabels):
    # Compare the first ten predictions with actuals.
    print( "%12s %12s" %( "Predicted","Actual" ))
    for j in range(10):
        print( "[%.2f, %.2f]   [%.2f, %.2f]    " % ( labels[j,0], labels[j,1], truelabels[j,0], truelabels[j,1] ) )
