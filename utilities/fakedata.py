import numpy as np
    
def generateXORData():
    data = np.zeros([4,2])
    data[0,:] = [0,0]
    data[1,:] = [0,1]
    data[2,:] = [1,0]
    data[3,:] = [1,1]
    labels = np.array([0,1,1,0])
    labels = (np.arange(2) == labels[:,None]).astype(np.float64)
    return data, labels


def generate2BlobData(n = 100, test = True):
    """Generates data from two different gaussian distributions.
    
    Arguments:
        n is the number of data points
    
    Returns:
        data is a 2n-by-2 array of 2n data points, n points 
        from 2 classes.
    
        labels is a 2n-by-2 array of 1-hot labels. In other 
        words, labels[i,:] is the probability distribution 
        of the class assingment of datapoint i."""
    
    if test:
        np.random.seed(0)

    def mvg(mu,C):
        data = np.random.randn(n,2)
        data = np.dot(data,C.T) + mu.reshape(1,2)
        return data

    # Define parameters    
    mu1    = np.array( [-1.5,0] ) # mean for first Gaussian
    mu2    = np.array( [3,0] ) # mean for second Gaussian
    C1     = np.array( [[1,.2] ,[.2,1]] ) # covariance matrix for first Gaussian
    C2     = np.array( [[1,-.5],[-.5,2]] ) # covariance matrix for second Gaussian

    # Simulate data from parameters
    data1  = mvg(mu1, C1)
    data2  = mvg(mu2, C2)
    data   = np.vstack( (data1,data2) )
    labels = np.hstack( ( 0*np.ones(n), 1*np.ones(n) ) )    
    labels = (np.arange(2) == labels[:,None]).astype(np.float64) # for one-hot encoding
    # randomly permute data
    idx    = np.random.permutation(2*n)
    data   = data[idx,:]
    labels = labels[idx]
    
    return data, labels