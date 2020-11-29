# Implementation of  Widrow's Adaptive Linear classifier algorithm
# Author: idebtor@gmail.com
# 2018.03.21 - Creation

class AdalineGD_Momentum(object):
    """ADAptive LInear NEuron classifier.
    Parameters
        eta: float, Learning rate (between 0.0 and 1.0)
        epochs: int, Passes over the training dataset.
        random_seed : int, random funtion seed for reproducibility

    Attributes
        w_ : 1d-array, Weights after fitting.
        cost_ : list, Sum-of-squares cost function value in each epoch.
    """
    def __init__(self, eta=0.01, epochs=10, random_seed=1):
        self.eta = eta
        self.epochs = epochs
        self.random_seed = random_seed

    def fit(self, X, y):
        """ Fit training data.
        Parameters
            X: numpy.ndarray, shape=(n_samples, m_features), 
            y: class label, array-like, shape = (n_samples, )
        Returns
            self : object
        """            
        np.random.seed(self.random_seed)
        self.w = np.random.random(size=X.shape[1] + 1)
            
        self.maxy, self.miny = y.max(), y.min()
        self.cost_ = []
        self.w_ = np.array([self.w])
        
        """Momentum"""
        self.v1 = np.zeros_like(self.w[1:])
        self.v2 = np.zeros_like(self.w[0])
        gamma = 0.5

        for i in range(self.epochs):
            yhat = self.activation(self.net_input(X))
            errors = (y - yhat)
            
            self.v1 = gamma * self.v1 + self.eta * np.dot(errors, X)
            self.v2 = gamma * self.v2 + self.eta * np.sum(errors)
            
            self.w[1:] += self.v1 #self.eta * np.dot(errors, X)
            self.w[0] += self.v2 #self.eta * np.sum(errors)
            cost = 0.5 * np.sum(errors**2)
            self.cost_.append(cost)
            self.w_ = np.vstack([self.w_, self.w]) 
        return self

    def net_input(self, X):            
        """Compute the value of z, net input  """
        return np.dot(X, self.w[1:]) + self.w[0]

    def activation(self, X):  
        """Identity activation function: """
        return X

    def predict(self, X):      
        """Predict the class label with  """
        mid = (self.maxy + self.miny) / 2
        Z = self.net_input(X)
        yhat = self.activation(Z)
        return np.where(yhat > mid, self.maxy, self.miny)
