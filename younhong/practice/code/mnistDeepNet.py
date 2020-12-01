#%load code/mnistDeepNet.py
# deep neural net
# version 0.1
# author: idebtor@gmail.com 

import sys
class DeepNeuralNet(object):
    """ implements a deep neural net. 
        Users may specify any number of layers.
        net_arch -- consists of a number of neurons in each layer 
    """
    def __init__(self, net_arch, activate = None, eta = 1.0, epochs = 100, random_seed = 1):
        self.eta = eta
        self.epochs = epochs
        self.net_arch = net_arch
        self.layers = len(net_arch)
        self.W = []
        self.random_seed = random_seed
        
        self.g       = [lambda x: sigmoid(x)   for _ in range(self.layers)]
        self.g_prime = [lambda x: sigmoid_d(x) for _ in range(self.layers)]
        
        if activate is not None:
            for i, (g, g_prime) in enumerate(zip(activate[::2], activate[1::2])):
                self.g[i+1] = g
                self.g_prime[i+1] = g_prime
                
        for i in range(len(self.g)):
            print(type(self.g[i]), id(self.g[i]))
        
        #print('X.shape={}, y.shape{}'.format(X.shape, y.shape))
        # Random initialization with range of weight values (-1,1)
        np.random.seed(self.random_seed)
        
        # A place holder [None] is used to indicated "unused place".
        self.W = [[None]]    ## the first W0 is not used.
        for layer in range(self.layers - 1):
            w = 2 * np.random.rand(self.net_arch[layer+1], 
                                   self.net_arch[layer]) - 1
            print('layer:', layer, 'shape:', w.shape)
            self.W.append(w)  
        print('Weight:', self.W)
            
    def forpass(self, A0):     
        Z = [[None]]   # Z0 is not used.
        A = []       # A0 = X0 is used. 
        A.append(A0)
        for i in range(1, len(self.W)):
            z = np.dot(self.W[i], A[i-1])
            Z.append(z)
            a = self.g[i](z)
            A.append(a)
        return Z, A
    
    def backprop(self, Z, A, Y):
        # initialize empty lists to save E and dZ
        # A place holder None is used to indicated "unused place".
        E  = [None for x in range(self.layers)]
        dZ = [None for x in range(self.layers)]
        
        # Get error at the output layer or the last layer
        ll = self.layers - 1
        error = Y - A[ll]
        E[ll] = error   
        dZ[ll] = error * self.g_prime[ll](Z[ll]) 
        
        # Begin from the back, from the next to last layer
        for i in range(self.layers-2, 0, -1):
            E[i]  = np.dot(self.W[i+1].T, E[i+1])
            dZ[i] = E[i] * self.g_prime[i](Z[i])
       
        # Adjust the weights, using the backpropagation rules
        m = Y.shape[0] # number of samples
        for i in range(ll, 0, -1):
            self.W[i] += self.eta * np.dot(dZ[i], A[i-1].T) / m
        return error
         
    def fit(self, X, y):
        print('fit')
        self.cost_ = []        
        for epoch in range(self.epochs):          
            Z, A = self.forpass(X)        
            cost = self.backprop(Z, A, y)   
            self.cost_.append(
                 np.sqrt(np.sum(cost * cost)))    
        return self

    def predict(self, X):
        print('predict')
        A0 = np.array(X, ndmin=2).T         # A0: inputs
        Z, A = self.forpass(A0)     # forpass
        return A[-1]                                       
   
    def evaluate(self, Xtest, ytest):       # fully vectorized calculation
        print('evaluate')
        m_samples = len(ytest)
        scores = 0        
        A3 = self.predict(Xtest)
        yhat = np.argmax(A3, axis = 0)
        scores += np.sum(yhat == ytest)
        return scores/m_samples * 100
    
