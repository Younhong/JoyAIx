#%load code/MnistBGD.py
import numpy as np
class MnistBGD_LS(object):
    """ Batch Gradient Descent
    """
    def __init__(self, n_x, n_h, n_y, eta = 0.1, epochs = 100, random_seed=1):
        """ 
        """
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.eta = eta
        self.epochs = epochs
        np.random.seed(random_seed)
        self.W1 = 2*np.random.random((self.n_h, self.n_x)) - 1  # between -1 and 1
        self.W2 = 2*np.random.random((self.n_y, self.n_h)) - 1  # between -1 and 1
                
    def forpass(self, A0):
        Z1 = np.dot(self.W1, A0)          # hidden layer inputs
        A1 = self.g(Z1)                      # hidden layer outputs/activation func
        Z2 = np.dot(self.W2, A1)          # output layer inputs
        A2 = self.g(Z2)                       # output layer outputs/activation func
        return Z1, A1, Z2, A2

    def fit(self, X, y):
        self.cost_ = []
        m_samples = len(y)       
        Y = joy.one_hot_encoding(y, self.n_y)       # (m, n_y) = (m, 10)   one-hot encoding
        
        # learning rate is scheduled to decrement by a step of which the inteveral from 0 to eta
        # eqaully divided by total number of iterations (or epochs * m_samples)
        learning_schedule = np.linspace(self.eta, 0.0001, self.epochs)
        
        # for momentum
        #self.v1 = np.zeros_like(self.W1)
        #self.v2 = np.zeros_like(self.W2)
        
        for epoch in range(self.epochs):
            if (epoch) % 100 == 0:
                print('Training epoch {}/{}.'.format(epoch + 1, self.epochs))

            # input X can be tuple, list, or ndarray
            A0 = np.array(X, ndmin=2).T       # A0 : inputs, minimum 2 dimensional array
            Y0 = np.array(Y, ndmin=2).T       # Y: targets

            Z1, A1, Z2, A2 = self.forpass(A0)   # forward pass

            E2 = Y0 - A2                           # E2: output errors
            E1 = np.dot(self.W2.T, E2)          # E1: hidden errors

            # back prop and update weights
            #self.W2 += self.eta * np.dot(E2 * A2 * (1.0 - A2), A1.T)
            #self.W1 += self.eta * np.dot(E1 * A1 * (1.0 - A1), A0.T)

            # back prop, error prop
            dZ2 = E2 * self.g_prime(Z2)        # backprop      # dZ2 = E2 * A2 * (1 - A2)  
            dZ1 = E1 * self.g_prime(Z1)        # backprop      # dZ1 = E1 * A1 * (1 - A1)  

            # udpate weight with momentum
            # eta = learning_schedule[epoch]
            #self.v2 = 0.9 * self.v2 + self.eta * np.dot(dZ2, A1.T) / m_samples
            #self.v1 = 0.9 * self.v1 + self.eta * np.dot(dZ1, A0.T) / m_samples
            #self.W2 += self.v2     
            #self.W1 += self.v1 
            
            # update weights without momentum
            #eta = learning_schedule[epoch]           
            self.W2 += self.eta * np.dot(dZ2, A1.T) / m_samples     
            self.W1 += self.eta * np.dot(dZ1, A0.T) / m_samples     

            self.cost_.append(np.sqrt(np.sum(E2 * E2)))
        return self

    def predict(self, X):
        A0 = np.array(X, ndmin=2).T         # A0: inputs
        Z1, A1, Z2, A2 = self.forpass(A0)   # forpass
        return A2                                       

    def g(self, x):                 # activation_function: sigmoid
        x = np.clip(x, -500, 500)   # prevent from overflow, 
        return 1.0/(1.0+np.exp(-x)) # stackoverflow.com/questions/23128401/
                                    # overflow-error-in-neural-networks-implementation
    
    def g_prime(self, x):                    # activation_function: sigmoid derivative
        return self.g(x) * (1 - self.g(x))
    
    def evaluate(self, Xtest, ytest):       # fully vectorized calculation
        m_samples = len(ytest)
        scores = 0        
        A2 = self.predict(Xtest)
        yhat = np.argmax(A2, axis = 0)
        scores += np.sum(yhat == ytest)
        return scores/m_samples * 100
    
    def evaluate_onebyone(self, Xtest, ytest):
        m_samples = len(ytest)
        scores = 0
        for m in range(m_samples):
            A2 = nn.predict(Xtest[m])
            yhat = np.argmax(A2)
            if yhat == ytest[m]:
                scores += 1
            #if m < 5:
                #print('A2.shape', A2.shape)
                #print('A2={}, yhat={}, ytest={}'.format(A2, yhat, ytest[m]))
        
        return scores/m_samples * 100
    
