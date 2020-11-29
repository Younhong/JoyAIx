# xor.py-A very simple neural network to do exclusive or.
# use WijT and column vector style data

import numpy as np

g = lambda x: 1/(1 + np.exp(-x))       # activation function
g_prime = lambda x: g(x) * (1 - g(x))   # derivative of sigmoid

epochs = 2000

X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])                #=(1, 4),  but [0, 1, 1, 0].shape = (4, ) 

n_x = X.shape[0]
n_y = Y.shape[0]
n_h = 3

np.random.seed(1)
W1 = 2*np.random.random((n_h, n_x)) - 1
W2 = 2*np.random.random((n_y, n_h)) - 1  
print('X.shape={}, Y.shape{}'.format(X.shape, Y.shape))
print('n_x={}, n_h={}, n_y={}'.format(n_x, n_h, n_y))
print('W1.shape={}, W2.shape={}'.format(W1.shape, W2.shape))
cost_ = []

for i in range(epochs):
    A0 = X                             # unnecessary, but to illustrate only
    Z1 = np.dot(W1, A0)           # hidden layer input
    A1 = g(Z1)                        # hidden layer output
    Z2 = np.dot(W2, A1)           # output layer input
    A2 = g(Z2)                        # output layer results
    
    E2 = Y - A2                       # error @ output
    E1 = np.dot(W2.T, E2)          # error @ hidden
    if i == 0:
        print('E1.shape={}, E2.shape={}'.format(E1.shape, E2.shape))

    dZ2 = E2 * g_prime(Z2)        # backprop      # dZ2 = E2 * A2 * (1 - A2)  
    dZ1 = E1 * g_prime(Z1)        # backprop      # dZ1 = E1 * A1 * (1 - A1)  
    
    W2 +=  np.dot(dZ2, A1.T)     # update output layer weights
    W1 +=  np.dot(dZ1, A0.T)       # update hidden layer weights
    cost_.append(np.sum(E2 * E2))

print('fit returns A2:', A2)

print("Final prediction of all")
for x, yhat in zip(X.T, A2.T):
    print(x, np.round(yhat, 3))
