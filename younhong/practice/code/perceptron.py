import numpy as np
def perceptron(X, y, w=None, eta=0.1, epochs=5, random_seed=1):
    if w is None:
        np.random_seed(random_seed)
        w = np.random.random((X.shape[1],1))
    
    maxlabel, minlabel = y.max(), y.min()
    
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            xi = xi.reshape(w.shape)
            z = np.dot(w.T, xi)
            yhat = np.where(z > 0.0, maxlabel, minlabel)
            delta = eta * (yi - yhat) * xi
            w += delta
    return w
