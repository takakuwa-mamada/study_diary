import numpy as np

def cross_entropy_error(y, t):
    if y.dim == 1:
        t = t.shape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size