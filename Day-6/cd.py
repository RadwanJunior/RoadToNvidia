import numpy as np

np.random.seed(0)

# https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c
# Argument y is # of feature sets, argument 2 is # of classes. In each feature set we have 2 features
def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    Y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0,1,points) # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        Y[ix] = class_number
    return X, Y


import matplotlib.pyplot as plt

print("here")
X, Y = create_data(100, 3)