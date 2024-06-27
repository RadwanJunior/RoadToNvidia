import numpy as np
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.0, 0.026]]


exp_values = np.exp(layer_outputs)

# norm_values = exp_values / np.sum(exp_values)

# print(norm_values)
# print(sum(norm_values))


#We are only getting one value, but we want 3 values for the batch -> pass axes as paramter
# Axis=0 is the sum of columns
# Axis=1 is the sum of rows
print(np.sum(layer_outputs, axis=1, keepdims=True))

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)

