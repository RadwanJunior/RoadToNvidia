# Lets assume prediction is the largest in the outputs -> but we want to train a model
layer_outputs = [4.8, 1.21, 2.385]

# layer_outputs = [4.8, 4.79, 4.25]

# Exponentiation

import math

E = math.e

# We want to exponentiate each of the layer outputs
exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)

print(exp_values)

# Normalization

norm_base = sum(exp_values)
norm_values = []

for value in norm_values:
    norm_values.append(value/norm_base)

print(norm_values)
print(sum(norm_values)) # Should add up to one

# Using numpy
import numpy as np
# import nnfs

# nnfs.init()

exp_values = np.exp(layer_outputs)

norm_values_2 = exp_values/np.sum(exp_values)

print(norm_values_2)
print(sum(norm_values_2)) # Should add up to one
