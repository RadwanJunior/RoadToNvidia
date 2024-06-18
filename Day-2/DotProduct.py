import numpy as np

# Dot product with a single neuron

# inputs = [1,2,3,2.5]
# weights = [0.2, 0.8, -0.5, 1.0]
# bias = 2

# output = np.dot(inputs, weights) + bias
# print(output)

# Dot product of layer of neurons
# Modelling 3 neurons -> we pass weights first since we want it indexed by weights
inputs = [1, 2, 3, 2.5]

weights= [[0.2,0.8,-0.5],
          [0.5,-0.91,0.26, -0.5],
          [-0.26,-0.27, 0.17, 0.87]]

biases = [2,3,0.5]

output = np.dot(weights, inputs) + biases
print(output)