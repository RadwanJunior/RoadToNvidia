import numpy as np
## MANUALLY DOING 2 LAYERS
# inputs = [[1, 2, 3, 2.5],
#           [2.0, 5.0, -1.0, 2.0],
#           [-1.5, 2.7, 3.3, -0.8]]

# weights= [[0.2,0.8,-0.5, 1.0],
#           [0.5,-0.91,0.26, -0.5],
#           [-0.26,-0.27, 0.17, 0.87]]

# biases = [2,3,0.5]

# weights2= [[0.1,-0.14, 0.5],
#           [-0.5,0.12, -0.33],
#           [-0.44,0.73, -0.13]]

# biases2 = [-1,2,-0.5]

# layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
# #First layer used as imputs for 2nd layer. Very easily to add another layet to neural network. Can add more nueron but easier to use objects
# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# print(layer2_outputs)

np.random.seed(0)

#Assume input data to neural network
X = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]



# Define 2 hidden layers
# Create Object for layer

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #Shape it makes for us. Arguemnt passed is what the shape will be. Shape will be # of inputs by number of neurons, to prevent transpose in forward method
        self.weights = np.random.randn(n_inputs, n_neurons)
        #Biases it creates for us. Argument passed is the shape (in tuple)
        self.biases = np.zeros((1,n_neurons))
    #inputs will either be X or the previous layer
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

#4 is the number of features in the input, 2nd argument is how many neurons you want in the layer (can be anything)
layer1 = Layer_Dense(4, 5)
# 1st argument has to be same as 2nd argument in previous layer
layer2 = Layer_Dense(5,2)

layer1.forward(X)
# print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)


# Gausian distribution bounded around 0. Some are bigger than 1, so we used 0.10*
# print(0.10*np.random.randn(4, 3))