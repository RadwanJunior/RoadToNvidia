# # Solving for x
# # e ** x = b
# import numpy as np
# import math

# b = 5.2

# print(np.log(b))
# # 5.2 but floating point precision issue
# print(math.e ** np.log(b))

import math

softmax_output = [0.7, 0.1, 0.2] # made up values of output from a softmax activation function from the output layer of your nn
target_output = [1,0,0]

# target_class = 0

loss = -(math.log(softmax_output[0])*target_output[0] + 
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

print(loss)

# Simplifies to this
loss = -math.log(softmax_output[0])
print(loss)

# Which is this in our case
loss = -math.log(0.7)
print(loss)

