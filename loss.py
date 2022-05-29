# Similar to activation functions, we have multiple loss functions too e.g. Mean Squared Error, should be familiar with this, since it is heavily used in linear regression. Another one is Categorical Cross-entropy, this is used in probabilities and mostly with softmax activation function.

# And this is where i need to read more about probabilities. And here I thought probability is the easiest topic in mathematics

import math

# An example of softmax_ouput
softmax_output = [0.7, 0.1, 0.2]
# Ground truth
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] +
        math.log(softmax_output[1]) * target_output[1] +
        math.log(softmax_output[2]) * target_output[2])

print(loss)

# this should already provide us with the necessary tooling we need to calculate the loss, in our case i.e. a classification problem. Different problems might need different
# type of loss function. 



