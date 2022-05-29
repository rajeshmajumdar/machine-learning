# Random values

inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

# This is a single neuron with 3 inputs. So a neuron, sums each input multiplied by it's weight and then adds the bias.

out = (inputs[0]*weights[0] + 
        inputs[1]*weights[1] + 
        inputs[2]*weights[2] + bias)

print(out) # we are getting an output of 2.3 here.

# This here is a single neuron with 3 inputs, we can change the number of inputs this neuron might have by simply adding the input in the input's array and adding it's respective weight
# Later during the learning process we will tune these biases and weights according to our liking

### LAYER OF NEURONS
####################

# Next we wanna add 3 neurons in our input layer, each neuron takes in 4 different output

inputs = [1, 2, 3, 2.5] # neuron with 4 inputs

weights1 = [0.2, 0.8, -0.5, 1]  # weights for the first neuron
weights2 = [0.5, -0.91, 0.26, -0.5] # weights for the second neuron
weights3 = [-0.26, -0.27, 0.17, 0.87] # weights for the third neuron

bias1 = 2 # bias for the first neuron
bias2 = 3 # bias for the second neuron
bias3 = 0.5 # bias for the third neuron

output = [
        # Neuron 1:
        inputs[0] * weights1[0] +
        inputs[1] * weights1[1] +
        inputs[2] * weights1[2] +
        inputs[3] * weights1[3] + bias1,

        # Neuron 2:
        inputs[0] * weights2[0] +
        inputs[1] * weights2[1] +
        inputs[2] * weights2[2] +
        inputs[3] * weights2[3] + bias2,

        # Neuron 3:
        inputs[0] * weights3[0] +
        inputs[1] * weights3[1] +
        inputs[2] * weights3[2] +
        inputs[3] * weights3[3] + bias3]

print(output)  # output: [4.8, 1.21, 2.385]

# In this previous code we hardcoded everything and just to add 3 neurons with 4 inputs in the first layer with one output, we have to code this much, now if we have to add a ton of these it would get a lot of messy
# Here we can use matrices and some neat coding stuff.

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)


print(layer_outputs)
