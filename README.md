## Machine Learning

Just some random projects to get me familiar with machine learning.

> [Naive Bayes](https://github.com/rajeshmajumdar/machine-learning/blob/master/naive_bayes.py)

### [Neural Network](https://github.com/rajeshmajumdar/machine-learning/blob/master/neural_network.py)
Implementing a basic neural network without using any framework like Tensorflow or pyTorch. It helps me understand the nuances and the mathematics that goes behind building a neural network. 

- Modules Used
1. Numpy (For matrices and basic linear algebra)
2. Matplotlib (To visualize the results)
3. Pandas (Simply to play around with data)

### [Sign Language NN](https://github.com/rajeshmajumdar/machine-learning/blob/master/sign_language_nn.py)
Tried implementing the same neural network with some tweaks in the hidden and output layer. And also added some extra functions like saving and loading weights and biases.
Currently, I was able to get 70% accuracy on train_data and ~65% accuracy on test_data. I think it could be better, if I add more hidden layers in between that's for later.

Same modules used as before.

### [Neuron](https://github.com/rajeshmajumdar/machine-learning/blob/master/neuron.py)
Implementation of a basic neuron taking multiple inputs, and a layer of neurons connected to each other with 4 inputs.

First I hardcoded each neuron to better understand the nuances of a neuron and the maths goes behind it, next I used loops to make it more dynamic, so we can change the number of neurons and inputs and it could handle it

### [Tensors](https://github.com/rajeshmajumdar/machine-learning/blob/master/tensors.py)
In this I added some more complexity to our neuron, like handling multiple input vectors and some basic maths we all have learned in 11th-12th standard.

### [Layers](https://github.com/rajeshmajumdar/machine-learning/blob/master/layers.py)
In this we created a class object which take number of neuron and the input vectors as parameters and give us the layer of neuron.
So that we don't always need to hardcode and repeat our code.

### [Activation Functions](https://github.com/rajeshmajumdar/machine-learning/blob/master/activation.py)
Here we get to understand why we need to use activation and why we have multiple choices and how we can choose which activation function we should use in which case.

### [Loss Functions](https://github.com/rajeshmajumdar/machine-learning/blob/master/loss.py)
In this file, I just hardcoded what cross-entropy loss function does to get the loss from the softmax output, although there are many loss functions and depending on the problem loss functions differ

### [ANN](https://github.com/rajeshmajumdar/machine-learning/blob/master/nn.py)
A full artificial feed forward neural network, tested and trained on some random data, with accuracy of ~83%. Here I coded pretty much everything needed to build a bare minimum feed forward network in python without using any 3rd party deep learning libraries except numpy and nnfs (for random dataset).

#### Next Steps

Now since, I am somewhat familiar with simple ANNs, next goals for me is to get familiar with other neural network architectures like CNN, RNN, Autoencoders and so on.