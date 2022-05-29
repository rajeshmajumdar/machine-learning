# Tensors are closely related to array, specially in machine learning tensors/arrays/vectors can be used
# interchangeably.

list_matrix_array = [[4, 2],
        [5, 1],
        [8, 2]]

# A simple 3x2 matrix, it's shape would be (3, 2) since it has 3 rows and 2 columns
# Similarly we can create n-dimensional matrices.
# Kinda similar is vector, mathematically we know what vector is but in python we can store it using lists
# only. But on both tensors and vectors, we do dot product and vector multiplications which are different
# from how we do in a simple array. So maybe in these cases all these three are different.

# So if we look at it this way, we have tensors as a container containing the vectors.

# Dot product of two vectors
##########
#  a.b = a1b1 + a2b2 + a3b3 ... anbn
# Here a and b are the vectors
##########

# As we do in maths, a dot product of two vectors is just a sum of products of consecutive vector element

a = [1, 2, 3]
b = [2, 3, 4]

# Here a and b are two vectors with 3 directions i^, j^ and k^. Now to do dot product on these two vectors.

dot_product = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
print(dot_product) # Output: 20

# Now if we recall back to our neuron.py, we did kinda similar thing if we consider a as the input vector
# and b as the weight vector

# This here is the dot product, we can also perform vector addition similarly, well the maths is different

############
# a + b = [a1 + b1, a2 + b2, a3 + b3 ... , an + bn]
############

# Now python is not really that good with handling numbers, vectors and tensors. So we have a library called numpy, we can use numpy to simply all this

import numpy as np

inputs= [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

output = np.dot(weights, inputs) + bias
print(output) # Output: 4.8

# now let's do this for multiple neurons.

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

layer_outputs = np.dot(weights, inputs) + biases
print(layer_outputs) # [4.8   1.21  2.385]

# now in real world application, we wouldn't have just single input vector, we can have multiple inputs
# vectors, so we can put together all of them in a matrix, just like we did with weights and perform the
# same operations. And this we call matrix multiplication

# Now if we remember how we use to do matrix product in school, I hated it, anyhow we keep multiplying and# adding the respective row and column and add them together to get a new matrix.
# For this to happen we need to have some thing beforehand i.e. the number of column in the first matrix 
# must be equal to the number of rows in the second matrix.
# m1.shape = (4, 5)
# m2.shape = (5, 7)

# We can multiply these two matrices
# in other words, row and column vectors are the matrices with on of there dimensions being of a size of 1
# and here we perform matrix product instead of dot product.

# another cool thing we used to do in school with matrices is to transpose it, by transposing we change
# columns to rows and the ros to columns, so kinda flip the matrix.

a = [1, 2, 3]
b = [2, 3, 4]

a = np.array([a]) # made it into a 1-d matrix
b = np.array([b]).T # and here we also transposed it

print(np.dot(a, b))

# now we are ready to handle multiple inputs, and work with them

inputs = [[1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]]

# If we just look at the shape of both of the matrices these are 3x4 and 3x4 respectively.
# but to do matrix multiplication we need to transpose on of them first so that we have a shape
# 3x4 and 4x3.

biases = [2.0, 3.0, 0.5]

outputs = np.dot(inputs, np.array(weights).T) + biases
# our output should be a new multiplied matrix.

print(outputs)





