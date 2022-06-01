'''
Now we have all the tools we need to backpropogate, or simply optimize our neural network.


'''


def tmp():
    x = [1.0, -2.0, 3.0]
    w = [-3.0, -1.0, 2.0]
    b = 1.0

    # this is just one single neuron with inputs. so we can do x1 * w1 + x2 * w2 + x3 * w3 + b

    xw0 = x[0] * w[0]
    xw1 = x[1] * w[1]
    xw2 = x[2] * w[2]

    print(xw0, xw1, xw2)

    # adding them all together

    res = xw0 + xw1 + xw2 + b
    print(res)

    # next we want to add relu to this output

    relu = max(res, 0)
    print(relu)

    # now this our full forward pass from a single neuron and a relu activation function

    '''
    Consider a mathematical equation for this:
    
    ReLU(xw0 + xw1 + xw2 + b)

    y = ReLU(sum(mul(x[0], w[0]), mul(x[1], w[1]), mul(x[2], w[2]), b))

    Looking at it like this, it looks a simple linear equations we can apply chain rule here to find it's derivative.

    y' = [ReLU() / sum()]' . [sum() / mul(x, w)]' . [mul(x, w) / x]'

    So going backward in our equation, 

    [ReLU() / sum()]' = 1 (x > 0)
    '''

    relu_dz = (1. if res > 0 else 0)

    # since the res is > 0 i.e. 6. The derivative will be 1.
    


if __name__ == '__main__':
    tmp()