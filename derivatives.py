'''
Let's first try to understand what a derivative is and what it does to a function.

Before this let's try to understand why we actually need to use derivatives here, this might give us an answer to what derivative
does to a function.

So to change our weights and biases of each neuron, we first need to understand how much these weights and biases affecting our loss

And if we somehow know much our weights and biases are affecting our loss, we can tweak our w and b to our liking to minimizing the
loss function. Also now let's think all of our neural network as a big maths equation. 

Some imput value are going through some addition and changes from the first layer these values are somehow dependent of that layers
respective w and b, and then it goes to the second layer and again some mathematical magic happens there which also depends on the
respective w and b and finally we get some value at the end, now this output value defines our loss. So in a way our loss function
is dependent on the respective w and b. And depending on the neural network and the activation functions we used, these w and b 
affects our loss differently. 
So if we think of it as a mathematical equation, our w and b might not be non-linear in every case it could be in higher dimension.

That's why we use derivative more precisely partial derivatives, which gives us a gradient and that's where the term gradient descent
comes from. Derivatives are just some special cases of partial derivatives.

Now let's understand what and how derivatives tells us the relation between variables.
'''

import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2 * x

x = np.arange((5))
y = f(x)

print(x)
print(y)

''' Output:
        x = [0 1 2 3 4]
        y = [0 2 4 6 8]
'''

# plt.plot(x, y)
# plt.show()

## as expected it gives us a straight line, since we have a equation, y = 2x. which is basically the equation of the line
# y = mx + c, where m = 2, and c = 0. This also might feel familiar with what we did in each of the neuron
# output = input * weight + bias, here y = output, x = input, m = weight, and c = bias.

# Now we if we need to find out how x is affecting the value of y, we can find it using slope of line. i.e. change in y / change in x
# in layman terms: how much of our y has changed if we change our x. And we call this slope in terms of line.

'''
slope = change in y / change in x
= (y2 - y1) / (x2 - x1)

So we simply take two points on the line and just subtract the respective ones. We have been doing it since class 11th in vector
subtraction. I mentioned vectors, because in higher dimensions we can't simply use lines, our w and b might not be non-linear like
in this case.
'''

slope = (y[2] - y[1]) / (x[2] - x[1])

print(f"Slope: {slope}") # We are expecting this to be our m i.e. 2 in this case, you might play with this to better understand this part.

'''
So what this is telling us, that everytime we change x, y is changed twice. Now just to understand why we are using derivatives here.

So our loss equation as: y = f(w, b)
By differentiating this f(w, b) with respective to w and b, we can get the slope or in higher dimensions case a gradient.
'''

## now let's different another function i.e. y = f(x^2), this is not a linear function. 

def f2(x):
    return x**2

y = f2(x)

print(f"x: {x}")
print(f"y=x^2: {y}")

# plt.plot(x, y)
# plt.show()

# we should see an exponential line
'''
Now we can't use the previous formula to calculate the slope, since it no longer a linear equation, so if we calculate slopes
at different x and y, we would get different values
'''

slope1 = (y[2] - y[1]) / (x[2] - x[1])
slope2 = (y[3] - y[2]) / (x[3] - x[2])

print(f"Slope1: {slope1}\nSlope2: {slope2}")  # as expected slopes are different that's here we use derivatives to solve this problem

# This we can solve by using traditional way of solving differentiation, since i already love calculus I know how to do this.

def differentiate(x):
    delta = 0.0001
    x1 = x
    x2 = x1 + delta

    y1 = f2(x1)
    y2 = f2(x2)

    slope = (y2 - y1) / (x2 - x1)
    print(f"Slope at {x}: {slope}")


# So for this equation we can find slopes at every point x and make a gradient of it.

differentiate(1) # this should be 2
differentiate(2) # this should be 4
differentiate(3) # this should be 6

'''
Just looking at this we can conclude that, change in y with respect to x is dependent on the value of x. 
that's why y = x ^ 2 or y = x * x

I am not gonna visualize this since i already know it, but you can use matplotlib to visualize this more and 
understand derivatives deeply.
'''



