# example of gradient descent for a one-dimensional function
import numpy as np
from numpy import asarray
from numpy.random import rand, __all__
from numpy.random import seed

# objective function
def primary(x):
    return x**2.0

# derivative of objective function
def derivative(x):
    return x * 2.0

# gradient descent algorithm
def gradient_descent(primary, derivative, bounds, epoch, eta):
    # generate an initial point
    solution = np.random.uniform(bounds[0][0], bounds[0][1])
    solution_eval = None
    for i in range(epoch):
        gradient = derivative(solution)
        solution = solution - eta * gradient
        solution_eval = primary(solution)
        print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
    return [solution, solution_eval]

# We would expect that gradient descent with momentum will accelerate the \
# optimization procedure and find a similarly evaluated solution in fewer iterations.

def gradient_descent_with_momentum(primary, derivative, bounds, epoch, eta, momentum):
    solution = np.random.uniform(bounds[0][0], bounds[0][1])
    solution_eval = None
    # keep track of the change
    change = 0.0
    for i in range(epoch):
        gradient = derivative(solution)
        new_change = eta * gradient + momentum * change
        solution = solution - new_change
        change = new_change
        solution_eval = primary(solution)
        print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
    return [solution, solution_eval]


seed(4)
bounds = asarray([[-1.0, 1.0]])
epoch = 13
eta  = 0.1
momentum = 0.9

best, score = gradient_descent(primary, derivative, bounds, epoch, eta)
print('Done!')
print('f(%s) = %f' % (best, score))
print('\n')

best, score = gradient_descent_with_momentum(primary, derivative, bounds, epoch, eta, momentum)
print('Done!')
print('f(%s) = %f' % (best, score))