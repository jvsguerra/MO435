import numpy as np
from math import e, sin, pi

def kernelSE(x1, x2, ell=1.0, s=1.0):
    "Squared-Exponential kernel"
    isScalar = np.isscalar(x1)
    if not isScalar:
        isScalar = len(x1) == 1

    if not isScalar:
        # Following Equation 15.20 of MLPP
        M = ell**-2 * np.identity(len(x1))
        return (s ** 2) * e ** (-1/2 * (x1 - x2).T.dot(M).dot(x1 - x2)) + (s ** 2)
    else:
        return (s ** 2) * e ** (-1 * ((x1 - x2) ** 2) / (2 * ell ** 2))

def kernelOU(x1, x2, ell=1.0, s=1.0):
    "Ornstein-Uhlenbeck kernel"
    return (s ** 2) * e ** (-1 * (abs(x1 - x2))/(2 * ell ** 2))

def kernelRQ(x1, x2, ell=1.0, s=1.0, alpha=1.0):
    "Rational Quadratic kernel"
    return (s ** 2) * (1 + ((x1 - x2) ** 2/(2 * alpha * ell ** 2))) ** (-alpha)

def kernelESS(x1, x2, ell=1.0, s=1.0, w=1.0):
    "Exponential Sine Squared kernel (periodic)"
    return (s ** 2) * e ** (-2 * (sin(pi / w * abs(x1 - x2)) ** 2) / ell ** 2)


# Reference:
# kernel functions: 
# - http://www.gaussianprocess.org/gpml/chapters/RW.pdf
# - https://www.cs.toronto.edu/~duvenaud/cookbook/
