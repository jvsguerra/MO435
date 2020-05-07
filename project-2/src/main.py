from GaussianProcesses import GP
from parameters_evaluation import evaluate_kernels, evaluate_train_size, evaluate_test_size, evaluate_ell_kernelSE, evaluate_s_kernelSE, evaluate_sigma
from kernels import kernelSE, kernelOU, kernelRQ, kernelESS
from functions import sampling1D, sampling2D, func1, func2, func3, func2d, plot_functions
import numpy as np
import matplotlib.pyplot as plt

# Set seed 
np.random.seed(0)

# Functions illustration
plot_functions()

####################### Experiments #######################
# > Comparison 1:
# Different training sample sizes:
# - sample_size = 5, 10 and 50
sample_size = [5, 10, 50]
evaluate_train_size(train_size=sample_size, test_size=50, sigma=0.01, func=func1, kernel=lambda x1, x2 : kernelSE(x1, x2, ell=1.0, s=1.0))
evaluate_train_size(train_size=sample_size, test_size=50, sigma=0.01, func=func2, kernel=lambda x1, x2 : kernelSE(x1, x2, ell=1.0, s=1.0))
evaluate_train_size(train_size=sample_size, test_size=50, sigma=0.01, func=func3, kernel=lambda x1, x2 : kernelSE(x1, x2, ell=1.0, s=1.0))

# > Comparison 2:
# Different test sample sizes:
# - sample_size = 5, 10 and 50
sample_size = [5, 10, 50]
evaluate_test_size(train_size=50, test_size=sample_size, sigma=0.01, func=func1, kernel=lambda x1, x2 : kernelSE(x1, x2, ell=1.0, s=1.0))
evaluate_test_size(train_size=50, test_size=sample_size, sigma=0.01, func=func2, kernel=lambda x1, x2 : kernelSE(x1, x2, ell=1.0, s=1.0))
evaluate_test_size(train_size=50, test_size=sample_size, sigma=0.01, func=func3, kernel=lambda x1, x2 : kernelSE(x1, x2, ell=1.0, s=1.0))

# Comparison 3:
# Different kernel parameters
# KernelSE: ell and s
# - Effect of ell (horizontal scale)
ell = [0.1, 1, 2, 5, 10, 100]
evaluate_ell_kernelSE(ell, s=1.0, func=func1)
# - Effect of s (vertical scale)
s = [0.01, 0.1, 0.5, 1, 10, 100]
evaluate_s_kernelSE(s=s, ell=1.0, func=func1)

# Comparison 4:
# Different noise (sigma)
sigma = [0.001, 0.01, 0.1, 1, 2, 10]
evaluate_sigma(sigma=sigma, func=func1, kernel=kernelSE)

# > Comparison 5: 
# Different kernels
kernels={'kernelSE': kernelSE, 'kernelOU': kernelOU, 'kernelRQ': kernelRQ, 'kernelESS': kernelESS}
evaluate_kernels(kernels, sigma=0.01, func=func1)

# > 2D Gaussian Process
xtrain, ytrain = sampling2D(100, 2, func2d)
# GP with kernel
kernel = lambda x1, x2 : +kernelSE(x1, x2, ell=1.0, s=1.0)
gp = GP(kernel, sigma=0.01)
gp.addData(xtrain, ytrain)
# GP posterior
y = x = np.linspace(-5, 5, 30)
x, y = np.meshgrid(x, y)
xtest = np.array([x.reshape(30*30), y.reshape(30*30)]).T
mu, s2 = gp.posterior(xtest)
gp.plot_posterior2D(mu, s2, func2d, 30, 'output/GP2D/posterior.png')
# ###########################################################