from GaussianProcesses import GP
from functions import sampling1D, sampling2D
from kernels import kernelSE
import numpy as np


def evaluate_kernels(kernels, sigma, func):
    """
    kernels: list of kernels to compare
    sigma: noise standard deviation
    """
    # Iterate for different kernels
    for name, kernel in kernels.items():
        # Sample function
        xtrain, ytrain = sampling1D(50, 1, func)
        # GP without noise
        gp = GP(kernel, 0.0)
        gp.addData(xtrain, ytrain)
        gp.samplingGP(f"output/kernels/prior/noise-free/{name}.png")
        # GP with noise
        gp = GP(kernel, sigma)
        gp.addData(xtrain, ytrain)
        gp.samplingGP(f"output/kernels/prior/noisy/{name}.png")
        # GP posterior
        xtest = np.linspace(-5, 5, 50)
        mu, s2 = gp.posterior(xtest)
        # Plot posterior
        gp.plot_posterior(mu, s2, 2.5, func, 50, f'output/kernels/posterior/{name}.png')


def evaluate_train_size(train_size, test_size, sigma, func, kernel):
    for sample in train_size:
        # Training set
        xtrain, ytrain = sampling1D(sample, 1, func)
        # GP prior
        gp = GP(kernel, sigma)
        gp.addData(xtrain, ytrain)
        # GP posterior
        xtest = np.linspace(-5, 5, test_size)
        mu, s2 = gp.posterior(xtest)
        # Plot posterior
        gp.plot_posterior(mu, s2, 2.5, func, test_size, f'output/train_size/{func.__name__}_{sample}.png')


def evaluate_test_size(train_size, test_size, sigma, func, kernel):
    for sample in test_size:
        # Training set
        xtrain, ytrain = sampling1D(train_size, 1, func)
        # GP prior
        gp = GP(kernel, sigma)
        gp.addData(xtrain, ytrain)
        # GP posterior
        xtest = np.linspace(-5, 5, sample)
        mu, s2 = gp.posterior(xtest)
        # Plot posterior
        gp.plot_posterior(mu, s2, 2.5, func, sample, f'output/test_size/{func.__name__}_{sample}.png')


def evaluate_ell_kernelSE(ell, s, func):
    for value in ell:
        # Define kernel
        kernel = lambda x1, x2 : kernelSE(x1, x2, ell=value, s=s)
        # Define noise
        sigma = 0.01
        # Training set
        xtrain, ytrain = sampling1D(50, 1, func)
        # GP prior
        gp = GP(kernel, sigma)
        gp.addData(xtrain, ytrain)
        # GP posterior
        xtest = np.linspace(-5, 5, 50)
        mu, s2 = gp.posterior(xtest)
        # Plot posterior
        valuename = str(value).replace(".", "_")
        gp.plot_posterior(mu, s2, 2.5, func, 50, f'output/kernelSE/ell/{valuename}.png')


def evaluate_s_kernelSE(s, ell, func):
    for value in s:
        # Define kernel
        kernel = lambda x1, x2 : kernelSE(x1, x2, ell=ell, s=value)
        # Define noise
        sigma = 0.01
        # Training set
        xtrain, ytrain = sampling1D(50, 1, func)
        # GP prior
        gp = GP(kernel, sigma)
        gp.addData(xtrain, ytrain)
        # GP posterior
        xtest = np.linspace(-5, 5, 50)
        mu, s2 = gp.posterior(xtest)
        # Plot posterior
        valuename = str(value).replace(".", "_")
        gp.plot_posterior(mu, s2, 2.5, func, 50, f'output/kernelSE/s/{valuename}.png')


def evaluate_sigma(sigma, func, kernel):
    for value in sigma:
        # Training set
        xtrain, ytrain = sampling1D(50, 1, func)
        # GP prior
        gp = GP(kernel, sigma=value)
        gp.addData(xtrain, ytrain)
        # GP posterior
        xtest = np.linspace(-5, 5, 50)
        mu, s2 = gp.posterior(xtest)
        # Plot posterior
        valuename = str(value).replace(".", "_")
        gp.plot_posterior(mu, s2, 2.5, func, 50, f'output/kernelSE/noise/{valuename}.png')
