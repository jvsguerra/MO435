import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
from matplotlib.colors import LogNorm


def covarianceMatrix(x1, x2, kernel):
    matrix = np.zeros((len(x1), len(x2)))
    for x1index in range(len(x1)):
        for x2Index in range(len(x2)):
            matrix[x1index][x2Index] = kernel(x1[x1index], x2[x2Index])
    return matrix
    
class GP:
    def __init__(self, kernel, sigma):
        self.kernel = kernel
        self.sigma = sigma

    def addData(self, xtrain, ytrain):
        self.x = xtrain
        self.y = ytrain
        self.Ky = covarianceMatrix(self.x, self.x, self.kernel) + np.identity(self.x.shape[0])*self.sigma
    
    def posterior(self, xtest):
        # Covariance
        # cov = [[Ky, K*], [K*', K**]]; K*=Ks, K**=Kss
        Ks = covarianceMatrix(self.x, xtest, self.kernel)
        Kss = covarianceMatrix(xtest, xtest, self.kernel)
        
        # Cholesky decomposition
        # K = L * L'
        # K- = L-' * L-; L-=Li
        L = np.linalg.cholesky(self.Ky)
        Li = np.linalg.inv(L)
        alpha = Li.T.dot(Li.dot(self.y))
        v = Li.dot(Ks)

        # Parameters: mu, s2
        mu = Ks.T.dot(alpha)
        s2 = Kss - v.T.dot(v)

        print(self.log_likelihood(alpha, L, self.y.shape[0]))

        return mu, s2

    def log_likelihood(self, alpha, L, N):
        return -(1/2) * self.y.T.dot(alpha) - np.diag(L).sum() - (N/2) * np.log(2 * np.pi)

    def samplingGP(self, filename, n_sample=5, lower_bound=-5, upper_bound=5):
        x = np.arange(lower_bound, upper_bound, 0.1)
        y = [0 for i in x]
        K = covarianceMatrix(x, x, self.kernel) + np.identity(x.shape[0])*self.sigma
        # Plot samples from 5 samples GP
        colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        plt_vals = []
        for i in range(0, n_sample):
            ys = np.random.multivariate_normal(y, K)
            plt_vals.extend([x, ys, colors[i]])
        # Plot
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(*plt_vals)
        ax.set_ylabel('y = f(x)')
        ax.set_xlabel('x')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def plot_posterior(self, mu, s2, confidence, func, test_size, filename):
        fig = plt.figure()
        x = np.linspace(-5, 5, test_size)
        std = np.sqrt(np.diag(s2))
        # Plot the distribution of the function (mean, covariance)
        if func.__name__ == 'func1':
            name = 'sin (x*\pi)'
        elif func.__name__ == 'func2':
            name = 'cos (x*\pi)'
        elif func.__name__ == 'func3':
            name = 'e^x'
        ax = fig.add_subplot()
        ax.plot(np.linspace(-5, 5, 100), func(np.linspace(-5, 5, 100)), 'b--', label=f'${name}$')
        ax.fill_between(x.flat, mu[:, 0]-confidence*std, mu[:, 0]+confidence*std, color='red', alpha=0.15, label=f'$\mu_* \pm {confidence} std$')
        ax.plot(x, mu[:, 0], 'r-', lw=2, label='$\mu_*$')
        ax.plot(self.x, self.y, 'ko', linewidth=2, label='$(xtest, ytest)$')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y = f(x)$')
        ax.set_xlim([-5.1, 5.1])
        ax.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def plot_posterior2D(self, mu, s2, func, test_size, filename):
        y = x = np.linspace(-5, 5, test_size)
        x, y = np.meshgrid(x, y)
        mu = mu.reshape(test_size, test_size)
        
        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(x, y, mu, antialiased=False, rstride=2, cstride=2, cmap=cm.coolwarm, linewidth=0)
        ax.set_zlabel('z = f(x, y)')
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        ax.set_zlim(bottom=-5, top=5) 
        fig.colorbar(surf, shrink=0.5, aspect=18)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()