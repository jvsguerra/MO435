import numpy as np
from numpy import sin, cos, exp, pi
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def func1(x):
    return sin(x*pi)

def func2(x):
    return cos(x*pi)

def func3(x):
    return exp(x)

def func2d(x, y):
    return sin(x*pi) + cos(y*pi)

def plot2D(func, output, lower_bound=-5, upper_bound=5):
    # Get 100 linearly spaced numbers
    x = np.linspace(lower_bound, upper_bound, 100)
    # Compute function
    y = func(x)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x, y)
    ax.set_ylabel('y = f(x)')
    ax.set_xlabel('x')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

def plot3D(func, output, lower_bound=-5, upper_bound=5):
    # Get 100 linearly spaced numbers
    x = np.linspace(lower_bound, upper_bound, 100)
    y = np.linspace(lower_bound, upper_bound, 100)
    x, y = np.meshgrid(x, y)
    # Compute function
    z = func(x, y)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(x, y, z, antialiased=False, rstride=2, cstride=2, cmap=cm.coolwarm, linewidth=0)
    ax.set_zlabel('z = f(x, y)')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_zlim(bottom=lower_bound, top=upper_bound) 
    fig.colorbar(surf, shrink=0.5, aspect=18)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

def plot_functions():
    # Plot functions
    plot2D(func1, 'output/functions/sin.png')
    plot2D(func2, 'output/functions/cos.png')
    plot2D(func3, 'output/functions/exp.png')
    plot3D(func2d, 'output/functions/sinx+cosy.png')

def sampling1D(n, d, func, lower_bound=-5, upper_bound=5):
    x = np.zeros((n, d))
    y = np.zeros((n))
    for i in range(d):
        np.random.seed(i)
        x[:,i] = np.random.uniform(lower_bound, upper_bound, n)
    func = np.vectorize(func)
    y = func(x)
    return x, y

def sampling2D(n, d, func, lower_bound=-5, upper_bound=5):
    x = np.zeros((n, d))
    y = np.zeros((n))
    for i in range(d):
        x[:,i] = np.random.uniform(lower_bound, upper_bound, n)
    func = np.vectorize(func)
    y = func(x[:, 0], x[:, 1])
    return x, y

def samples_plot2D(n, d, func, output, lower_bound=-5, upper_bound=5):
    x, y = sampling1D(n, d, func)  
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x, y)
    ax.set_ylabel('y = f(x)')
    ax.set_xlabel('x')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

def samples_plot3D(n, d, func, output, lower_bound=-5, upper_bound=5):
    x, z = sampling2D(n, d, func)
    y = x[:, 1]
    x = x[:, 0]
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    scatter = ax.scatter(x, y, z, c=z, cmap='coolwarm', vmin=-2, vmax=2)
    ax.set_zlabel('z = f(x, y)')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_xlim(lower_bound, upper_bound) 
    ax.set_ylim(lower_bound, upper_bound) 
    ax.set_zlim(bottom=lower_bound, top=upper_bound)
    fig.colorbar(scatter, shrink=0.5)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

def plot_samples_scatter():
    # Plot func1
    samples_plot2D(5, 1, func1, 'output/samples/sin5.png')
    samples_plot2D(10, 1, func1, 'output/samples/sin10.png')
    samples_plot2D(50, 1, func1, 'output/samples/sin50.png')
    # Plot func2
    samples_plot2D(5, 1, func2, 'output/samples/cos5.png')
    samples_plot2D(10, 1, func2, 'output/samples/cos10.png')
    samples_plot2D(50, 1, func2, 'output/samples/cos50.png')
    # Plot func3
    samples_plot2D(5, 1, func3, 'output/samples/exp5.png')
    samples_plot2D(10, 1, func3, 'output/samples/exp10.png')
    samples_plot2D(50, 1, func3, 'output/samples/exp50.png')
    # Plot func2d
    samples_plot3D(5, 2, func2d, 'output/samples/sinx+cosy5.png')
    samples_plot3D(10, 2, func2d, 'output/samples/sinx+cosy10.png')
    samples_plot3D(50, 2, func2d, 'output/samples/sinx+cosy50.png')
    plt.close()