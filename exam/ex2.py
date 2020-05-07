import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def cov(s):
    return np.array([[s, 0], [0, s]])

def KL(cov1, cov2, k=2):
    term1 = np.log(np.linalg.det(cov2)/np.linalg.det(cov1))
    term2 = np.trace(np.linalg.inv(cov2).dot(cov1))
    term3 = k
    return 0.5 * (term1 + term2 + term3)

x, y = np.mgrid[-4:4:.01, -4:4:.01]
position = np.empty(x.shape + (2,))
position[:, :, 0] = x
position[:, :, 1] = y

# different values for the covariance matrix
real = np.array([[1, -1], [-1, 4]])
sx = real[0,0]
sy = real[1,1]

ro2 = (real[0,1] ** 2)/(sx * sy)
print(ro2)
s = 2 * (sx*sy - ro2*sx*sy) / (sx + sy)

for i in np.arange(0.1, 5, 0.1):
    if 0 <= i-s <= 0.01:
        print('[======> ', end='') 
    print(f"{i:.2f}: {KL(cov(i), real):.5f}")
print(f"KL(q||p, {s:.2f}): {KL(cov(s), real):.2f}")
covariances = [ real, np.array([[s, 0], [0, s]]) ]
titles = ['real', 'rKL']

plt.figure(figsize = (15, 6))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    z = multivariate_normal([0, 0], covariances[i]).pdf(position)
    plt.contour(x, y, z)
    plt.title('{}, {}'.format(titles[i], covariances[i]))
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
plt.show()

# Forward KL-divergence
sx = real[0,0]
sy = real[1,1]
s = (sx + sy) / 2

for i in np.arange(0.1, 5, 0.1):
    if 0 < i-s < 0.01:
        print('[======> ', end='') 
    print(f"{i:.2f}: {KL(real, cov(i)):.5f}")
print(f"KL(p||q, {s:.2f}): {KL(real, cov(s)):.2f}")
covariances = [ real, np.array([[s, 0], [0, s]]) ]
titles = ['real', 'fKL']

plt.figure(figsize = (15, 6))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    z = multivariate_normal([0, 0], covariances[i]).pdf(position)
    plt.contour(x, y, z)
    plt.title('{}, {}'.format(titles[i], covariances[i]))
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])

plt.show()