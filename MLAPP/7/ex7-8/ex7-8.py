import numpy as np
import statsmodels.api as sm

# Data
X = np.array([94,96,94,95,104,106,108,113,115,121,131])
Y = np.array([0.47, 0.75, 0.83, 0.98, 1.18, 1.29, 1.40, 1.60, 1.75, 1.90, 2.23])

# Add intercept to X
X = sm.add_constant(X)

# Model
results = sm.OLS(Y, X).fit()

# print(results.params)
# w0 and w1
w0 = results.params[0]
w1 = results.params[1]

# Ypred
Ypred = w0 + w1*X[:,1]

# Calculate Sigma^2
sigma = np.sum((Y - Ypred) ** 2)/(X.shape[0]-2)
print(f"Estimate of Sigma^2: {sigma} ~ {sigma:.2f}")

# c and d
# https://github.com/ppham27/MLaPP-solutions/blob/master/chap07/8.ipynb