import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

X = np.array(([1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1])).T

Y = np.array(([-1, -1], [-1, -2], [-2, -1], [1, 1], [1, 2], [2, 1]))

y1 = Y[:, 0]
y2 = Y[:, 1]

w11 = sm.OLS(y1, X[0, :])
results = w11.fit()
print('w11: ', results.params)
w12 = sm.OLS(y1, X[1, :])
results = w12.fit()
print('w12: ', results.params)
w21 = sm.OLS(y2, X[0, :])
results = w21.fit()
print('w21: ', results.params)
w22 = sm.OLS(y2, X[1, :])
results = w22.fit()
print('w22: ', results.params)

# simulataneously
w = sm.OLS(Y, X.T)
results = w.fit()
print('Parameters: ', results.params)

# Pred
Ypred = X.T.dot(results.params)
# print(Y.shape)
# print(Ypred.shape)
# print(X.shape)

# Plot each axis (x1, x2, y1, y2)
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(X[0,:], Y[:, 1], 'o', label='Data')
ax.plot(X[0,:], Ypred[:, 1], 'b-', label='Pred')
ax.legend(loc="best")
plt.show()