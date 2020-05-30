import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

data = pd.read_csv('Data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

#using scipy to do least squares
slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

#creating least squares matrices
B = np.zeros([len(X), 1])
A = np.zeros([len(X), 2])

for i in range(len(X)):
    B[i, 0] = Y[i]
    A[i, 0] = X[i]
    A[i, 1] = 1

dat = np.dot(np.linalg.pinv(A), B)

print(dat[0, 0])
print(dat[1, 0])

print(slope)
print(intercept)

YP = np.zeros([len(X), 1])
for i in range((len(X))):
    # YP[i, 0] = slope*X[i] + intercept
    YP[i, 0] = dat[0, 0]*X[i] + dat[1, 0]

plt.figure()
plt.plot(X, Y, 'bo')
plt.plot(X, YP[:, 0], 'r-')
plt.grid()
plt.show()




