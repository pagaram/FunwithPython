import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PCAFunctions as PFunc

datF = pd.read_csv('DataPoints.csv', header=None)
data = datF.to_numpy()
dataM = np.mean(data, axis=0)
dataC = PFunc.centerData(data)
cov_mat = PFunc.DatCovariance(data)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

#finding significant eigvenvector in 2D matrix
feature_Mat = np.zeros((2, 1))
if eig_vals[0] > eig_vals[1]:
    feature_Mat[0, 0] = eig_vecs[0, 0]
    feature_Mat[1, 0] = eig_vecs[1, 0]

else:
    feature_Mat[0, 0] = eig_vecs[0, 1]
    feature_Mat[1, 0] = eig_vecs[1, 1]

#setting up eigvenvectors to plot
xv = np.linspace(-2, 2, 5)
y1 = []
y2 = []
yf = []
for i in range(len(xv)):
    valA = xv[i] * (eig_vecs[1, 0]/eig_vecs[0, 0])
    valB = xv[i] * (eig_vecs[1, 1]/eig_vecs[0, 1])
    y1.append(valA)
    y2.append(valB)

print(cov_mat)
print()
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
print('\nFeature \n%s' %feature_Mat)

TransformedData = np.dot(np.transpose(feature_Mat), np.transpose(dataC))
RecovData = np.transpose(np.dot(feature_Mat, TransformedData)) + dataM

plt.figure()
plt.subplot(221)
plt.plot(data[:, 0], data[:, 1], '*r')
plt.grid()
plt.title('Original Data')

plt.subplot(222)
plt.plot(dataC[:, 0], dataC[:, 1], '*b')
plt.plot(xv, y1, '--k')
plt.plot(xv, y2, '--k')
plt.grid()
plt.title('Centered Data')

plt.subplot(223)
plt.plot(RecovData[:, 0], RecovData[:, 1], '*b')
plt.grid()
plt.title('Recovered Data')
plt.show()
