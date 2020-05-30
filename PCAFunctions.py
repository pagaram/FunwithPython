import numpy as np

def centerData(X):
    mean_vec = np.mean(X, axis=0)
    XC = X - mean_vec

    return XC

def DatCovariance(X):
    XC = centerData(X)
    XCT = np.transpose(XC)
    cov_mat = np.dot(XCT, XC)
    cov_mat = cov_mat/(XC.shape[0] - 1)

    return cov_mat

