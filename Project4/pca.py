import numpy as np
import matplotlib.pyplot as plt

X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
def compute_Z(X, centering=True, scaling=False):
    
  if centering is True:
    # Calculate mean
    x_i = 0
    x_j = 0
    for i in range(len(X)):
      x_i += X[i][0]
      x_j += X[i][1]
    x_i = x_i / len(X)
    x_j = x_j / len(X)

     # Subtract mean from values
    if (x_i and x_j != 0):
      for i in range(len(X)):
        X[i][0] = X[i][0] - x_i
        X[i][1] = X[i][1] - x_j
    
  if scaling is True:
    std = np.std(X)
    for i in range(len(X)):
      for j in range(X.shape[1]):
        X[i][j] = X[i][j] / std
    
  return X


def compute_covariance_matrix(Z):
  A = np.dot(X.T, X)
  return A

def find_pcs(COV):
  x, y = np.linalg.eig(COV)
  
  # Sort by largest to smallest PCS


# def project_data(Z, PCS, L, k, var):

Z = compute_Z(X)
COV = compute_covariance_matrix(Z)
L, PCS = find_pcs(COV)
