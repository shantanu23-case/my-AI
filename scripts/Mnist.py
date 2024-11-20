import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import sympy as sp
import scipy as scp
from sklearn.decomposition import PCA
#How to upload minist data sets
m_sample=fetch_openml('mnist_784', version=1)
#To use 2000 samples
x=m_sample.data[:2000]
#Scaling the image to 255
x=x/255
cov_matrix=np.cov(x)
print (x)
#Eigen decompostion decompose a square matrix into its eigenvalues and eigenvectors.
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(eigenvalues)
print(eigenvectors)
#Sorting the Eigen values in descending order
eigenvalues = eigenvalues[::-1] 
#Var Percentage
var_per= eigenvalues / np.sum(eigenvalues)
print(var_per)
#Cumulative Variance
cum_var = np.cumsum(var_per)
print(cum_var)
plt.plot(cum_var, marker='o')
plt.title('Cumulative Explained Variance vs Number of Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()