import numpy as np
from scipy import linalg
A=np.array([[1,2],[5,6]])
#Rank of Matrix
print("Rank:",np.linalg.matrix_rank(A))
# Trace of Matrix
print("Trace:",np.trace(A))
# Determinat of Matrix
print("Determinant of A:", np.linalg.det(A))
#Inverse of a matrix
print("Inverse of a matrix:",np.linalg.inv(A))
#Transpose of a Matrix
C=print (A.T)
print (C)

B=np.array([[5,6],[7,8]])
print (np.multiply(A,B)) #element wise multiplication
print (np.dot(A,B)) #Matrix Multiplication