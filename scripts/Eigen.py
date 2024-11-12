import numpy as np
from scipy import linalg
A=np.array([[1,2],[5,6]])
m_rows=len(A)
n_cols=len(A[0])
print(m_rows)
print(n_cols)
#Calucaltion of Eigen value
print("Eigen value Calculation A-VI=0")
#Checking if the matrix is Square
if m_rows!=n_cols:
    print("It's a Non-Square Matrix so We cannot Find the Determinant")
    exit ()
else:
    print("Lets do the Matrix operations Further !!")
E,V=np.linalg.eig(A)
print("Eigen value",E)
print("Eigen Vector",V)