import numpy as np
from scipy import linalg
#Solving Linear Equations
A=np.array([[1,2,3],[5,6,7]])
B=np.array([4,5])
m_rows=len(A)
n_cols=len(A[0])
print(m_rows)
print(n_cols)
#Finding the Determinant of Matrix A
if m_rows!=n_cols:
    print("It's a Non-Square Matrix so We cannot Find the Determinant")
    exit ()
else:
    print("Lets do the Matrix operations Further !!")
det_A=np.linalg.det(A)
print (det_A)
if m_rows==n_cols:
    print ("it could have Solution")
    if (det_A!=0):
        x=np.linalg.inv(A)@B
        print(x)
    else:
        print("Infinite solution")
elif m_rows>n_cols:
    print ("It Still could have a Solution")
    x=np.linalg.inv((A.T)@A)@(A.T)@B
    print(x)
else:
    x=(A.T)@np.linalg.inv(A@(A.T))@B
    print ("------------")
    print(x)