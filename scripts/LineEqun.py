import numpy as np
from scipy import linalg
#Solving Linear Equations
A=np.array([[1,2],[5,6]])
B=np.array([4,5])
m_rows=len(A)
n_cols=len(A[0])
print(m_rows)
print(n_cols)

if m_rows!=n_cols:
    print("It's a Non-Square Matrix so We cannot Find the Determinant")
    exit ()
else:
    print("Lets do the Matrix operations Further !!")
#Finding the Determinant of Matrix A
det_A=np.linalg.det(A)
print (det_A)
#Finding the Solution of Linear Equations
#If a system of equations has one or more solutions, it is called a consistent system of equations. 
#>Consistent>DetA Not equal to 0.
#>
#If the system doesn't have any solution, it is an inconsistent system of equations.
if m_rows==n_cols:
    print ("it could have Solution")
    if (det_A!=0):
        x=np.linalg.inv(A)@B
        print(x)
        print("Syetem is Consistent beavuse determinat is Not 0")
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