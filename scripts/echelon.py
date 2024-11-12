import numpy as np
from scipy import linalg
import sympy
#Matrix Reduced row echelon form
A=np.array([[2,-2,4,-2],[2,1,10,7],[-4,4,-8,4],[4,-1,14,6]])
# print(A)
Echleon=sympy.Matrix(A).rref(iszerofunc=lambda x:abs(x)<1e-9)
#>.rref() is used to reduce the matrix in echleon form
#it aslo print the tuple containing the indices of Pivot coloumns means Index of leading 1's is printed 
#so here for 1st coloumn index would be 0,second coloum index wpuld be 1,As we dont have 1 in third coloumns
#in 4th coloumn 1 is present at index 3. Index calcualtion is done horizointally.
#Reduced Row Echelon Form (RREF)
print(Echleon)