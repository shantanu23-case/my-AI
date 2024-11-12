import numpy as np
from scipy import linalg
A=np.array([[1,2,3]
            ,[7,3,2]
            ,[9,0,8]])
B=np.array([[4]
            ,[2]
            ,[8]])
augmented_matrix = np.hstack((A, B))
print(augmented_matrix)
#Finding the rank of Agumented Matrix
r=np.linalg.matrix_rank(augmented_matrix)
print(r)