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

##Task2
pca_50 = PCA(n_components=50)
X_pca_50 = pca_50.fit_transform(X)

pca_250 = PCA(n_components=250)
X_pca_250 = pca_250.fit_transform(X)

pca_500 = PCA(n_components=500)
X_pca_500 = pca_500.fit_transform(X)

##Task3
# Reconstruct the images from the reduced data
# We reconstruct the images using PCA.inverse_transform.
X_reconstructed_50 = pca_50.inverse_transform(X_pca_50)
X_reconstructed_250 = pca_250.inverse_transform(X_pca_250)
X_reconstructed_500 = pca_500.inverse_transform(X_pca_500)

##Task4
import random

# Select 5 random indices
random_indices = random.sample(range(2000), 5)

# Calculate PSNR for each random image
psnrs = []
for idx in random_indices:
    original_image = X[idx].reshape(28, 28) * 255  # Convert back to original range
    reconstructed_image_50 = X_reconstructed_50[idx].reshape(28, 28) * 255
    reconstructed_image_250 = X_reconstructed_250[idx].reshape(28, 28) * 255
    reconstructed_image_500 = X_reconstructed_500[idx].reshape(28, 28) * 255

    psnr_50 = psnr(original_image, reconstructed_image_50)
    psnr_250 = psnr(original_image, reconstructed_image_250)
    psnr_500 = psnr(original_image, reconstructed_image_500)

    psnrs.append((psnr_50, psnr_250, psnr_500))

# Output the PSNR values
for i, (psnr_50, psnr_250, psnr_500) in enumerate(psnrs):
    print(f"Image {i+1} - PSNR (50 components): {psnr_50:.2f} dB, PSNR (250 components): {psnr_250:.2f} dB, PSNR (500 components): {psnr_500:.2f} dB")