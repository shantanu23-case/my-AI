import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import log10
import random
from PIL import Image

# Task-1: Eigen Decomposition and Variance Explained
# Load the MNIST dataset and extract the first 2000 test samples
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Take the first 2000 samples and flatten the images
x_test = x_test[:2000].reshape(2000, 28*28).astype('float32')

# Scale the images to [0, 1] by dividing by 255
x_test_scaled = x_test / 255.0

# Compute the covariance matrix
cov_matrix = np.cov(x_test_scaled.T)

# Eigen decomposition
eigvals, eigvecs = np.linalg.eigh(cov_matrix)

# Sort the eigenvalues in descending order
sorted_eigvals = np.flip(eigvals)
sorted_eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]

# Calculate the percentage of variance explained
explained_variance = sorted_eigvals / np.sum(sorted_eigvals)
cumulative_variance = np.cumsum(explained_variance)

# Plot the cumulative variance explained
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100)
plt.title('Cumulative Variance Explained by Eigenvalues')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Percentage of Variance Explained')
plt.grid(True)
plt.show()

# Task-2: PCA for different dimensions (p ∈ {50, 250, 500})
p_values = [50, 250, 500]

for p in p_values:
    # Apply PCA using Eigen decomposition for reduced dimensions
    pca_p = PCA(n_components=p)
    x_test_reduced_pca = pca_p.fit_transform(x_test_scaled)
    
    print(f"Shape of reduced data with {p} components: {x_test_reduced_pca.shape}")

# Task-3: Reconstruct Data Using Reduced Components
def reconstruct_data_pca(pca_p, reduced_data_pca):
    return pca_p.inverse_transform(reduced_data_pca)

# Pick a random image to test reconstruction for each p
random_index_pca = random.randint(0, 1999)
original_image_pca = x_test[random_index_pca].reshape(28, 28)

for p in p_values:
    # Apply PCA to reduce dimensionality
    pca_p = PCA(n_components=p)
    reduced_data_pca = pca_p.fit_transform(x_test_scaled)
    
    # Reconstruct the data
    reconstructed_image_pca = reconstruct_data_pca(pca_p, reduced_data_pca[random_index_pca]).reshape(28, 28)
    
    # Display original and reconstructed image
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_pca, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image_pca, cmap='gray')
    plt.title(f'Reconstructed Image ({p} components)')
    plt.show()

# Task-4: Compute PSNR for Random Images
def psnr(original_pca, reconstructed_pca):
    mse = mean_squared_error(original_pca.flatten(), reconstructed_pca.flatten())
    if mse == 0:
        return 100  # No error
    max_pixel = 255.0
    return 20 * log10(max_pixel / np.sqrt(mse))

# Select 5 random images for comparison
for _ in range(5):
    random_index_pca = random.randint(0, 1999)
    original_image_pca = x_test[random_index_pca].reshape(28, 28)
    
    # Reconstruct the image using PCA with 50 components (for example)
    pca_p = PCA(n_components=50)
    reduced_data_pca = pca_p.fit_transform(x_test_scaled)
    reconstructed_image_pca = reconstruct_data_pca(pca_p, reduced_data_pca[random_index_pca]).reshape(28, 28)
    
    # Calculate PSNR
    psnr_value_pca = psnr(original_image_pca, reconstructed_image_pca)
    
    # Display original vs reconstructed and PSNR
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_pca, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image_pca, cmap='gray')
    plt.title(f'Reconstructed Image (PSNR = {psnr_value_pca:.2f} dB)')
    plt.show()
