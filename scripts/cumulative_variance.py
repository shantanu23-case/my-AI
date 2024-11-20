import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sample data: 5 samples, 3 features
X = np.array([[2.5, 2.4, 3.5],
              [0.5, 0.7, 1.1],
              [2.2, 2.9, 2.6],
              [1.9, 2.2, 2.3],
              [3.1, 3.0, 3.4]])

# Fit PCA
pca = PCA()
pca.fit(X)

# Eigenvalues (explained variance)
explained_variance = pca.explained_variance_

# Proportion of total variance explained by each component
explained_variance_ratio = pca.explained_variance_ratio_

# Cumulative variance
cumulative_variance = np.cumsum(explained_variance_ratio)

# Print results
print("Explained Variance by each component:", explained_variance)
print("Cumulative Variance:", cumulative_variance)

# Plot the cumulative variance to help visualize the number of components needed
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained by Principal Components')
plt.grid(True)
plt.show()
