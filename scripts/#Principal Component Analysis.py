#Principal Component Analysis
#The goal of PCA is to reduce the number of dimensions while retaining as much variance (information) as possible.
#1. Its a Dimensionality reduction technique.
#2. PCA transforms high-dimensional data into a lower-dimensional form while retaining most of the original data's variance (or information).
#3. PCA identifies the directions (principal components) along which the variance of the data is maximized. 
# The first principal component captures the largest variance, the second captures the second-largest variance, and so on.
#4. The principal components are the eigenvectors of the data's covariance matrix, and the eigenvalues indicate how much variance 
# is explained by each eigenvector. The larger the eigenvalue, the more variance is captured by the corresponding eigenvector.


# Steps Involved in PCA

# Standardize the Data: PCA is sensitive to the scale of the data, so it's common to standardize each feature to have a mean of 0 and a variance of 1 before applying PCA.
# Compute the Covariance Matrix: The covariance matrix represents how the features of the data are correlated. For PCA to work effectively, it needs to understand how features vary together.
# Compute the Eigenvalues and Eigenvectors: The eigenvalues and eigenvectors of the covariance matrix give the directions (principal components) along which the data has the most variance. The eigenvalues determine how much variance is explained by each component.
# Sort Eigenvectors: Sort the eigenvectors in order of the magnitude of their corresponding eigenvalues (i.e., most variance first).
# Choose Top k Components: Select the top k eigenvectors (or principal components) that explain the most variance. This reduces the dimensionality of the data.
# Transform the Data: Finally, project the original data onto the new lower-dimensional space formed by the top k principal components.

# Relationship between Eigen Values and Explained Variance
#1. Eigenvalues represent the magnitude of the variance captured by each principal component.
#2. The "explained variance" for a given principal component tells you how much of the total variance(information) in the dataset is captured by that component.
#3. The explained variance of a component is directly proportional to its corresponding eigenvalue.
#4. The "number of principal components" refers to the number of dimensions (or new features) you select from the original data after performing PCA. 
# Each principal component (PC) is a linear combination of the original features, and it represents a new axis in the transformed feature space.