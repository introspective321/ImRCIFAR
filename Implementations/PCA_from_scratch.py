#Standardizing
X_training_mean = np.mean(X_training, axis=0)  # Compute mean along each feature
X_training_std = np.std(X_training, axis=0)    # Compute standard deviation along each feature
X_training_scaled = (X_training - X_training_mean) / X_training_std

# Standardize the testing data using mean and std computed from training data
X_testing_scaled = (X_testing - X_training_mean) / X_training_std

# After standardization
print('X_training_scaled shape is {}'.format(X_training_scaled.shape))
print('X_testing_scaled shape is {}'.format(X_testing_scaled.shape))

print('X_training_scaled after standardization:')
print(X_training_scaled)

# Compute the covariance matrix
cov_matrix = np.cov(X_training_scaled, rowvar=False)

# Print the covariance matrix
print("Covariance matrix:")
print(cov_matrix)

# Find the eigenvectors and eigenvalues of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Adjusting the eigenvectors (loadings) that are largest in absolute value to be positive
max_abs_idx = np.argmax(np.abs(eigenvectors), axis=0)
signs = np.sign(eigenvectors[max_abs_idx, range(eigenvectors.shape[0])])
eigenvectors = eigenvectors * signs[np.newaxis, :]
eigenvectors = eigenvectors.T

# Print the eigenvalues and eigenvectors
print("Eigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[i, :]) for i in range(len(eigenvalues))]

# Sort the tuples from the highest to the lowest based on eigenvalues magnitude
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# For further usage
eig_vals_sorted = np.array([x[0] for x in eig_pairs])
eig_vecs_sorted = np.array([x[1] for x in eig_pairs])

print(eig_pairs)

# Calculate explained variance and cumulative explained variance
eig_vals_total = np.sum(eig_vals_sorted)
explained_variance = [(i / eig_vals_total) * 100 for i in eig_vals_sorted]
explained_variance = np.round(explained_variance, 2)
cum_explained_variance = np.cumsum(explained_variance)

print('Explained variance: {}'.format(explained_variance))
print('Cumulative explained variance: {}'.format(cum_explained_variance))


# Plotting cumulative explained variance
num_components = len(explained_variance)
plt.plot(np.arange(1, num_components + 1), cum_explained_variance, '-o')

# Set x-axis tick labels explicitly
custom_ticks = [0, 50, 100, 150, 200, 250, 300]
plt.xticks(custom_ticks)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()


# Visualizing Over two components
k = 130
W = eig_vecs_sorted[:k, :]  # Projection matrix

X_proj = X_training_scaled.dot(W.T)

print(X_proj.shape)

plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y_training)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('2 components, captures {} of total variation'.format(cum_explained_variance[1]))
plt.show()



class MyPCA:
    
    def __init__(self, n_components):
        self.n_components = n_components   
        
    def fit(self, X):
        # Standardize data 
        X = X.copy()
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        X_std = (X - self.mean) / self.scale
        
        # Eigendecomposition of covariance matrix       
        cov_mat = np.cov(X_std.T)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat) 
        
        # Adjusting the eigenvectors that are largest in absolute value to be positive    
        max_abs_idx = np.argmax(np.abs(eig_vecs), axis=0)
        signs = np.sign(eig_vecs[max_abs_idx, range(eig_vecs.shape[0])])
        eig_vecs = eig_vecs * signs[np.newaxis, :]
        eig_vecs = eig_vecs.T
       
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[i, :]) for i in range(len(eig_vals))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        eig_vals_sorted = np.array([x[0] for x in eig_pairs])
        eig_vecs_sorted = np.array([x[1] for x in eig_pairs])
        
        self.components = eig_vecs_sorted[:self.n_components, :]
        
        # Explained variance ratio
        self.explained_variance_ratio = [i / np.sum(eig_vals) for i in eig_vals_sorted[:self.n_components]]
        
        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)

        return self

    def transform(self, X):
        X = X.copy()
        X_std = (X - self.mean) / self.scale
        X_proj = X_std.dot(self.components.T)
        
        return X_proj



my_pca = MyPCA(n_components=130).fit(X_training)

#print('Components:\n', my_pca.components)
#print('Explained variance ratio:\n', my_pca.explained_variance_ratio)
#print('Cumulative explained variance:\n', my_pca.cum_explained_variance)

X_train_pca = my_pca.transform(X_training)
X_test_pca = my_pca.transform(X_testing)
print('Transformed train data shape:', X_train_pca.shape)
print('Transformed test data shape:', X_test_pca.shape)
