
def computeCentroid(features):
    """
    Compute centroid of a list of features.

    Parameters:
    features (numpy array): Array of features.

    Returns:
    numpy array: Centroid of the features.
    """
    num_features = len(features)
    if num_features == 0:
        return None

    # Compute mean of the features
    centroid = np.mean(features, axis=0)

    return centroid

# Convert the image_np array to a list of 3-dimensional features
features_list = X_training

# Compute the centroid of the features
centroid = computeCentroid(features_list)

def mykmeans(X, k, max_iters=100):
    # Step 1: Initialize cluster centers randomly
    np.random.seed(42)  # For reproducibility
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):
        # Step 2: Assign each data point to the nearest cluster center
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # Step 3: Update cluster centers
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids

# Define the function to compress the image using centroids
def compress_image_with_centroids(image, centroids):
    distances = np.sqrt(((image - centroids[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)
    compressed_image = centroids[labels]
    return compressed_image

 # Display compressed images
    plt.figure(figsize=(16, 6))
    for i in range(10):  # Display first 10 compressed images
        plt.subplot(2, 5, i + 1)
        plt.imshow(compressed_images[i].reshape(X_training_gray[0].shape).astype(np.uint8), cmap='gray')
        plt.title('Compressed Image {}'.format(i + 1))
        plt.axis('off')
