# Train perceptron
def initialize_weights(num_features):
    return np.random.uniform(-1, 1, size=num_features + 1)

def perceptron_train(X_train, y_train, iterations, learning_rate, num_classes):
    num_features = X_train.shape[1]
    weights = []

    for class_label in range(num_classes):
        class_weights = initialize_weights(num_features)
        for _ in range(iterations):
            # Shuffle data for each epoch
            shuffled_indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[shuffled_indices]
            y_train_shuffled = y_train[shuffled_indices]
            for features, label in zip(X_train_shuffled, y_train_shuffled):
                features_with_bias = np.insert(features, 0, 1)
                target = 1 if label == class_label else 0
                prediction = 1 if np.dot(class_weights, features_with_bias) >= 0 else 0
                error = target - prediction
                class_weights += learning_rate * error * features_with_bias
        weights.append(class_weights)

    return weights

# Training perceptron
weights = perceptron_train(X_train_pca, y_training, iterations=100, learning_rate=0.01, num_classes=10)


def predict(X_test, weights):
    predictions = []
    for sample in X_test:
        scores = [np.dot(sample, class_weights[1:]) + class_weights[0] for class_weights in weights]
        predicted_label = np.argmax(scores)
        predictions.append(predicted_label)
    return np.array(predictions)

predicted_labels = predict(X_test_pca, weights)

# Compute accuracy
accuracy = np.mean(predicted_labels == y_testing)
print("Accuracy:", accuracy)
