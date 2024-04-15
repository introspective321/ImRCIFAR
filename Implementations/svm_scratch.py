import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

class SVM(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel='linear', C=1.0, gamma=0.01):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel = kernel
        self.C = C
        self.gamma = gamma  # Add gamma as a parameter
        self.w = None
        self.b = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        if self.kernel == 'rbf':
            self.support_vectors = []
            self.sv_coefs = np.zeros(n_samples)

        # Shuffle once before the iterations
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y_[indices]

        for _ in range(self.n_iters):
            for idx in range(n_samples):
                x_i = X_shuffled[idx]
                y_i = y_shuffled[idx]
                if self.kernel == 'linear':
                    condition = y_i * (np.dot(x_i, self.w) - self.b) >= 1
                    if condition:
                        self.w -= self.lr * (2 * self.lambda_param * self.w)
                    else:
                        self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_i))
                        self.b -= self.lr * y_i
                elif self.kernel == 'rbf':
                    K = self.kernel_matrix(X, x_i)
                    prediction = np.sign(np.dot(self.sv_coefs, K))
                    error = y_i - prediction
                    self.sv_coefs[idx] += self.lr * error[0]  # Extracting the scalar value
                    if np.any(error != 0):
                        self.support_vectors.append(x_i)

    def predict(self, X):
        if self.kernel == 'linear':
            return np.sign(np.dot(X, self.w) - self.b)
        elif self.kernel == 'rbf':
            K = self.kernel_matrix(self.support_vectors, X)
            return np.sign(np.dot(self.sv_coefs, K.T))

    def kernel_matrix(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            if X1.ndim == 1:
                X1 = X1.reshape(-1, 1)
            if X2.ndim == 1:
                X2 = X2.reshape(-1, 1)
            sq_distances = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
            return np.exp(-sq_distances / (2 * self.C ** 2))

    def get_params(self, deep=True):
        return {"learning_rate": self.lr, "lambda_param": self.lambda_param, "n_iters": self.n_iters, "kernel": self.kernel, "C": self.C}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def tune_parameters(X_train, y_train):
    svm = SVM(kernel='rbf')
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]}
    grid_search = GridSearchCV(svm, param_grid, cv=min(2, len(np.unique(y_train))), scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_



# Load your dataset
# X_train, y_train, X_test, y_test = ...
# Reshape the data
X_train_hog_flat = X_train_pca.reshape(len(X_train_pca), -1)
X_test_hog_flat = X_test_pca.reshape(len(X_test_pca), -1)

# Tune parameters
best_params = tune_parameters(X_train_hog_flat[:], y_training[:])
print("Best parameters:", best_params)

# Train the SVM with the best hyperparameters
svm = SVM(**best_params)
svm.fit(X_train_hog_flat[:], y_training[:])

# Evaluate the model
y_pred = svm.predict(X_test_hog_flat[:])
acc = accuracy_score(y_testing[:], y_pred)
print("Accuracy:", acc)
