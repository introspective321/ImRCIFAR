import numpy as np
from collections import Counter
import random
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn import datasets

class RandomForest:
    def __init__(self, n_trees=100, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []

        # Create a pool of processes
        with mp.Pool(processes=mp.cpu_count()) as pool:

            # Define the arguments for each process
            args = [(X, y) for _ in range(self.n_trees)]

            # Use starmap to apply the function to each set of arguments
            self.trees = pool.starmap(self._train_tree, args)

    def _train_tree(self, X, y):
        tree = DecisionTree(min_sample_split=self.min_sample_split, max_depth=self.max_depth, n_feats=self.n_feats)
        X_sample, y_sample = bootstrap_sample(X, y)
        tree.fit(X_sample, y_sample)
        return tree

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

    def prune(self, X_val, y_val):
        for tree in self.trees:
            tree.prune(X_val, y_val)

def bootstrap_sample(X,y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples , size=n_samples,replace = True)
    return X[idxs],y[idxs]

def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]  # Get the label, not the count
    return most_common

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
