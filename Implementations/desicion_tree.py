import numpy as np
from collections import Counter
import random

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    en = -np.sum([p*np.log2(p) for p in ps if p>0])
    return en

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=100, n_feats=None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if self.n_feats is None else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_sample_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        best_feat, best_thresh = self._best_criteria(X, y, self.n_feats)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _best_criteria(self, X, y, n_feats):
        """
        Find the best split for a node.
        """
        # Ensure that n_feats is an integer
        n_feats = int(n_feats)

        # Randomly select a subset of the features without replacement
        feat_idxs = random.sample(range(X.shape[1]), n_feats)

        best_feat, best_thresh = None, None
        best_gain = -1
        for feature in feat_idxs:
            X_column = X[:, feature]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feature
                    best_thresh = threshold

        return best_feat, best_thresh

    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = entropy(y)

        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def prune(self, X_val, y_val):

        self.accuracy_with_child = np.sum(self.predict(X_val) == y_val) / len(y_val)
        self._prune(self.root, X_val, y_val)

    def _prune(self, node, X_val, y_val):
        """
        Recursively prune the tree.
        """
        if node is None or node.is_leaf_node():
            return

        if node.left:
            if node.left.is_leaf_node():
                self._try_prune(node, 'left', X_val, y_val)
            else:
                self._prune(node.left, X_val, y_val)

        if node.right:
            if node.right.is_leaf_node():
                self._try_prune(node, 'right', X_val, y_val)
            else:
                self._prune(node.right, X_val, y_val)

    def _try_prune(self, node, child_side, X_val, y_val):
        """
        Try to prune a child of a node and check if it improves the accuracy.
        """
        child_node = getattr(node, child_side)
        setattr(node, child_side, None)

        y_pred = self.predict(X_val)
        accuracy_without_child = np.sum(y_val == y_pred) / len(y_val)

        setattr(node, child_side, child_node)

        if accuracy_without_child >= self.accuracy_with_child:
            setattr(node, child_side, None)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node is None:
            return None

        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)
