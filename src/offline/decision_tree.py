import numpy as np


class Node(object):

    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, labels=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.labels = labels

class DecisionTreeClassifier(object):

    def __init__(self, max_depth=np.inf, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.depth = 0
        self.nodes = []

    def _gini(self, y):
        _, p = np.unique(y, return_counts=True)
        p = p / p.sum()
        value = -(p * np.log2(p)).sum()
        return value

    def _entropy(self, y):
        _, p = np.unique(y, return_counts=True)
        p = p / p.sum()

        value = 1 - (p ** 2).sum()
        return value

    def _inf_gain(self, X, y, left_mask, right_mask):
        if self.criterion == 'gini':
            criterion = self._gini
        else:
            criterion = self._entropy

        X_left, y_left = X[left_mask, :], y[left_mask]
        X_right, y_right = X[right_mask, :], y[right_mask]

        ig = criterion(y) - len(y_left)/len(y) * criterion(y_left) - len(y_right)/len(y) * criterion(y_right)

        return ig


    def fit(self, X, y):

        m, d = X.shape

        # check whether there are different
        # classes in the current node
        unique_y, counts_y = np.unique(y, return_counts=True)
        if len(unique_y) == 1:
            self.current_node = Node()
            self.current_node.prediction = unique_y[0]
            self.nodes.append(self.current_node)
            return self.current_node

        # find the best split
        max_ig = 0

        for feat_idx in range(d):
            for i in range(m):
                threshold = X[i, feat_idx]
                left_mask = np.argwhere(X[:, feat_idx] < threshold)
                right_mask = np.argwhere(X[:, feat_idx] >= threshold)
                ig = self._inf_gain(X, y, left_mask, right_mask)

                if ig > max_ig:
                    max_ig = ig
                    best_split = (left_mask, right_mask)
                    best_feature_idx = feat_idx
                    best_threshold = threshold

        # split the data
        best_left_mask, best_right_mask = best_split
        X_left, y_left = X[best_left_mask.squeeze(axis=1)], y[best_left_mask].reshape(-1, 1)
        X_right, y_right = X[best_right_mask.squeeze(axis=1)], y[best_right_mask].reshape(-1, 1)

        if (len(y_left) == 0) or (len(y_right) == 0) or (self.depth == self.max_depth):
            self.current_node = Node()
            self.current_node.prediction = unique_y[np.argmax(counts_y)]
            self.nodes.append(self.current_node)
            return self.current_node

        # create a node and grow two subtrees
        self.current_node = Node()
        self.current_node.threshold = best_threshold
        self.current_node.feature_idx = best_feature_idx
        self.nodes.append(self.current_node)
        self.depth += 1
        self.current_node.left = self.fit(X_left, y_left)
        self.current_node.right = self.fit(X_right, y_right)
        return self.current_node

    def _predict_sample(self, x, node):
        if (node.left is None) or (node.right is None):
            return node.prediction

        if x[:, node.feature_idx] < node.threshold:
            node = node.left
        else:
            node = node.right

        self._predict_sample(x, node)

    def predict(self, X):
        n = X.shape[0]
        y_pred = np.empty((n, 1), dtype=np.int16)

        for idx in range(n):
            y_pred[idx] = self._predict_sample(X[idx, :], self.nodes[1])

        return y_pred

