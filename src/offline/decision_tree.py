import numpy as np
from base.utils import gini, entropy, inf_gain

### TODO 1. Speed up algorithm
### TODO 2. Implement different pre-pruning strategies
### TODO 3. Improve structure of code
### TODO 4. Implement DecisionTreeRegressor

class Node(object):

    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, depth=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.depth = depth

class DecisionTreeClassifier(object):

    def __init__(self, max_depth=np.inf, min_samples_split=2, criterion='gini', splitter='best', max_features=None,
                 random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.depth = 0
        self.splitter = splitter
        self.max_features = max_features
        self.random_state = random_state

    def _split(self, X, y):

        m, d = X.shape

        np.random.seed(self.random_state)
        feat_idxs = np.random.permutation(d)

        max_features = {'log': np.log2, 'sqrt': np.sqrt}
        if self.max_features:
            d = int(np.floor(max_features[self.max_features](d)))
            feat_idxs = np.random.choice(feat_idxs, d, replace=False)


        max_ig = -np.inf

        if self.criterion == 'gini':
            criterion = gini
        else:
            criterion = entropy

        for feat_idx in feat_idxs:
            if self.splitter == 'best':
                sorted_idxs = np.argsort(X[:, feat_idx])
                y_sorted = y[sorted_idxs].squeeze()
                threshold_idxs = np.argwhere(y_sorted[1:] != y_sorted[:-1])
                thresholds = X[threshold_idxs, feat_idx]
            elif self.splitter == 'random':
                thresholds = np.random.choice(X[:, feat_idx], 3)

            for t in thresholds:
                left_mask = np.argwhere(X[:, feat_idx] < t)
                right_mask = np.argwhere(X[:, feat_idx] >= t)

                ig = inf_gain(X, y, left_mask, right_mask, criterion)

                if ig > max_ig:
                    max_ig = ig
                    best_split = (left_mask, right_mask)
                    best_feature_idx = feat_idx
                    best_threshold = t

        return best_split, best_feature_idx, best_threshold

    def _grow_subtree(self, X, y, node=None):

        m, d = X.shape

        # check whether there are different
        # classes in the current node
        unique_y, counts_y = np.unique(y, return_counts=True)
        if len(unique_y) == 1:
            node.prediction = unique_y[0]
            return node

        # pre-prunning
        if (node.depth == self.max_depth) or (m < self.min_samples_split):
            node.prediction = unique_y[np.argmax(counts_y)]
            return node

        # split the data
        best_split, best_feature_idx, best_threshold = self._split(X, y)
        best_left_mask, best_right_mask = best_split
        X_left, y_left = X[best_left_mask.squeeze(axis=1)], y[best_left_mask].reshape(-1, 1)
        X_right, y_right = X[best_right_mask.squeeze(axis=1)], y[best_right_mask].reshape(-1, 1)

        if (len(y_left) == 0) or (len(y_right) == 0):
            node.prediction = unique_y[np.argmax(counts_y)]
            return node

        # create a node and grow two subtrees
        node.threshold = best_threshold
        node.feature_idx = best_feature_idx
        node.left = self._grow_subtree(X_left, y_left, node=Node(depth=node.depth+1))
        node.right = self._grow_subtree(X_right, y_right, node=Node(depth=node.depth+1))
        return node

    def fit(self, X, y):

        self.root_node = Node(depth=0)
        self._grow_subtree(X, y, node=self.root_node)

        return self

    def _predict_sample(self, x, node):
        if (node.left is None) or (node.right is None):
            return node.prediction

        if x[node.feature_idx] < node.threshold:
            node = node.left
        else:
            node = node.right

        y_pred = self._predict_sample(x, node)
        return y_pred

    def predict(self, X):
        n = X.shape[0]
        y_pred = np.empty((n, 1), dtype=np.int16)

        for idx in range(n):
            y_pred[idx] = self._predict_sample(X[idx, :], self.root_node)

        return y_pred

