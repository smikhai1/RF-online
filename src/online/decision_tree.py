import numpy as np
from base.utils import gini_online, entropy_online, inf_gain_online

class Node(object):

    def __init__(self, left=None, right=None, alpha=None, beta=None, max_features=None, criterion='gini',
                 random_state=None):
        self.left = left
        self.right = right

        self.test = ()  # the best pair of (feature_idx, threshold)
        self.node_stats = {} # dictionary contains counts of objects from a class in the node
        self.left_node_stats = {} # dictionary contains counts of objects from a class in the left child node
        self.right_node_stats = {} # dictionary contains counts of objects from a class in the right child node

        self.is_terminal = True
        self.alpha = alpha
        self.beta = beta
        self.max_features = max_features
        self.X_cache = None
        self.y_cache = None
        self.random_state = None
        self.criterion = criterion

    def get_node_stats(self, y, node_stats):
        if y not in self.node_stats:
            node_stats[y] = 0
        node_stats[y] += 1

    def _split(self):

        m, d = self.X_cache.shape

        np.random.seed(self.random_state)
        feat_idxs = np.random.permutation(d)

        max_features = {'log': np.log2, 'sqrt': np.sqrt}
        if self.max_features:
            d = int(np.floor(max_features[self.max_features](d)))
            feat_idxs = np.random.choice(feat_idxs, d, replace=False)


        max_ig = -np.inf

        if self.criterion == 'gini':
            criterion = gini_online
        else:
            criterion = entropy_online

        for feat_idx in feat_idxs:
            min_feat_val = np.min(self.X_cache[:, feat_idx])
            max_feat_val = np.max(self.X_cache[:, feat_idx])

            threshold = np.random.uniform(min_feat_val, max_feat_val, 1)

            left_mask = np.argwhere(self.X_cache[:, feat_idx] < threshold)
            right_mask = np.argwhere(self.X_cache[:, feat_idx] >= threshold)

            y_left, y_right = self.y_cache[left_mask], self.y_cache[right_mask]
            self.get_node_stats(y_left, self.left_node_stats)
            self.get_node_stats(y_right, self.right_node_stats)

            ig = inf_gain_online(self.node_stats, self.left_node_stats, self.right_node_stats, criterion)

            if ig > max_ig:
                max_ig = ig
                best_feature_idx = feat_idx
                best_threshold = threshold

        return best_feature_idx, best_threshold, max_ig

    def propagate(self, x, y):
        if self.is_terminal:
            self.get_node_stats(y, self.node_stats)

            m, d = self.X_cache.shape
            if  m > self.alpha:
                self.best_feature_idx, self.best_threshold, max_ig = self._split

                if max_ig > self.beta:
                    # splitting conditions are satisfied
                    self.is_terminal = False
                    # create left child node
                    self.left = Node(alpha=self.alpha, beta=self.beta, criterion=self.criterion,
                                     max_features=self.max_features, random_state=self.random_state
                                     )
                    self.left.node_stats = self.left_node_stats
                    # create right child node
                    self.right = Node(alpha=self.alpha, beta=self.beta, criterion=self.criterion,
                                     max_features=self.max_features, random_state=self.random_state
                                     )
                    self.right.node_stats = self.right_node_stats
        else:
            if x[self.best_feature_idx] < self.best_threshold:
                self.left.propagate(x, y)
            else:
                self.right.propagate(x, y)


class DTOnlineClassifier(object):

    def __init__(self, criterion='gini', max_features=None, alpha=100, beta=0.1, random_state=None):

        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.root_node = Node(max_features=self.max_features, alpha=alpha, beta=beta,
                              criterion=self.criterion, random_state=self.random_state)


    def fit(self, x, y):

        self.root_node.propagate(x, y)

        return self

    def predict(self, X):
        pass


