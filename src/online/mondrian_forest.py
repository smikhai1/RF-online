import numpy as np
from scipy.stats import uniform


class Node:
    def __init__(self, X, y, lam, parent=None, tree=None, flag=True):
        self.tree = tree
        self.parent = parent
        self.left = None
        self.right = None
        self.lam = lam
        self.discount = 0
        self.l = np.zeros(X.shape[1])
        self.u = np.zeros(X.shape[1])
        self.diff = np.zeros(X.shape[1])
        self.cjk = np.zeros_like(self.tree.classes)
        self.leaf = True
        self.tables = np.zeros_like(self.tree.classes)
        if flag:
            self.sample_mondrian_block(X, y)

    def initialize_posterior_counts(self, X, y):
        for i, cls in enumerate(self.tree.classes):
            self.cjk[i] = np.count_nonzero(y == cls)
        this_node = self
        while True:
            if this_node.leaf == False:
                table_left_j_k = this_node.left.tables if this_node.left else np.zeros_like(this_node.cjk)
                table_right_j_k = this_node.right.tables if this_node.right else np.zeros_like(this_node.cjk)
                this_node.cjk = table_left_j_k + table_right_j_k

            this_node.tables = np.minimum(this_node.cjk, 1)
            if this_node.parent is None:
                return
            else:
                this_node = this_node.parent

    def update_posterior_counts(self, y):
        class_index = self.tree.class_indices[y]
        self.cjk[class_index] += 1
        this_node = self
        while True:
            if this_node.tables[class_index] == 1:
                return
            else:
                if this_node.leaf == False:
                    table_left_j_y = this_node.left.tables[class_index] if this_node.left else 0
                    table_right_j_y = this_node.right.tables[class_index] if this_node.right else 0
                    this_node.cjk[class_index] = table_left_j_y + table_right_j_y
                this_node.tables[class_index] = np.minimum(this_node.cjk[class_index], 1)
                if this_node.parent is None:
                    return
                else:
                    this_node = this_node.parent

    def sample_mondrian_block(self, X, y):

        self.l = np.min(X, axis=0)
        self.u = np.max(X, axis=0)
        self.diff = self.u - self.l

        if np.all(y == y[0]) or len(y) <= 0:
            self.tilda = self.lam
        else:
            E = np.random.exponential(1.0 / self.diff.sum())
            parent_tilda = self.parent.tilda if self.parent is not None else 0
            self.tilda = parent_tilda + E

        if self.tilda < self.lam:
            self.delta = np.random.choice(np.arange(X.shape[1]), p=(self.diff / self.diff.sum()))
            self.xi = uniform.rvs(loc=self.l[self.delta], scale=self.diff[self.delta])
            left_indices = X[:, self.delta] <= self.xi

            N_left_j = X[X[:, self.delta] <= self.xi]
            D_left_j = y[X[:, self.delta] <= self.xi]
            N_right_j = X[X[:, self.delta] > self.xi]
            D_right_j = y[X[:, self.delta] > self.xi]
            self.leaf = False

            self.left = Node(N_left_j, D_left_j, lam=self.lam, parent=self, tree=self.tree, flag=False)
            self.left.sample_mondrian_block(N_left_j, D_left_j)
            self.right = Node(N_right_j, D_right_j, lam=self.lam, parent=self, tree=self.tree, flag=False)
            self.right.sample_mondrian_block(N_right_j, D_right_j)
        else:
            self.tilda = self.lam
            self.tree.leaf_nodes.add(self)
            self.initialize_posterior_counts(X, y)


class MondrianTree:

    def __init__(self, lam=np.inf, classes=None):
        self.leaf_nodes = set()
        self.classes = classes
        if classes is not None:
            self.class_indices = {cls: i for i, cls in enumerate(self.classes)}
        else:
            self.class_indices = None
        self.X = None
        self.y = None
        self.lam = lam
        self.root = None
        self.has_fit = False

    def fit(self, X, y):
        self.X = X
        self.y = y
        if self.classes is None:
            self.classes = np.unique(y)
            self.class_indices = {cls: i for i, cls in enumerate(self.classes)}

        self.root = Node(X, y, parent=None, lam=self.lam, tree=self)
        self.compute_predictive_posterior()

        self.has_fit = True

    def compute_predictive_posterior(self):
        J = []
        J.append(self.root)
        while len(J) != 0:
            element = J.pop()
            if element.parent is None:
                parent_posterior = np.ones_like(self.classes) / len(self.classes)
            else:
                parent_posterior = element.parent.posterior_predictive

            cjk = element.cjk
            tables = element.tables
            discount = element.discount

            element.posterior_predictive = (cjk - discount * tables + discount * np.sum(
                tables) * parent_posterior) / np.sum(cjk)

            if element.left:
                J.append(element.left)
            if element.right:
                J.append(element.right)

    def predict(self, x):
        this_node = self.root
        p_not_sep_yet = 1.0
        s_k = np.zeros((len(self.classes),))
        gamma = 15
        while True:
            parent_tilda = this_node.parent.tilda if this_node.parent is not None else 0
            tilda_difference = this_node.tilda - parent_tilda
            nu_j = np.sum(np.maximum(x - this_node.u, 0) + np.maximum(this_node.l - x, 0))
            p_s = 1 - np.exp(-nu_j * tilda_difference) if tilda_difference < np.inf else 1
            if p_s > 0:

                E_discount = (nu_j / (nu_j + gamma)) * (1 - np.exp(-(nu_j + gamma) * tilda_difference)) / p_s

                cjk = tables = np.minimum(this_node.cjk, 1)

                if this_node.parent is None:
                    parent_pos = np.ones_like(self.classes) / len(self.classes)
                else:
                    parent_pos = this_node.parent.posterior_predictive

                posterior = (cjk / np.sum(cjk) - E_discount * tables + E_discount * tables.sum() * parent_pos)
                s_k += p_not_sep_yet * p_s * posterior
            if this_node.leaf:
                s_k += p_not_sep_yet * (1 - p_s) * this_node.posterior_predictive
                return s_k
            else:
                p_not_sep_yet = p_not_sep_yet * (1 - p_s)
                if x[this_node.delta] <= this_node.xi:
                    this_node = this_node.left
                else:
                    this_node = this_node.right


class MRF:
    def __init__(self, n_estimators=10, lam=np.inf):
        self.n_estimators = n_estimators
        self.trees = []
        self.lam = lam

    def fit(self, X, y):
        for i in range(self.n_estimators):
            self.trees.append(MondrianTree(self.lam))
            self.trees[i].fit(X, y)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.trees[0].classes)))
        for i, x in enumerate(X):
            y_pred = np.zeros((self.n_estimators, len(self.trees[0].classes)))
            for j, tree in enumerate(self.trees):
                y_pred[j] = tree.predict(x)
            predictions[i] = y_pred.mean(axis=0)

        return np.argmax(predictions, axis=1)