import numpy as np
from offline.decision_tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from scipy.stats import mode
from sklearn.utils import resample
from sklearn.metrics import accuracy_score


### TODO 1. Implement computation of OOB scores
### TODO 2. Make multi-thread implementation

class RandomForestClassifier(BaseEstimator):

    def __init__(self, n_estimators=10, max_depth=np.inf, min_samples_split=2, criterion='gini', splitter='best',
                 max_features='sqrt', bootstrap=True, oob_score=False, random_state=None):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.splitter = splitter
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.estimators_ = None
        self.oob_scores_ = None


    def fit(self, X, y):

        # initialize
        self.estimators_ = []
        self.oob_scores_ = []

        m, d = X.shape
        X_, y_ = np.copy(X), np.copy(y)
        for t in range(self.n_estimators):
            if self.bootstrap:
                # make bootstrap
                bootstrap_idx = np.random.choice(np.arange(m), m, replace=True)
                X_, y_ = X[bootstrap_idx], y[bootstrap_idx]

                # out of bag samples
                bag_idx = np.unique(bootstrap_idx)
                X_oob, y_oob = np.delete(X, bag_idx, axis=0), np.delete(y, bag_idx, axis=0)

            # create and fit a base estimator
            base_estimator = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                                    criterion=self.criterion, splitter=self.splitter,
                                                    max_features=self.max_features, random_state=self.random_state)
            self.estimators_.append(base_estimator.fit(X_, y_))

            if self.oob_score:
                # estimate out of bag (OOB) error for an estimator
                y_pred = base_estimator.predict(X_oob)
                self.oob_scores_.append(accuracy_score(y_oob, y_pred))

        return self

    def predict(self, X):
        n, d = X.shape

        y_preds = np.empty((n, self.n_estimators))

        for t, base_esitmator in enumerate(self.estimators_):
            y_preds[:, t] = base_esitmator.predict(X).squeeze()

        # aggregate predictions
        y_pred, _ = mode(y_preds, axis=1)

        return y_pred

    def refit(self, X, y, n_refitted=3):
        m, d = X.shape

        if not n_refitted:
            n_refitted = int(self.n_estimators // 2)

        if self.estimators_:
            # delete n_reffited trees from the ensamble
            oob_errors = 1 - np.array(self.oob_scores_)
            p = oob_errors / np.sum(oob_errors)
            ids = np.random.choice(np.arange(self.n_estimators), n_refitted, replace=False, p=p)
            self.oob_scores_ = list(np.delete(self.oob_scores_, ids))
            self.estimators_ = list(np.delete(self.estimators_, ids))


            # fit new n_reffited trees and add to the ensamble
            for t in range(n_refitted):
                if self.bootstrap:
                    # make bootstrap
                    bootstrap_idx = np.random.choice(np.arange(m), m, replace=True)
                    X_, y_ = X[bootstrap_idx], y[bootstrap_idx]

                    # out of bag samples
                    bag_idx = np.unique(bootstrap_idx)
                    X_oob, y_oob = np.delete(X, bag_idx, axis=0), np.delete(y, bag_idx, axis=0)

                # create and fit a base estimator
                base_estimator = DecisionTreeClassifier(max_depth=self.max_depth,
                                                        min_samples_split=self.min_samples_split,
                                                        criterion=self.criterion, splitter=self.splitter,
                                                        max_features=self.max_features, random_state=self.random_state)
                self.estimators_.append(base_estimator.fit(X_, y_))

                if self.oob_score:
                    # estimate out of bag (OOB) error for an estimator
                    y_pred = base_estimator.predict(X_oob)
                    self.oob_scores_.append(accuracy_score(y_oob, y_pred))
        return self