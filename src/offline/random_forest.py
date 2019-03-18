import numpy as np
from offline.decision_tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from scipy.stats import mode
from sklearn.utils import resample


### TODO 1. Implement computation of OOB scores
### TODO 2. Make multi-thread implementation
class RandomForestClassifier(BaseEstimator):

    def __init__(self, n_estimators=10, max_depth=np.inf, min_samples_split=2, criterion='gini', splitter='best',
                 max_features='sqrt', bootstrap=True, oob_score=False, random_state=None):

        super(RandomForestClassifier, self).__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.splitter = splitter
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.estimators_ = []
        self.oob_scores_ = []


    def fit(self, X, y):
        m, d = X.shape
        X_, y_ = np.copy(X), np.copy(y)
        for t in range(self.n_estimators):
            if self.bootstrap:
                # make bootstrap
                X_, y_ = resample(X, y, replace=True)

            # create and fit a base estimator
            base_estimator = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                                    criterion=self.criterion, splitter=self.splitter,
                                                    max_features=self.max_features, random_state=self.random_state)
            self.estimators_.append(base_estimator.fit(X_, y_))
        return self

    def predict(self, X):
        n, d = X.shape

        y_preds = np.empty((n, self.n_estimators))

        for t, base_esitmator in enumerate(self.estimators_):
            y_preds[:, t] = base_esitmator.predict(X).squeeze()

        # aggregate predictions
        y_pred, _ = mode(y_preds, axis=1)

        return y_pred
