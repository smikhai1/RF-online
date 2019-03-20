import numpy as np
from online.decision_tree import DTOnlineClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mode


class RFOnlineClassifier(object):

    def __init__(self, n_estimators=10, criterion='gini', max_features='sqrt', random_state=None,
                 alpha=100, beta=0.1):

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_ = []
        self.oob_scores_ = None

        # initialization
        for i in range(self.n_estimators):
            self.estimators_.append(DTOnlineClassifier(criterion=self.criterion, max_features=self.max_features,
                                                       random_state=self.random_state, alpha=alpha, beta=beta)
                                    )

    def fit(self, x, y):

        for t in range(self.n_estimators):

            k = int(np.random.poisson(1, 1))
            if k > 0:
                # make bootstrap
                for _ in range(k):
                    self.estimators_[t].fit(x, y)
            #else:
                # compute oob score
                #y_pred = self.estimators_[t].predict(x)
                #self.oob_scores_[t] = accuracy_score(y, y_pred)

        return self

    def predict(self, X):
        n, d = X.shape

        y_preds = np.empty((n, self.n_estimators))

        for t, base_esitmator in enumerate(self.estimators_):
            y_preds[:, t] = base_esitmator.predict(X).squeeze()

        # aggregate predictions
        y_pred, _ = mode(y_preds, axis=1)

        return y_pred


