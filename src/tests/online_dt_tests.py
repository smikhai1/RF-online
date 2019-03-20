import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.append('/Users/mikhail/projects/edu/skoltech/ml/project/RF-online/src/')

from online.decision_tree import DTOnlineClassifier


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
m, d = X_train.shape

clf = DTOnlineClassifier(alpha=10)

for i in range(m):
    x = X_train[i, :]
    y_ = y_train[i]

    clf.fit(x, y_)

y_pred = clf.predict(X_test[:5, :])

print(accuracy_score(y_test[:5], y_pred))