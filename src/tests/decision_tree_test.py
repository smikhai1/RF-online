import sys
import numpy as np
from sklearn.datasets import load_iris

sys.path.append('/Users/mikhail/projects/edu/skoltech/ml/project/RF-online/src/')

from offline.decision_tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)

clf = DecisionTreeClassifier()

clf.fit(X, y)
y_pred = clf.predict(X)

print(y_pred)