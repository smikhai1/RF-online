import numpy as np

# utils for offline
def gini(y):
    _, p = np.unique(y, return_counts=True)
    p = p / p.sum()
    value = 1 - (p ** 2).sum()

    return value

def entropy(y):
    _, p = np.unique(y, return_counts=True)
    p = p / p.sum()
    value = -(p * np.log2(p)).sum()

    return value

def inf_gain(X, y, left_mask, right_mask, criterion):

    X_left, y_left = X[left_mask, :], y[left_mask]
    X_right, y_right = X[right_mask, :], y[right_mask]

    ig = criterion(y) - len(y_left)/len(y) * criterion(y_left) - len(y_right)/len(y) * criterion(y_right)

    return ig


# utils for online
def gini_online(node_stats):
    p = np.empty(len(node_stats))
    for i, count in enumerate(node_stats.values()):
        p[i] = count

    p = p / p.sum()
    value = 1 - (p ** 2).sum()

    return value

def entropy_online(node_stats):
    p = np.empty(len(node_stats))
    for i, count in enumerate(node_stats.values()):
        p[i] = count

    p = p / p.sum()
    value = -(p * np.log2(p)).sum()

    return value

def inf_gain_online(node_stats, left_node_stats, right_node_stats, criterion):

    ig = criterion(node_stats) - len(left_node_stats)/len(node_stats) * criterion(left_node_stats) \
                               - len(right_node_stats)/len(node_stats) * criterion(right_node_stats)

    return ig