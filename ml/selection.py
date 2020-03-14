# sequential floating feature selection and feature selection and RBF-kernel SVM
import sys
import pickle
import numpy as np
import arff
from sklearn.base import BaseEstimator, clone
from sklearn.svm import SVC as svc
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from mlxtend.feature_selection.sequential_feature_selector import SequentialFeatureSelector


class CutEstimator(BaseEstimator):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, _X, _y):
        self._base_estimator = clone(self.base_estimator)
        self._base_estimator.fit(_X, _y[:, 1].ravel())
        return self

    def predict(self, _X):
        return self._base_estimator.predict(_X)

# initialize par-10 scorer
def par10(y_truth, y_pred, indices=None):
    if indices is None:
        indices = y_truth[:, 0]
    # print(indices)
    return round(sum([labels[index]['runtime_%d' % y_pred[i]] + Z[index][-1]
                      for i, index in enumerate(indices)]) / len(indices), 3)

if __name__ == "__main__":
    # number of threads
    n_cores = int(sys.argv[1])

    # train_test_split
    variables = np.load(open("../data/aslib_data-not_verified/TSP-ECJ2018/"
                             "train_test_split.npz", 'rb'))
    train_index = variables['train_index']

    # read labels, features and costs for computing features
    labels = np.load(open("../data/aslib_data-not_verified/TSP-ECJ2018/labels.npy", 'rb'))
    y = labels['label']
    insts_size = labels.shape[0]
    y = np.hstack((np.arange(insts_size).reshape((insts_size, 1)), y.reshape((insts_size, 1))))
    y = y[train_index, :]
    insts_size = y.shape[0]

    X = arff.load(
        open("../data/aslib_data-not_verified/TSP-ECJ2018/feature_values.arff", 'r'))
    X = X['data']
    # total feature number: 405, Pihera feature number: 287
    X = np.array([x[-287:] for x in X[0:-1:10]], dtype='f')
    X = X[train_index, :]

    Z = arff.load(
        open("../data/aslib_data-not_verified/TSP-ECJ2018/feature_costs.arff", 'r'))
    Z = Z['data']
    Z = np.array([z[2:] for z in Z[0:-1:10]], dtype='f')

    # initialize svm with RBF kernel
    classifier = svc(kernel='rbf', gamma='scale')

    # initialize 5-fold cv splits
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf = list(kf.split(range(insts_size)))

    scorer = make_scorer(par10, greater_is_better=False)

    # initialize sffs feature selector
    selector = SequentialFeatureSelector(estimator=CutEstimator(classifier), k_features=16,
                                         forward=True, floating=True, verbose=2,
                                         scoring=scorer, cv=kf, n_jobs=n_cores,
                                         pre_dispatch='n_jobs', clone_estimator=True,
                                         fixed_features=None)
    selector.fit(X, y)
    pickle.dump(selector, open('selector', 'wb'))
