# sequential floating feature selection and feature selection and RBF-kernel SVM
import sys
import pickle
import random as rd
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.svm import SVC as svc
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from mlxtend.feature_selection.sequential_feature_selector import SequentialFeatureSelector
from pre_process import create_labels


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
    return round(sum([t[index][y_pred[i]] + z[index]
                      for i, index in enumerate(indices)]) / len(indices), 3)

def find_rank(index, label):
    r1, r2 = -1, 0
    tmp_list = sorted(t[index])
    for i, elem in enumerate(tmp_list):
        if elem == t[index, label]:
            r1 = i + 1
            break
    # EAX.restart, label 4
    if t[index, label] <= t[index, 4]:
        r2 = 1
    return r1, r2

if __name__ == "__main__":
    # number of threads
    n_cores = int(sys.argv[1])
    # random seed, related to train/test split and CV split
    seed = int(sys.argv[2])
    rd.seed(seed, version=2)

    feature_num = sys.argv[3]
    if feature_num != 'best':
        feature_num = int(feature_num)
    # create_labels(t_max, penalize_factor, alg_num, repeat)
    cl = create_labels(900.0, 10, 6, 5)
    X, y, z, t = cl()
    insts_size = X.shape[0]
    X_pd = pd.DataFrame(X)

    # train_test_split
    X_train, X_test, _, _ = train_test_split(X_pd, y, test_size=0.3, random_state=seed)
    train_index = X_train.index.values
    test_index = X_test.index.values

    X_train = X[train_index]

    y = np.hstack((np.arange(insts_size).reshape((insts_size, 1)), y.reshape((insts_size, 1))))
    y_train = y[train_index]

    # initialize svm with RBF kernel
    classifier = svc(kernel='rbf', gamma='scale')

    # initialize 5-fold cv splits
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    kf = list(kf.split(range(X_train.shape[0])))

    scorer = make_scorer(par10, greater_is_better=False)

    # initialize sffs feature selector
    selector = SequentialFeatureSelector(estimator=CutEstimator(classifier), k_features=feature_num,
                                         forward=True, floating=True, verbose=2,
                                         scoring=scorer, cv=kf, n_jobs=n_cores,
                                         pre_dispatch='n_jobs', clone_estimator=True,
                                         fixed_features=None)
    selector.fit(X_train, y_train)
    pickle.dump(selector, open('selector', 'wb'))

    # use the selected features to build selector and test it
    subset = np.array(selector.k_feature_idx_)
    X_train = X[train_index, :][:, subset]
    X_test = X[test_index, :][:, subset]
    y_train = y[train_index]
    y_test = y[test_index]

    # initialize svm with RBF kernel
    classifier = svc(kernel='rbf', gamma='scale')
    classifier.fit(X_train, y_train)
    pickle.dump(classifier, open('classifier', 'wb'))

    # stastics of results
    f = open('summary', 'w+')
    sys.stdout = f
    y_train_pred = classifier.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    par10_train = par10(y_train, y_train_pred, train_index)
    print('Train acc: %f' % train_acc, 'Train Par10: %f' % par10_train)
    print('The portion of each predicted class on training set:')
    print('LKH (label 0): ', np.sum(y_train_pred == 0)/y_train_pred.shape[0])
    print('LKH.restart (label 1): ', np.sum(y_train_pred == 1)/y_train_pred.shape[0])
    print('LKH.crossover (label 2): ', np.sum(y_train_pred == 2)/y_train_pred.shape[0])
    print('EAX (label 3): ', np.sum(y_train_pred == 3)/y_train_pred.shape[0])
    print('EAX.restart (label 4): ', np.sum(y_train_pred == 4)/y_train_pred.shape[0])
    print('MAOS (label 5): ', np.sum(y_train_pred == 5)/y_train_pred.shape[0])
    mean_rank_train = sum([find_rank(index, y_train_pred[i])[0]
                           for i, index in enumerate(train_index)]) / y_train_pred.shape[0]
    better_sbs_train = sum([find_rank(index, y_train_pred[i])[1]
                            for i, index in enumerate(train_index)]) / y_train_pred.shape[0]
    print('mean_rank_train: %f' % round(mean_rank_train, 2))
    print('not_worse_than_sbs_train: %f' % better_sbs_train)

    y_test_pred = classifier.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    par10_test = par10(y_test, y_test_pred, test_index)
    print('------------------------------------------------------')
    print('Test acc: %f' % test_acc, 'Test Par10: %f' % par10_test)
    print('The portion of each predicted class on test set:')
    print('LKH (label 0): ', np.sum(y_test_pred == 0)/y_test_pred.shape[0])
    print('LKH.restart (label 1): ', np.sum(y_test_pred == 1)/y_test_pred.shape[0])
    print('LKH.crossover (label 2): ', np.sum(y_test_pred == 2)/y_test_pred.shape[0])
    print('EAX (label 3): ', np.sum(y_test_pred == 3)/y_test_pred.shape[0])
    print('EAX.restart (label 4): ', np.sum(y_test_pred == 4)/y_test_pred.shape[0])
    print('MAOS (label 5): ', np.sum(y_test_pred == 5)/y_test_pred.shape[0])
    mean_rank_test = sum([find_rank(index, y_test_pred[i])[0]
                          for i, index in enumerate(test_index)]) / y_test_pred.shape[0]
    better_sbs_test = sum([find_rank(index, y_test_pred[i])[1]
                           for i, index in enumerate(test_index)]) / y_test_pred.shape[0]
    print('mean_rank_test: %f' % round(mean_rank_test, 2))
    print('not_worse_than_sbs_test: %f' % better_sbs_test)

    f.close()
