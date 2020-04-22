# sequential floating feature selection and feature selection and RBF-kernel SVM
import sys
import time
import pickle
import math
import random as rd
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.svm import SVC as svc
from sklearn.svm import SVR as svr
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, accuracy_score
from mlxtend.feature_selection.sequential_feature_selector import SequentialFeatureSelector
from pre_process import create_labels


class CutClassifier(BaseEstimator):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, _X, _y):
        self._base_estimator = clone(self.base_estimator)
        self._base_estimator.fit(_X, _y[:, 1].reshape(_y.shape[0]))
        return self

    def predict(self, _X):
        return self._base_estimator.predict(_X)

class CutRegressor(BaseEstimator):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, _X, _y):
        alg_num = _y.shape[1] - 1
        self._base_estimator_list = [clone(self.base_estimator).fit(_X, _y[:, i+1].reshape(_y.shape[0]))\
                                     for i in range(alg_num)]
        return self

    def predict(self, _X):
        return np.argmin(np.concatenate([estimator.predict(_X).reshape((_X.shape[0], 1)) for\
                                         estimator in self._base_estimator_list], axis=1), axis=1)

class CutPairedRegressor(BaseEstimator):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, _X, _y):
        tmp = _y.shape[1] - 1
        alg_num = int((1 + math.sqrt(1+8*tmp)) / 2)
        self.alg_num = alg_num
        self._base_estimator_list = []
        self.dict = dict()
        index = 0
        for i in range(alg_num):
            for j in range(i+1, alg_num):
                self._base_estimator_list.append(\
                    clone(self.base_estimator).fit(_X, _y[:, index+1].reshape(_y.shape[0])))
                self.dict[(i, j)] = index
        return self

    def predict(self, _X):
        tmp = np.zeros((_X.shape[0], self.alg_num))
        alg_num = self.alg_num
        for i in range(alg_num):
            for j in range(alg_num):
                if i == j:
                    continue
                if i < j:
                    tmp[:, i] = tmp[:, i] + self._base_estimator_list[self.dict[(i, j)]].predict(_X)
                else:
                    tmp[:, i] = tmp[:, i] - self._base_estimator_list[self.dict[(j, i)]].predict(_X)
        return np.argmin(tmp, axis=1)

# initialize par-10 scorer
def par10(y_truth, y_pred, indices=None):
    if indices is None:
        indices = y_truth[:, 0].astype(int)
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
    # random seed, related to CV split
    seed = int(sys.argv[2])
    rd.seed(seed, version=2)

    feature_num = sys.argv[3]
    if feature_num != 'best':
        feature_num = int(feature_num)
    feature_index_dict = {'Phiera' : (0, 287),
                          'UBC-cheap' : (288, 300),
                          'UBC' : (288, 337),
                          'All' : (0, 337)}
    feature_time_index_dict = {'Phiera' : 0,
                               'UBC-cheap' : 1,
                               'UBC' : 2
                               }

    scorer = make_scorer(par10, greater_is_better=False)
    f = open('results/summary', 'w+')

    for method in ['classification', 'regression', 'paired-regression']:
        cl = create_labels(900.0, 10, 6, 5, method)
        # create_labels(t_max, penalize_factor, alg_num, repeat, method)
        ori_X, ori_y, ori_z, t, labels, train_index, test_index = cl()
        for feature_set in ['Phiera', 'UBC-cheap', 'UBC', 'All']:
            X = ori_X[:, feature_index_dict[feature_set][0] : feature_index_dict[feature_set][1]]
            if feature_set == 'All':
                z = ori_z[:, 0] + ori_z[:, 2]
            else:
                z = ori_z[:, feature_time_index_dict[feature_set]]

            insts_size = X.shape[0]
            X_train = X[train_index]
            # initialize 5-fold cv splits
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            kf = list(kf.split(range(X_train.shape[0])))

            if method == 'classification':
                y = np.hstack((np.arange(insts_size).reshape((insts_size, 1)),\
                               ori_y.reshape((insts_size, 1))))
                y_train = y[train_index, :]
                model_names = [(dtc, 'dtc'), (rfc, 'rfc'), (svc, 'svc')]
            elif method == 'regression':
                y = np.hstack((np.arange(insts_size).reshape((insts_size, 1)), ori_y))
                y_train = y[train_index, :]
                model_names = [(dtr, 'dtr'), (rfr, 'rfr'), (svr, 'svr')]
            elif method == 'paired-regression':
                y = np.hstack((np.arange(insts_size).reshape((insts_size, 1)), ori_y))
                y_train = y[train_index, :]
                model_names = [(dtr, 'dtr'), (rfr, 'rfr'), (svr, 'svr')]

            for base_model, learner in model_names:
                start = time.time()
                print('method=%s, learner=%s, feature-set=%s\n' % (method, learner, feature_set))
                # use default hyper-parameters
                model = base_model()

                # initialize sffs feature selector
                if method == 'classification':
                    selector = SequentialFeatureSelector(estimator=CutClassifier(model),
                                                         k_features=min(feature_num, X.shape[1]),
                                                         forward=True, floating=True, verbose=2,
                                                         scoring=scorer, cv=kf, n_jobs=n_cores,
                                                         pre_dispatch='2*n_jobs',
                                                         clone_estimator=True,
                                                         fixed_features=None)
                elif method == 'regression':
                    selector = SequentialFeatureSelector(estimator=CutRegressor(model),
                                                         k_features=min(feature_num, X.shape[1]),
                                                         forward=True, floating=True, verbose=2,
                                                         scoring=scorer, cv=kf, n_jobs=n_cores,
                                                         pre_dispatch='2*n_jobs',
                                                         clone_estimator=True,
                                                         fixed_features=None)
                elif method == 'paired-regression':
                    selector = SequentialFeatureSelector(estimator=CutPairedRegressor(model),
                                                         k_features=min(feature_num, X.shape[1]),
                                                         forward=True, floating=True, verbose=2,
                                                         scoring=scorer, cv=kf, n_jobs=n_cores,
                                                         pre_dispatch='2*n_jobs',
                                                         clone_estimator=True,
                                                         fixed_features=None)

                selector.fit(X_train, y_train)
                end = time.time()
                pickle.dump(selector, open('results/selector_%s_%s_%s' %\
                                           (method, learner, feature_set), 'wb'))

                # use the selected features to build selector and test it
                subset = np.array(selector.k_feature_idx_)
                X_train = X[train_index, :][:, subset]
                X_test = X[test_index, :][:, subset]
                y_train = y[train_index, :]
                y_test = y[test_index, :]

                if method == 'classification':
                    model = CutClassifier(base_model())
                elif method == 'regression':
                    model = CutRegressor(base_model())
                elif method == 'paired-regression':
                    model = CutPairedRegressor(base_model())

                model.fit(X_train, y_train)
                pickle.dump(model, open('results/%s_%s_%s' %\
                                        (method, learner, feature_set), 'wb'))

                f.write('------------------------------------------------------------------\n')
                f.write('method=%s, learner=%s, feature-set=%s, time=%s\n' % (method, learner, feature_set, str(end-start)))
                f.write('-----------------------------Train--------------------------------\n')
                label_train_pred = model.predict(X_train)
                train_acc = accuracy_score(labels[train_index], label_train_pred)
                par10_train = par10(y_train, label_train_pred, train_index)
                f.write('Train acc: %f Train Par10: %f\n' % (train_acc, par10_train))
                f.write('The portion of each predicted class on training set:\n')
                f.write('LKH (label 0): %f\n' % (np.sum(label_train_pred == 0)/label_train_pred.shape[0]))
                f.write('LKH.restart (label 1): %f\n' %\
                        (np.sum(label_train_pred == 1)/label_train_pred.shape[0]))
                f.write('LKH.crossover (label 2): %f\n' %\
                        (np.sum(label_train_pred == 2)/label_train_pred.shape[0]))
                f.write('EAX (label 3): %f\n' %\
                        (np.sum(label_train_pred == 3)/label_train_pred.shape[0]))
                f.write('EAX.restart (label 4): %f\n' %\
                        (np.sum(label_train_pred == 4)/label_train_pred.shape[0]))
                f.write('MAOS (label 5): %f\n' %\
                        (np.sum(label_train_pred == 5)/label_train_pred.shape[0]))
                mean_rank_train = sum([find_rank(index, label_train_pred[i])[0] for i, index in\
                                       enumerate(train_index)]) / label_train_pred.shape[0]
                better_sbs_train = sum([find_rank(index, label_train_pred[i])[1] for i, index in\
                                        enumerate(train_index)]) / label_train_pred.shape[0]
                f.write('mean_rank_train: %f\n' % round(mean_rank_train, 2))
                f.write('not_worse_than_sbs_train: %f\n' % better_sbs_train)

                label_test_pred = model.predict(X_test)
                test_acc = accuracy_score(labels[test_index], label_test_pred)
                par10_test = par10(y_test, label_test_pred, test_index)
                f.write('-----------------------------Test--------------------------------\n')
                f.write('Test acc: %f Test Par10: %f\n' % (test_acc, par10_test))
                f.write('The portion of each predicted class on test set:\n')
                f.write('LKH (label 0): %f\n' % (np.sum(label_test_pred == 0)/label_test_pred.shape[0]))
                f.write('LKH.restart (label 1): %f\n' %\
                         (np.sum(label_test_pred == 1)/label_test_pred.shape[0]))
                f.write('LKH.crossover (label 2): %f\n' %\
                         (np.sum(label_test_pred == 2)/label_test_pred.shape[0]))
                f.write('EAX (label 3): %f\n' % (np.sum(label_test_pred == 3)/label_test_pred.shape[0]))
                f.write('EAX.restart (label 4): %f\n' %\
                         (np.sum(label_test_pred == 4)/label_test_pred.shape[0]))
                f.write('MAOS (label 5): %f\n' % (np.sum(label_test_pred == 5)/label_test_pred.shape[0]))
                mean_rank_test = sum([find_rank(index, label_test_pred[i])[0] for i, index in\
                                      enumerate(test_index)]) / label_test_pred.shape[0]
                better_sbs_test = sum([find_rank(index, label_test_pred[i])[1] for i, index in\
                                       enumerate(test_index)]) / label_test_pred.shape[0]
                f.write('mean_rank_test: %f\n' % round(mean_rank_test, 2))
                f.write('not_worse_than_sbs_test: %f\n' % better_sbs_test)
                f.flush()
    f.close()
