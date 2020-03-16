# train a classifier with selected features
import sys
import pickle
import numpy as np
import arff
from sklearn.svm import SVC as svc
from sklearn.metrics import accuracy_score
from .feature_selection_for_ECJ import CutEstimator, par10

if __name__ == '__main__':
    # train_test_split
    variables = np.load(open("../data/aslib_data-not_verified/TSP-ECJ2018/"
                             "train_test_split.npz", 'rb'))
    train_index = variables['train_index']
    test_index = variables['test_index']

    # read labels, features and costs for computing features
    labels = np.load(
        open("../data/aslib_data-not_verified/TSP-ECJ2018/labels.npy", 'rb'))
    y_all = labels['label']
    y_train = y_all[train_index]
    y_test = y_all[test_index]

    features = arff.load(
        open("../data/aslib_data-not_verified/TSP-ECJ2018/feature_values.arff", 'r'))
    X_all = features['data']
    # total feature number: 405, Pihera feature number: 287
    X_all = np.array([x[-287:] for x in X_all[0:-1:10]], dtype='f')
    selector = pickle.load(open('selector', 'rb'))
    subset = np.array(selector.k_feature_idx_)
    X_train = X_all[train_index, :][:, subset]
    X_test = X_all[test_index, :][:, subset]

    Z = arff.load(
        open("../data/aslib_data-not_verified/TSP-ECJ2018/feature_costs.arff", 'r'))
    Z = Z['data']
    Z = np.array([z[2:] for z in Z[0:-1:10]], dtype='f')

    # initialize svm with RBF kernel
    classifier = svc(kernel='rbf', gamma='scale')
    classifier.fit(X_train, y_train)
    pickle.dump(classifier, open('classifier', 'wb'))

    def par_10(y_truth, y_pred, indices=None):
        if indices is None:
            indices = y_truth[:, 0]
        # print(indices)
        return round(sum([labels[index]['runtime_%d' % y_pred[i]] + Z[index][-1]
                          for i, index in enumerate(indices)]) / len(indices), 3)

    def find_rank(index, label):
        tmp = labels[index]
        r1, r2 = -1, 0
        tmp_list = sorted([tmp['runtime_1'], tmp['runtime_2'],
                           tmp['runtime_3'], tmp['runtime_4'], tmp['runtime_5']])
        for i, elem in enumerate(tmp_list):
            if elem == tmp['runtime_%d' % label]:
                r1 = i + 1
                break
        if tmp['runtime_%d' % label] <= tmp['runtime_2']:
            r2 = 1
        return r1, r2

    # stastics of results
    f = open('summary', 'w+')
    sys.stdout = f
    y_train_pred = classifier.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    par10_train = par_10(y_train, y_train_pred, train_index)
    print('Train acc: %f' % train_acc, 'Train Par10: %f' % par10_train)
    print('The portion of each predicted class on training set:')
    print('eax (label 1): ', np.sum(y_train_pred == 1)/y_train_pred.shape[0])
    print('eax.restart (label 2): ', np.sum(
        y_train_pred == 2)/y_train_pred.shape[0])
    print('lkh (label 3): ', np.sum(y_train_pred == 3)/y_train_pred.shape[0])
    print('lkh.restart (label 4): ', np.sum(
        y_train_pred == 4)/y_train_pred.shape[0])
    print('maos (label 5): ', np.sum(y_train_pred == 5)/y_train_pred.shape[0])
    mean_rank_train = sum([find_rank(index, y_train_pred[i])[0]\
                           for i, index in enumerate(train_index)]) / y_train_pred.shape[0]
    better_sbs_train = sum([find_rank(index, y_train_pred[i])[1]
                            for i, index in enumerate(train_index)]) / y_train_pred.shape[0]
    print('mean_rank_train: %f' % round(mean_rank_train, 2))
    print('not_worse_than_sbs_train: %f' % better_sbs_train)

    y_test_pred = classifier.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    par10_test = par_10(y_test, y_test_pred, test_index)
    print('------------------------------------------------------')
    print('Test acc: %f' % test_acc, 'Test PAR10: %f' % par10_test)
    print('The portion of each predicted class on test set:')
    print('eax (label 1): ', np.sum(y_test_pred == 1)/y_test_pred.shape[0])
    print('eax.restart (label 2): ', np.sum(
        y_test_pred == 2)/y_test_pred.shape[0])
    print('lkh (label 3): ', np.sum(y_test_pred == 3)/y_test_pred.shape[0])
    print('lkh.restart (label 4): ', np.sum(
        y_test_pred == 4)/y_test_pred.shape[0])
    print('maos (label 5): ', np.sum(y_test_pred == 5)/y_test_pred.shape[0])
    mean_rank_test = sum([find_rank(index, y_test_pred[i])[0]
                          for i, index in enumerate(test_index)]) / y_test_pred.shape[0]
    better_sbs_test = sum([find_rank(index, y_test_pred[i])[1]
                           for i, index in enumerate(test_index)]) / y_test_pred.shape[0]
    print('mean_rank_test: %f' % round(mean_rank_test, 2))
    print('not_worse_than_sbs_test: %f' % better_sbs_test)

    f.close()
