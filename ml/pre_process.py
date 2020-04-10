# pre-proccess script for ECJ paper data
# data structures: instance_name, feature, label
import random as rd
from copy import deepcopy
import pandas as pd
import arff
import numpy as np
from sklearn.model_selection import train_test_split


class create_labels_for_ECJ(object):
    def __init__(self, file, t_max=3600.0, pel=10, alg_num=5, rep_run=10):
        self.__file = file
        self.t_max = t_max
        self.pel = pel
        self.alg_num = alg_num
        self.rep_run = rep_run

    def __call__(self, out_dir="../data/aslib_data-not_verified/TSP-ECJ2018/"):
        runs = arff.load(open(self.__file, 'r'))
        x = [tuple(run) for run in runs['data']]
        x = np.array(x, dtype=[('id', 'S40'), ('run_count', '<i4'),
                               ('alg', 'S10'), ('runtime', 'f'), ('status', 'S10')])

        ins_num = int(x.shape[0] / (self.alg_num * self.rep_run))

        performance = np.ndarray(shape=(ins_num, ),
                                 dtype=[('id', 'S40'), ('label', 'i4'),
                                        ('runtime_1', 'f4'), ('status_1', 'S10'),
                                        ('runtime_2', 'f4'), ('status_2', 'S10'),
                                        ('runtime_3', 'f4'), ('status_3', 'S10'),
                                        ('runtime_4', 'f4'), ('status_4', 'S10'),
                                        ('runtime_5', 'f4'), ('status_5', 'S10')])
        count = 0
        for ins_index in range(ins_num):
            runtime_array = np.ndarray(shape=(self.alg_num,), dtype='f4')
            mean_runtime_array = np.ndarray(shape=(self.alg_num,), dtype='f4')
            for alg_index in range(self.alg_num):
                tmp = x[ins_index*self.alg_num*self.rep_run+alg_index :\
                        (ins_index+1)*self. alg_num*self.rep_run : self.alg_num]
                if np.sum(tmp['status'] == b'ok') >= 6:
                    runtime = np.round(np.median(tmp['runtime']), 3)
                    status = 'ok'
                else:
                    runtime = np.round(self.t_max * self.pel, 3)
                    status = 'timeout'
                performance[ins_index]['id'] = tmp[0]['id']
                performance[ins_index]['runtime_%d'%(alg_index+1)] = runtime
                performance[ins_index]['status_%d'%(alg_index+1)] = status
                runtime_array[alg_index] = runtime
                mean_runtime_array[alg_index] = np.round(np.mean(tmp['runtime']), 3)

            indices = np.where(runtime_array == runtime_array.min())[0]
            if indices.shape[0] > 1:
                count += 1
                indices = np.where(mean_runtime_array == mean_runtime_array.min())[0]

            performance[ins_index]['label'] = indices[0] + 1
        print('count %d' % count)
        np.save('%slabels.npy' % out_dir, performance)


def split_for_ECJ(ratio=0.2, out_dir="../data/aslib_data-not_verified/TSP-ECJ2018/"):
    # read labels, features and costs for computing features
    labels = np.load(
        open("../data/aslib_data-not_verified/TSP-ECJ2018/labels.npy", 'rb'))
    y = pd.Series(labels['label'])

    X = arff.load(
        open("../data/aslib_data-not_verified/TSP-ECJ2018/feature_values.arff", 'r'))
    # total feature number: 405, Pihera feature number: 287
    X = arff.load(
        open("../data/aslib_data-not_verified/TSP-ECJ2018/feature_values.arff", 'r'))
    X = X['data']
    # total feature number: 405, Pihera feature number: 287
    X = np.array([x[-287:] for x in X[0:-1:10]], dtype='f')
    X = pd.DataFrame(X)

    X_train, X_test, _, _ = train_test_split(X, y, test_size=ratio, random_state=42)
    print(X_train.index.values)
    np.savez('%strain_test_split' % out_dir,\
             train_index=X_train.index.values, test_index=X_test.index.values)


class create_labels(object):
    def __init__(self, t_max, penalize_factor, alg_num, repeat, method):
        self.t_max = t_max
        self.pel = penalize_factor
        self.alg_num = alg_num
        self.rep_run = repeat
        self.method = method

    def __call__(self):
        # directly handle *_algorithm_runs.npy
        # out_dir = '../data/TSP/runs/'
        # result_files = glob('%s*_algorithm_runs*' % out_dir)
        result_files = ['../data/TSP/runs/RUE_algorithm_runs.npy',
                        '../data/TSP/runs/explosion_algorithm_runs.npy',
                        '../data/TSP/runs/grid_algorithm_runs.npy',
                        '../data/TSP/runs/cluster_algorithm_runs.npy',
                        '../data/TSP/runs/implosion_algorithm_runs.npy',
                        '../data/TSP/runs/expansion_algorithm_runs.npy']
        w = np.zeros((0, self.rep_run, self.alg_num),
                     dtype=[('alg', 'S20'),
                            ('ins_name', 'S100'), ('runtime', 'f8'),
                            ('quality', 'f8'), ('status', 'S10')
                            ])
        for result in result_files:
            pm = np.load(open(result, 'rb'))
            w = np.vstack((w, pm))
        print('Whole instance number', w.shape[0])

        w['runtime'][w['status'] == b'TIMEOUT'] = self.t_max * self.pel

        # t(runtime of each algorithm for each ins): shape(ins_num, alg_num)
        t = np.median(np.round(w['runtime'], 2), axis=1)

        # X(features): shape(ins_num, feature_num)
        df = pd.read_csv('../data/TSP/all_feature_values.csv')
        X = np.zeros((w.shape[0], df.shape[1]-1))
        name_index_dict = dict()
        for index in range(w.shape[0]):
            if index == w.shape[0] - 1:
                print(" ")
            X[index, ] = df[df['name'] == w[index, 0, 0]['ins_name'].decode("utf-8")].values[0, 1:]
            name_index_dict[w[index, 0, 0]['ins_name'].decode("utf-8")] = index

        # train_index/test_index: shape(ins_num, )
        with open('../data/TSP/train_instance_id.txt', 'r') as f:
            train_insts = f.read().strip().split('\n')
        with open('../data/TSP/test_instance_id.txt', 'r') as f:
            test_insts = f.read().strip().split('\n')

        train_index = [name_index_dict[ins_name] for ins_name in train_insts]
        test_index = [name_index_dict[ins_name] for ins_name in test_insts]
        train_index = np.array(train_index)
        test_index = np.array(test_index)

        # z(feature cost): shape(ins_num, feature_type_num)
        df = pd.read_csv('../data/TSP/all_feature_computation_time.csv')
        z = np.zeros((w.shape[0], df.shape[1]-1))
        for index in range(w.shape[0]):
            z[index, ] = df[df['name'] == w[index, 0, 0]['ins_name'].decode("utf-8")].values[0, 1:]

        if self.method == 'regression':
            # y: shape(ins_num, alg_num)
            y = deepcopy(t)

        # labels: shape(ins_num, )
        count = 0
        labels = np.zeros(w.shape[0], dtype=int)
        for index, line in enumerate(t):
            min_value = np.min(line)
            min_L = []
            for i, v in enumerate(line):
                if v == min_value:
                    min_L.append(i)
            if len(min_L) > 1:
                count += 1
            labels[index] = rd.sample(min_L, 1)[0]

        if self.method == 'classification':
            y = deepcopy(labels)

        if self.method == 'paired-regression':
            # y: shape(ins_num, class_num*(class_num-1)/2)
            y = np.zeros((t.shape[0], int(t.shape[1]*(t.shape[1]-1)/2)))
            index = 0
            for i in range(t.shape[1]):
                for j in range(i+1, t.shape[1]):
                    y[:, index] = t[:, i] - t[:, j]
                    index += 1

        return X, y, z, t, labels, train_index, test_index

# if __name__ == '__main__':
#     cl = create_labels(900.0, 10, 6, 5)
#     cl()
