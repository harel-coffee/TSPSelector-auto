# pre-prossess data
# data structures: instance_name, feature, label
import pandas as pd
import arff
import numpy as np
from sklearn.model_selection import train_test_split


class create_labels(object):
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


def split(ratio=0.2, out_dir="../data/aslib_data-not_verified/TSP-ECJ2018/"):
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
