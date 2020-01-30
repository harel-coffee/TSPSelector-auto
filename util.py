import os
import pickle
import tsplib95
import numpy as np
from math import pi, sin, cos
import scipy.sparse as sp
from sklearn.decomposition import PCA
from scipy import sparse
from hilbertcurve.hilbertcurve import HilbertCurve

from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.spatial.distance import euclidean
pathes = [
        #   '/home/kfzhao/data/ECJ_instances/national'
        #  ,'/home/kfzhao/data/ECJ_instances/rue'
          '/home/kfzhao/data/ECJ_instances/tsplib'
        # , '/home/kfzhao/data/ECJ_instances/vlsi'
        ]

filename = '/home/kfzhao/data/ECJ_instances/tsplib/att532.tsp'
def load_tsp_instance(filename):
    problem = tsplib95.load_problem(filename)
    N = problem.dimension

    print("problem size:", N)
    # skip large graph with >10000 nodes
    #if N > 10000 or problem.edge_weight_type != 'EUC_2D':
    #    return

    """
    G = problem.get_graph()
#    print(G.number_of_nodes())
    #print(G.nodes)
#    print(G.number_of_edges())
    #print(G.edges)
    """

    mat = np.zeros(shape=(N, N), dtype= np.float32)
    x = np.zeros(shape=(N, 2), dtype= np.float32) # node coordinate


    for i in range(N):
        x[i][0], x[i][1] = problem.node_coords[i + 1][0], problem.node_coords[i + 1][1]
    x = coordinate_normalize(x)
    for i in range(N):
        for j in range(i + 1, N):
            #mat[i, j] = mat[j, i] = problem.wfunc(i + 1, j + 1)
            mat[i, j] = mat[j, i] = euclidean(x[i], x[j])

    A = sp.csr_matrix(mat)
    edge_index, edge_attr = from_scipy_sparse_matrix(A)
    edge_index, edge_attr = edge_index.numpy(), edge_attr.numpy()

    instance = {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr,  'adj': A.todense()}
    with open(os.path.splitext(filename)[0] + '_norm.pickle', 'wb') as out_file:
        pickle.dump(obj = instance, file = out_file, protocol= 3)
        print(filename + " saved.")
        out_file.close()


def coordinate_normalize(x):
    # normalize the tsp coordinate
    x_min, x_max = x[:,0].min(), x[:,0].max()
    y_min, y_max = x[:,1].min(), x[:,1].max()
    x[:,0], x[:,1] = x[:,0] - x_min, x[:,1] - y_min
    scale = max(x_max - x_min, y_max - y_min)
    x = x / scale
    return x


def matrix_reorder(filename):
    in_file = open(os.path.splitext(filename)[0] + '_norm.pickle', 'rb')
    data = pickle.load(in_file)
    x = data['x']
    A = data['adj']
    N = A.shape[0]
    pca = PCA(n_components=1)
    x = pca.fit_transform(x).squeeze()
    temp = x.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(x))
    for i in range(N):
        A[:, i] = A[ranks, i]
    for i in range(N):
        A[i, :] = A[i, ranks]
    print(A.shape)


def hilbert_matrix_reorder(filename):
    in_file = open(os.path.splitext(filename)[0] + '_norm.pickle', 'rb')
    data = pickle.load(in_file)
    x = data['x']
    A = data['adj']
    N = A.shape[0]
    x_norm = x * 1000
    x_norm = x_norm.astype(int)
    hilbert_curve = HilbertCurve(10,  2)
    temp = np.zeros(shape=( N ))
    for i in range( N ):
        temp[i] = hilbert_curve.distance_from_coordinates([x_norm[i][0], x_norm[i][1]])

    temp = temp.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(temp))
    for i in range(N):
        A[:, i] = A[ranks, i]
    for i in range(N):
        A[i, :] = A[i, ranks]
    print(A.shape)

def savetoTSPImage(filename, num_grid = 256):
    in_file = open(os.path.splitext(filename)[0] + '_norm.pickle', 'rb')
    data = pickle.load(in_file)
    x = data['x']
    image = coordinate_to_grid_image(x = x, num_grid= num_grid)
    for direction in range(4):
        image = np.rot90(image)
        instance = {'adj': image}
        out_file_dir = os.path.splitext(filename)[0] + '_{0}_{1}_image.pickle'.format(num_grid, direction)
        with open(out_file_dir, 'wb') as out_file:
            pickle.dump(obj=instance, file=out_file, protocol=3)
            out_file.close()

def tsp_image_rotate_and_flip(filename, num_grid = 256):
    in_file = open(os.path.splitext(filename)[0] + '_norm.pickle', 'rb')
    data = pickle.load(in_file)
    x = data['x']
    image = coordinate_to_grid_image(x = x, num_grid= num_grid)

    v_image = sparse.csr_matrix(np.flip(image, axis= 0))
    instance = {'adj': v_image}
    out_file_dir = os.path.splitext(filename)[0] + '_{0}_flip0_image.pickle'.format(num_grid)
    with open(out_file_dir, 'wb') as out_file:
        pickle.dump(obj=instance, file=out_file, protocol=3)
        out_file.close()

    h_image = sparse.csr_matrix(np.flip(image, axis=1))
    instance = {'adj': h_image}
    out_file_dir = os.path.splitext(filename)[0] + '_{0}_flip1_image.pickle'.format(num_grid)
    with open(out_file_dir, 'wb') as out_file:
        pickle.dump(obj=instance, file=out_file, protocol=3)
        out_file.close()

    '''
    for direction in range(4):
        image = np.rot90(image)
        instance = {'adj': np.flip(image, axis=0)}
        out_file_dir = os.path.splitext(filename)[0] + '_{0}_{1}_flip0_image.pickle'.format(num_grid, direction)
        with open(out_file_dir, 'wb') as out_file:
            pickle.dump(obj=instance, file=out_file, protocol=3)
            out_file.close()

        instance = {'adj': np.flip(image, axis=1)}
        out_file_dir = os.path.splitext(filename)[0] + '_{0}_{1}_flip1_image.pickle'.format(num_grid, direction)
        with open(out_file_dir, 'wb') as out_file:
            pickle.dump(obj=instance, file=out_file, protocol=3)
            out_file.close()
    '''


def coordinate_rotate(x, angle):
    new_x = np.zeros(shape= x.shape)
    angle = float(angle) / 180.0 * pi
    new_x[:,0] = x[:,0] * cos(angle) - x[:,1] * sin(angle)
    new_x[:, 1] = x[:, 0] * sin(angle) + x[:, 1] * cos(angle)
    return new_x

def coordinate_to_grid_image(x, num_grid = 256):
    image = np.zeros((num_grid, num_grid), dtype=np.float32)
    for i in range(x.shape[0]):
        idx_x = int(x[i][0] * num_grid) - 1
        idx_y = int(x[i][1] * num_grid) - 1
        image[idx_x][idx_y] = image[idx_x][idx_y] + 1.0
    return image


def tsp_image_rotate(filename, num_grid = 256):
    in_file = open(os.path.splitext(filename)[0] + '.pickle', 'rb')
    data = pickle.load(in_file)
    x = data['x']

    for angle in [45, 90, 135, 180, 225, 270, 315, 360]:
        new_x = coordinate_rotate(x, angle)
        # rotate on the original coordinate and normalize
        new_x = coordinate_normalize(new_x)
        image = coordinate_to_grid_image(x = new_x, num_grid=num_grid)
        image = sparse.csr_matrix(image)
        instance = {'adj': image}
        out_file_dir = os.path.splitext(filename)[0] + '_{0}_{1}_image.pickle'.format(num_grid, angle)
        with open(out_file_dir, 'wb') as out_file:
            pickle.dump(obj=instance, file=out_file, protocol=3)
            out_file.close()



def test():
    instances_num = 0
    for path in pathes:
        filelist = os.listdir(path)
        for file in filelist:
            # if file is a directory, skip
            print("process file:" + file)
            if os.path.isdir(os.path.join(path, file)):
                continue
            if os.path.splitext(os.path.join(path, file))[1] == '.tsp':
                load_tsp_instance(os.path.join(path, file))
                instances_num = instances_num + 1

    return instances_num


def input_test(path = '/home/kfzhao/data/ECJ_instances/'):
    labels = load_labels()
    file_list = []
    for key in labels.keys():
        dataset = key.strip().split('_')[0]
        instance_id = key.strip().split('_')[1]
        if dataset == 'morphed':
            node_num = instance_id.strip().split('-')[0]
            tmp1, tmp2 = instance_id.strip().split('---')[0], instance_id.strip().split('---')[1]
            instance_id = node_num + "---" + tmp1 + '.tsp---' + tmp2 + '.tsp'
        full_instance_dir = os.path.join(path, dataset, instance_id) + '.pickle'
        file_list.append((key, full_instance_dir))

    max_size = 0
    for key, full_instance_dir in file_list:
        try:
            """
            with open(full_instance_dir, 'rb') as in_file:

                data = pickle.load(in_file)
                x = data['x']
                max_size = max(x.shape[0], max_size)
                #max_val, min_val = x.max(), x.min()
                #print('coo range: {}, {}'.format(min_val, max_val))
                
                in_file.close()
            """
            #load_tsp_instance(full_instance_dir)
            #savetoTSPImage(full_instance_dir)
            tsp_image_rotate_and_flip(full_instance_dir)
            print("proceed: " + full_instance_dir)
        except IOError:
            print("cannot open: " + full_instance_dir)





def process_one_instance(data):
    instance_id = data[0][0]
    best_runtime = 36000
    best_algorithm = 'all'
    algorithm_to_result = {}
    for i in range(50):
        ins_id, repeat, algorithm, runtime, runstatus = \
        data[i][0], data[i][1], data[i][2], data[i][3], data[4]
        if ins_id != instance_id:
            return None, None
        if algorithm not in algorithm_to_result.keys():
            algorithm_to_result[algorithm] = list([runtime])
        else:
            algorithm_to_result[algorithm].append(runtime)

    for key, value in algorithm_to_result.items():
        value.sort()
        if value[5] < 3600:
            if (value[4] + value[5]) / 2.0 < best_runtime:
                best_runtime = (value[4] + value[5]) / 2.0 # median of the test performance
                best_algorithm = key
    return instance_id, best_algorithm

def load_labels(filename = '/home/kfzhao/data/ECJ_instances/algorithm_runs.arff.txt'):

    file = open(filename, 'r')
    data = []
    labels = {}
    for line in file:
        if line.strip().startswith('@') or line.strip() == '':
            continue
        line = line.strip().split(',')
        ins_id, repeat, algorithm, runtime, runstatus = \
        str(line[0]), int(line[1]), str(line[2]), float(line[3]), str(line[4])
        data.append((ins_id, repeat, algorithm, runtime, runstatus))
    file.close()
    print("num of record:", len(data))
    for i in range(int(len(data) / 50)):
        instance_id, best_algorithm = process_one_instance(data[i * 50: i * 50 + 50])
        #print(instance_id, best_algorithm)
        if instance_id is not None:
            labels[instance_id] = best_algorithm
    print("num of labels:", len(labels))
    return labels

if __name__ == "__main__":
    print("start")
    #load_tsp_instance(filename)
    #tsp_image_rotate(filename)

    input_test()
    #matrix_reorder(filename)
    print("haha")