import os
import subprocess
import json
import numpy as np

option = 'expansion'

os.chdir('../')
with open('data/TSP/runs/%s_algorithm_runs.npy' % option, 'rb') as f:
    ex = np.load(f)
with open('data/TSP/additional/runs/%s_algorithm_runs.npy' % option, 'rb') as f:
    ex_add = np.load(f)

ex['runtime'][ex['status'] == b'TIMEOUT'] = 9000.0
ex_add['runtime'][ex_add['status'] == b'TIMEOUT'] = 9000.0

ex_median = np.median(ex['runtime'], axis=1)
ex_add_median = np.median(ex_add['runtime'], axis=1)

ex_diff = ex_median[:, 4] - ex_median[:, 5]
ex_diff_argsort = np.argsort(ex_diff)
ex_add_diff = ex_add_median[:, 4] - ex_add_median[:, 5]
ex_add_diff_argsort = np.argsort(ex_add_diff)

rep_num = 200
start = 10
with open('data/TSP/runs/%s_algorithm_runs.npy' % option, 'rb') as f:
    ex_new = np.load(f)

for i in range(1, rep_num):
    index_1 = ex_diff_argsort[i-1+start]
    index_2 = ex_add_diff_argsort[-i]
    ex_new[index_1, :, :] = ex_add[index_2, :, :]

# insert GA-EAX-restart timeout
# results = np.argwhere(ex_add_median[:, 4] == 9000.0)
# results = results.reshape(results.shape[0])
# for index_2 in results:
#     if index_2 in ex_add_diff_argsort[-rep_num:]:
#         continue
#     index_1 = ex_diff_argsort[rep_num-1+start]
#     ex_new[index_1, :, :] = ex_add[index_2, :, :]
#     rep_num += 1

ex_new['runtime'][ex_new['status'] == b'TIMEOUT'] = 9000.0
ex_new_median = np.median(ex_new['runtime'], axis=1)
count = 0
for i in range(ex_new_median.shape[0]):
    if np.min(ex_new_median[i, :]) == 9000.0:
        index_2 = ex_add_diff_argsort[-rep_num-count-1]
        ex_new[i, :, :] = ex_add[index_2, :, :]
        count += 1
print("%d instances, vbs with timeout\n" % count)
ex_new['runtime'][ex_new['status'] == b'TIMEOUT'] = 900.0

with open('data/TSP/optimum.json', 'r') as f:
    old_opt = json.load(f)
with open('data/TSP/additional/optimum.json', 'r') as f:
    new_opt = json.load(f)

# update instances, optimum
for i in range(ex_new.shape[0]):
    if b'additional' in ex_new[i, 0, 0]['ins_name']:
        # change instance file
        new_file = ex_new[i, 0, 0]['ins_name']
        old_file = ex[i, 0, 0]['ins_name']
        cmd = 'cp %s %s' % (new_file.decode('utf-8'),\
                            old_file.decode('utf-8'))
        pid = subprocess.Popen(cmd, shell=True)
        pid.communicate()

        # change ins_name
        ex_new['ins_name'][i, :, :] = old_file

        # change optimum file
        old_opt[old_file.decode('utf-8')] = new_opt[new_file.decode('utf-8')]

np.save('new_algorithm_runs', ex_new)
with open('new_optimum.json', 'w+') as f:
    json.dump(old_opt, f)
