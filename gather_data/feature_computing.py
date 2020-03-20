import os
import subprocess
# Phiera features, index (0 is ins_name): 1-287, 287 in total
# for each kind of instances
#     using Phirea (on Windows) exe compute features
#     rename n.tsp -> data/%s/n.tsp % option
#     extract computation time -> save in a new file

# UBC-cheap features 288-300, 13 in total
# UBC features 288-337, 50 in total
def ubc():
    binary = 'gather_data/feature_compute/UBC/TSP-feature -all'
    flag = False
    print('ubc')
    for i, line in enumerate(feature_lines[1:3]):
        ins = line.split(',')[0]
        print(ins)
        cmd = '%s %s > tmp.txt' % (binary, ins)
        pid = subprocess.Popen(cmd, shell=True)
        pid.communicate()
        with open('tmp.txt', 'r') as ff:
            stdout_data = ff.read().strip()
        line_1, line_2 = stdout_data.split('\n')
        if not flag:
            feature_lines[0] = '%s,%s' %\
                               (feature_lines[0], line_1.strip())
            print(feature_lines[0])
            flag = True
        feature_lines[i+1] = '%s,%s' %\
                             (feature_lines[i+1], line_2.strip())
        print(feature_lines[i+1])
        values = line_2.split(',')
        ubc_cheap_time = float(values[4]) + float(values[12])
        ubc_time = ubc_cheap_time + float(values[16]) + float(values[35]) +\
                   float(values[47]) + float(values[49])
        for j, cost_line in enumerate(cost_lines[1:]):
            if cost_line.split(',')[0] == ins:
                cost_lines[j+1] = '%s,%f,%f' %\
                                  (cost_lines[j+1], ubc_cheap_time, ubc_time)
                print(cost_lines[j+1])
                break
    os.remove('tmp.txt')

# TSP meta features 338-405, 68 in total
def tsp_meta():
    cmd = 'Rscript gather_data/calculate_tspmeta.R'
    pid = subprocess.Popen(cmd, shell=True)
    pid.communicate()
    with open("tspmeta_feature.txt", 'r') as ff:
        lines = ff.read().strip().split('\n')
    feature_lines[0] = '%s,%s' % (feature_lines[0], lines[0])
    print(feature_lines[0])
    for line in lines[1:]:
        values = line.split(',')
        ins = values[0]
        tspmeta_time = float(values[-1])
        for j, feature_line in enumerate(feature_lines[1:]):
            if feature_line.split(',')[0] == ins:
                feature_lines[j+1] = '%s,%s' %\
                                     (feature_lines[j+1], ','.join(values[1:-1]))
                print(feature_lines[j+1])
                break
        for j, cost_line in enumerate(cost_lines[1:]):
            if cost_line.split(',')[0] == ins:
                cost_lines[j+1] = '%s,%f' %\
                                  (cost_lines[j+1], tspmeta_time)
                print(cost_lines[j+1])
                break
    os.remove('tspmeta_feature.txt')

if __name__ == '__main__':
    os.chdir('../')
    with open('data/TSP/feature_values.csv', 'r') as f:
        feature_lines = f.read().strip().split('\n')
    with open('data/TSP/feature_computation_time.csv', 'r') as f:
        cost_lines = f.read().strip().split('\n')
    cost_lines[0] = '%s,%s,%s,%s' %\
                    (cost_lines[0], 'UBC_cheap_time', 'UBC_time', 'TSPmeta_time')

    ubc()
    tsp_meta()

    with open('data/TSP/all_feature_values.csv', 'w+') as f:
        f.writelines(feature_lines)
    with open('data/TSP/all_feature_computation_time.csv', 'w+') as f:
        f.writelines(cost_lines)
