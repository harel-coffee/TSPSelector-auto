# Gather performance matrix
import time
import subprocess
import sys
import os
import datetime
from glob import glob
import numpy as np


def extract_result(needtest, data, insts,
                   insnum, algs, retimes,
                   dire, tmax, instype):
    for n in range(insnum):
        for nn in range(retimes):
            for nnn, tu in enumerate(algs):
                file = '%s%s_%s_%s_rep%d' %\
                       (dire, instype, insts[n].split('/')[-1],
                        tu[0], nn)
                if os.path.isfile(file):
                    with open(file, 'r') as fp:
                        lines = fp.read().strip().split('\n')
                        flag = False
                        for line in lines:
                            if 'Result for' not in line:
                                continue
                            flag = True
                            values = line[line.find(
                                ':') + 1:].strip().replace(' ', '').split(',')
                            (status, runtime, _, _, rngseed) = values[0:5]
                            runtime = float(runtime)
                            rngseed = int(rngseed)
                            if status == 'CRASHED':
                                needtest.append((n, nn, nnn))
                                continue
                            if status in ['TIMEOUT', 'Unsuccessful']:
                                runtime = tmax
                            data[n, nn, nnn] = (tu[0], insts[n], runtime, 0, status)
                        if not flag:
                            print("Not found result for SMAC/PARAMILS in %s" % (file))
                            needtest.append((n, nn, nnn))
                            continue
                else:
                    print("Not found file %s" % (file))
                    needtest.append((n, nn, nnn))

def summary():
    out_dir = 'data/TSP/runs/'
    result_files = glob('%s*_algorithm_runs*' % out_dir)
    f = open('data/TSP/algorithm_runs.csv', 'w+')
    f.write('ins_name,alg_name,repeat,status,runtime\n')
    for result_file in result_files:
        pm = np.load(open(result_file, 'rb'))
        ins_size, repeat_times, algs_size = pm.shape
        for ins_index in range(ins_size):
            for alg_index in range(algs_size):
                for repeat_index in range(repeat_times):
                    v = pm[ins_index, repeat_index, alg_index]
                    f.write('%s,%s,%d,%s,%.2f\n' % (v['ins_name'].decode("utf-8"),
                                                    v['alg'].decode("utf-8"),
                                                    repeat_index,
                                                    v['status'].decode("utf-8"),
                                                    round(v['runtime'], 2)))

    f.close()


if __name__ == '__main__':
    # python algorithm_runs.py paralism option
    # for each algorithm, we run it on each instance for three times
    os.chdir('..')
    maxParalism = int(sys.argv[1])
    if sys.argv[2] == "additional":
        options = sys.argv[3:]
    else:
        options = sys.argv[2:]

    seeds = [0, 42, 64, 128, 1024]
    repeat = len(seeds)
    cutoff_time = 900
    if sys.argv[2] == "additional":
        result_dir = 'data/TSP/additional/runs/'
    else:
        result_dir = 'data/TSP/runs/'

    # instances = []
    # instances.extend(glob('data/TSP/RUE/*'))
    # instances.extend(glob('data/TSP/cl/*'))
    # instances.extend(glob('data/TSP/explosion/*'))
    # instances.extend(glob('data/TSP/implosion/*'))
    # instances.extend(glob('data/TSP/cluster/*'))
    # instances.extend(glob('data/TSP/compression/*'))
    # instances.extend(glob('data/TSP/rotation/*'))
    # instances.extend(glob('data/TSP/expansion/*'))
    # instances.extend(glob('data/TSP/linearprojection/*'))
    # instances.extend(glob('data/TSP/grid/*'))
    for option in options:
        output_dir = '%s%s/' % (result_dir, option)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        if sys.argv[2] == "additional":
            instances = glob('data/TSP/additional/%s/*' % option)
            optima = "data/TSP/additional/optimum.json"
        else:
            instances = glob('data/TSP/%s/*' % option)
            optima = "data/TSP/optimum.json"
        ins_num = len(instances)

        algos = []
        algos.append(("LKH", "python -u solver/LKH/wrapper.py --mem-limit 2048"
                             " --solutionity %s " % optima))
        algos.append(("LKH-restart", "python -u solver/LKH-restart/wrapper.py --mem-limit 2048"
                                     " --solutionity %s " % optima))
        algos.append(("LKH-crossover", "python -u solver/LKH-crossover/wrapper.py --mem-limit 2048"
                                       " --solutionity %s " % optima))
        algos.append(("GA-EAX", "python -u solver/GA-EAX/wrapper.py --mem-limit 2048"
                                " --solutionity %s " % optima))
        algos.append(("GA-EAX-restart", "python -u solver/GA-EAX-restart/wrapper.py "
                                        "--mem-limit 2048"
                                        " --solutionity %s " % optima))
        # memory control of MAOS is done by jvm
        algos.append(("MAOS", "python -u solver/MAOS-TSP/wrapper.py "
                              " --solutionity %s " % optima))
        alg_num = len(algos)

        algorithm_runs = np.zeros((ins_num, repeat, alg_num),
                                  dtype=[('alg', 'S20'),
                                         ('ins_name', 'S100'), ('runtime', 'f8'),
                                         ('quality', 'f8'), ('status', 'S10')
                                        ])

        runningTask = 0
        processSet = set()
        for i in range(ins_num):
            print('------------------Testing instance %d---------------' % i)
            for k in range(repeat):
                seed = seeds[k]
                for t in algos:
                    while True:
                        if runningTask >= maxParalism:
                            time.sleep(0.1)
                            finished = [
                                pid for pid in processSet if pid.poll() is not None
                            ]
                            processSet -= set(finished)
                            runningTask = len(processSet)
                            continue
                        else:
                            instance = instances[i]
                            output_file = '%s%s_%s_%s_rep%d' %\
                                        (output_dir, option, instance.split('/')[-1],\
                                        t[0], k)
                            cmd = ("%s %s %d %.1f %d %d > %s" %\
                                    (t[1], instance, 0, cutoff_time, 0, seed, output_file))
                            print(cmd)
                            processSet.add(subprocess.Popen(cmd, shell=True))
                            runningTask = len(processSet)
                            break

        # check if subprocess all exits
        while processSet:
            time.sleep(5)
            print('Still %d sub process not exits' % len(processSet))
            finished = [pid for pid in processSet if pid.poll() is not None]
            processSet -= set(finished)

        need_test_ones = []
        extract_result(need_test_ones, algorithm_runs, instances,
                       ins_num, algos, repeat, output_dir, cutoff_time, option)

        re_test = 1
        while need_test_ones:
            print("Repeat %d test, %d ones no result in there\n" %
                  (re_test, len(need_test_ones)))
            runningTask = 0
            processSet = set()
            total = len(need_test_ones)
            for index, one in enumerate(need_test_ones):
                while True:
                    if runningTask >= maxParalism:
                        time.sleep(0.1)
                        finished = [
                            pid for pid in processSet if pid.poll() is not None
                        ]
                        processSet -= set(finished)
                        runningTask = len(processSet)
                        continue
                    else:
                        seed = seeds[one[1]]
                        output_file = '%s%s_%s_%s_rep%d' %\
                                    (output_dir, option, instances[one[0]].split('/')[-1],\
                                    algos[one[2]][0], one[1])
                        cmd = ("%s %s %d %.1f %d %d > %s" %\
                               (algos[one[2]][1], instances[one[0]], 0, cutoff_time, 0,
                                seed, output_file))
                        print(("Repeat: %d, %d/%d, cmd:\n" % (re_test, index+1, total)) + cmd)
                        processSet.add(subprocess.Popen(cmd, shell=True))
                        runningTask = len(processSet)
                        break

            # check if subprocess all exits
            while processSet:
                time.sleep(5)
                print('Still %d sub process not exits' % len(processSet))
                finished = [pid for pid in processSet if pid.poll() is not None]
                processSet -= set(finished)

            need_test_ones = []
            extract_result(need_test_ones, algorithm_runs, instances,
                           ins_num, algos, repeat, output_dir, cutoff_time, option)
            re_test += 1

        np.save(result_dir + '%s_algorithm_runs' % option, algorithm_runs)
        print(datetime.datetime.now())
