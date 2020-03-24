'''
solve all instances with concorde
'''
import os
import sys
import time
import subprocess
import json

def solve(option, parallism):
    directory = '../data/TSP/%s/' % option
    setSize = 1000

    running_tasks = 0
    sub_process = set()
    solver_path = '../solver/Concorde/concorde -x'
    start_time = time.time()

    for i in range(setSize):
        while True:
            if running_tasks >= parallism:
                time.sleep(1)
                finished = [pid for pid in sub_process if pid.poll() is not None]
                sub_process -= set(finished)
                running_tasks = len(sub_process)
                continue
            else:
                instance = '%s%d.tsp' % (directory, i+1)
                output_file = '%s%s_%d' % ('../data/TSP/optimal_solutions/', option, i+1)
                cmd = solver_path + ' ' + instance + ' > ' + output_file
                sub_process.add(subprocess.Popen(cmd, shell=True))
                running_tasks = len(sub_process)
                break

    # check if subprocess all exits
    while sub_process:
        time.sleep(20)
        print('Still %d sub process not exits' % len(sub_process))
        finished = [pid for pid in sub_process if pid.poll() is not None]
        sub_process -= set(finished)

    print('Total consumned ' + str(time.time() - start_time))


def solve_additional(parallism, option):
    directory = '../data/TSP/additional/%s/' % option
    setSize = 5000

    running_tasks = 0
    sub_process = set()
    solver_path = '../../ACPP/Solver/Concorde/concorde -x'
    start_time = time.time()

    for i in range(setSize):
        while True:
            if running_tasks >= parallism:
                time.sleep(1)
                finished = [pid for pid in sub_process if pid.poll()
                            is not None]
                sub_process -= set(finished)
                running_tasks = len(sub_process)
                continue
            else:
                instance = '%s%d.tsp' % (directory, i+1)
                output_file = '%s%s_%d' % (
                    '../data/TSP/additional/optimal_solutions/', option, i+1)
                cmd = solver_path + ' ' + instance + ' > ' + output_file
                sub_process.add(subprocess.Popen(cmd, shell=True))
                running_tasks = len(sub_process)
                break

    # check if subprocess all exits
    while sub_process:
        time.sleep(20)
        print('Still %d sub process not exits' % len(sub_process))
        finished = [pid for pid in sub_process if pid.poll() is not None]
        sub_process -= set(finished)

    print('Total consumned ' + str(time.time() - start_time))


def extract_additional():
    setSize = 5000
    optimumFile = '../data/TSP/additional/optimum.json'
    options = ["expansion", "grid", "linearprojection"]

    optimum_f = open(optimumFile, 'w+')
    optimum = dict()

    for option in options:
        instancePath = 'data/TSP/additional/%s/' % option
        for i in range(setSize):
            solutionFile = '%s%s_%d' % (
                '../data/TSP/additional/optimal_solutions/', option, i+1)
            insFile = '%s%d.tsp' % (instancePath, i+1)
            if not os.path.isfile(solutionFile):
                continue
            with open(solutionFile, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'Optimal Solution' in line:
                        solution = line[line.find(':')+1:].strip()
                        optimum[insFile] = solution
                        break
    json.dump(optimum, optimum_f)
    optimum_f.close()

def extract():
    setSize = 1000
    optimumFile = '../data/TSP/optimum.json'
    options = ["RUE", "cl", "explosion", "implosion", "cluster",
               "compression", "expansion", "grid", "linearprojection",
               "rotation"]

    optimum_f = open(optimumFile, 'w+')
    optimum = dict()

    for option in options:
        instancePath = 'data/TSP/%s/' % option
        for i in range(setSize):
            solutionFile = '%s%s_%d' % ('../data/TSP/optimal_solutions/', option, i+1)
            insFile = '%s%d.tsp' % (instancePath, i+1)
            if not os.path.isfile(solutionFile):
                continue
            with open(solutionFile, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'Optimal Solution' in line:
                        solution = line[line.find(':')+1:].strip()
                        optimum[insFile] = solution
                        break
    json.dump(optimum, optimum_f)
    optimum_f.close()

if __name__ == '__main__':
    # options: [RUE, cl, explosion, implosion .cluster,compression,expansion,grid,linearprojection
    # rotation]

    if sys.argv[1] == 'solve':
        solve(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == 'extract':
        extract()
    elif sys.argv[1] == 'additional_solve':
        solve_additional(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == 'additional_extract':
        extract_additional()
