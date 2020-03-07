'''
solve all instances with concorde
'''
import sys
import time
import subprocess

def solve(option):
    directory = '../data/TSP/%s/' % option
    setSize = 1000

    parallism = 80
    running_tasks = 0
    sub_process = set()
    solver_path = '../../ACPP/Solver/Concorde/concorde'
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
        print 'Still %d sub process not exits' % len(sub_process)
        finished = [pid for pid in sub_process if pid.poll() is not None]
        sub_process -= set(finished)

    print 'Total consumned ' + str(time.time() - start_time)

def extract():
    pass

if __name__ == '__main__':
    if sys.argv[1] == 'solve':
        solve(sys.argv[2])
    if sys.argv[1] == 'extract':
        extract()
