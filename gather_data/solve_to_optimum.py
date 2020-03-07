'''
solve all instances in folder "instances" with concorde
'''

import time
import subprocess
import re
import sys
import psutil
sys.path.append(sys.path[0] + '/../../../')
from path_converter import path_con

start_time = time.time()
directory = path_con('instance_set/TSP/instances/')
setSize = 300
parallism = 80
running_tasks = 0
sub_process = set()
solver_path = path_con('Solver/Concorde/concorde')

for i in range(setSize):
    if i <= setSize / 2 - 1:
        option = 'uniform'
        j = i
    else:
        option = 'clustering'
        j = i - setSize / 2
    while True:
        if running_tasks >= parallism:
            time.sleep(1)
            finished = [pid for pid in sub_process if pid.poll() is not None]
            sub_process -= set(finished)
            running_tasks = len(sub_process)
            continue
        else:
            instance = directory + option + '_' + str(j + 1)
            num = re.search(r'\d+', instance).group()
            newinstance = '%s%s.txt' % (num, option)
            cmd = 'cp %s ./%s' % (instance, newinstance)
            subprocess.check_output(cmd, shell=True)
            output_file = '%s%s_%d' % ('../optimal_solutions/', option, j + 1)

            cmd = solver_path + ' ' + newinstance + ' > ' + output_file
            sub_process.add(psutil.Popen(cmd, shell=True))
            running_tasks = len(sub_process)
            break

# check if subprocess all exits
while sub_process:
    time.sleep(20)
    print 'Still %d sub process not exits' % len(sub_process)
    finished = [pid for pid in sub_process if pid.poll() is not None]
    sub_process -= set(finished)

print 'Total consumned ' + str(time.time() - start_time)
