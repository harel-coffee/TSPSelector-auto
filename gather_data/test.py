# test multiple runs of each solver wrapper
import os
import sys
import time
import glob
import subprocess

class testing(object):
    def __init__(self, algo):
        self.algos = {"LKH": "python -u solver/LKH/wrapper.py --mem-limit 1024"
                             " --solutionity data/TSP/optimum.json ",
                      "LKH-restart": "python -u solver/LKH-restart/wrapper.py --mem-limit 1024"
                                     " --solutionity data/TSP/optimum.json ",
                      "LKH-crossover": "python -u solver/LKH-crossover/wrapper.py --mem-limit 1024"
                                       " --solutionity data/TSP/optimum.json ",
                      "GA-EAX": "python -u solver/GA-EAX/wrapper.py --mem-limit 1024"
                                " --solutionity data/TSP/optimum.json ",
                      "GA-EAX-restart": "python -u solver/GA-EAX-restart/wrapper.py "
                                        "--mem-limit 1024 --solutionity data/TSP/optimum.json ",
                      "MAOS": "python -u solver/MAOS-TSP/wrapper.py"
                              " --solutionity data/TSP/optimum.json "}
        self.algo = algo
        self.cmd = self.algos[algo]

    def run(self, tmax):
        os.chdir('..')
        insts = glob.glob('data/TSP/*/1.tsp')
        seeds = [0, 42]
        processSet = set()
        outdir = 'data/TSP/runs/'
        for instance in insts:
            for i, seed in enumerate(seeds):
                option = instance.split('/')[-2]
                output_file = '%s%s_%s_%s_rep%d' %\
                            (outdir, option, instance.split('/')[-1], self.algo, i)
                cmd = ("%s %s %d %.1f %d %d > %s" %\
                    (self.cmd, instance, 0, tmax, 0, seed, output_file))
                processSet.add(subprocess.Popen(cmd, shell=True))

        # check if subprocess all exits
        while processSet:
            time.sleep(5)
            print('Still %d sub process not exits' % len(processSet))
            finished = [pid for pid in processSet if pid.poll() is not None]
            processSet -= set(finished)

if __name__ == "__main__":
    test = testing(sys.argv[1])
    test.run(int(sys.argv[2]))
