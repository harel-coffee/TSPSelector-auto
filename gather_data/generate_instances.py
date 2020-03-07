import random as rd
import subprocess


if __name__ == '__main__':
    rd.seed(42)
    # generate rue instances
    rue_generator = '../../../projects/ACPP/instance_set/TSP/generator/portgen'
    num_rue = 1000
    instanceRecordingFile = "repeat_generation"
    f = open(instanceRecordingFile, 'w+')
    f.write('RUE Instances \n')
    for i in range(num_rue):
        city_num = rd.randint(500, 2000)
        seed = rd.randint(1, 10000000)
        ins = '%d.tsp' % (i + 1)
        cmd = '%s %d %d > ../data/TSP/RUE/%s' %\
              (rue_generator, city_num, seed, ins)
        pid = subprocess.Popen(cmd, shell=True)
        pid.wait()
        f.write('%s\n' % cmd)

    # generate netgen instances (centers from 4-8, 200)
    num_ngen_each = 200
    num_c_l = 4
    num_c_u = 8
    f.write('Netgen Instances \n')
    seed = rd.randint(1, 1000000)
    # Rscript call_netgen.R point_lower point_upper clu.lower clu.upper ins_num seed
    cmd = 'Rscript call_netgen.R %d %d %d %d %d %d' %\
            (500, 2000, num_c_l, num_c_u, num_ngen_each, seed)
    pid = subprocess.Popen(cmd, shell=True)
    pid.wait()
    f.write('%s\n' % cmd)

    # generate mutation instances
    opts = ["explosion", "implosion", "cluster",
            "compression", "expansion", "grid",
            "linearprojection", "rotation"]
    num_mutation = 1000
    for operator in opts:
        f.write('%s Instances \n' % operator)
        seed = rd.randint(1, 1000000)
        # Rscript call_tspgen.R operator point_lower point_upper ins_num seed
        cmd = 'Rscript call_tspgen.R %s %d %d %d %d' %\
              (operator, 500, 2000, num_mutation, seed)
        pid = subprocess.Popen(cmd, shell=True)
        pid.wait()
        f.write('%s\n' % cmd)
    f.close()
