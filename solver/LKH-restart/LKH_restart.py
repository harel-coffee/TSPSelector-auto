# restart LKH it exits with 42
# with different seeds
# input: paramfile
# output is the same as LKH
# LKH exit code: 0: success, signum: terminated, 42: restart
import sys
import os
import signal
import random
from subprocess import Popen

pid = 0

def signal_handler(sig, _):
    if not pid.poll():
        pid.terminate()
    sys.exit(sig)

def call_LKH(argv):
    count = 0
    while True:
        real_dir = os.path.dirname(os.path.realpath(__file__))
        if count == 0:
            cmd = '%s %s %s' % (real_dir+'/LKH', argv[1], argv[2])
        else:
            cmd = '%s %s %d' % (real_dir+'/LKH', argv[1], random.randint(0, 100))
        global pid
        pid = Popen(cmd, shell=True, stdout=sys.stdout)
        signum = pid.wait()
        if signum != 42:
            break
        print("restart LKH!\n")
        count += 1


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if len(sys.argv) != 2:
        print ("./LKH_restart.py paramfile\n")
        sys.exit(-1)
    call_LKH(sys.argv)
