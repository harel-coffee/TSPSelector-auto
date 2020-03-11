import json
import re
import os
from tempfile import NamedTemporaryFile

from genericWrapper4AC.generic_wrapper import AbstractWrapper


class MAOSWrapper(AbstractWrapper):
    '''
        Simple wrapper for LKHWrapper
    '''

    def __init__(self):
        AbstractWrapper.__init__(self)
        self.__script_dir = os.path.abspath(os.path.split(__file__)[0])
        self.parser.add_argument("--solutionity",
                                 dest="obj_file",
                                 default=None,
                                 help="optimal solutions")

    def get_command_line_args(self, runargs, _):
        '''
        Returns the command line call string to execute the target algorithm
        Args:
            runargs: a map of several optional arguments for the execution of the target algorithm.
                    {
                      "instance": <instance>,
                      "specifics" : <extra data associated with the instance>,
                      "cutoff" : <runtime cutoff>,
                      "runlength" : <runlength cutoff>,
                      "seed" : <seed>
                    }
            config: a mapping from parameter name to parameter value
        Returns:
            A command call list to execute the target algorithm.
        '''
        os.chdir("solver/MAOS-TSP/myprojects")
        binary = "java -server -Xmx1024M -cp ../release/MAOS_SEQ.jar maosKernel.MAOSExecuter"
        # TSP:Problem=rl1323 N=300 T=500 Tcon=100 DUP_TIMES=100"
        optimumFile = self.args.obj_file
        with open(optimumFile, 'r') as f:
            optimum = json.load(f)
        values = runargs["instance"].split('/')
        instance = "%s_%s" % (values[2], values[3])
        cmd = '%s TSP:Problem=%s N=300 T=500 Tcon=100 DUP_TIMES=1000000 opt=%f' %\
              (binary, instance, optimum[runargs["instance"]])
        return cmd

    def process_results(self, filepointer, _):
        '''
        Parse a results file to extract the run's status (SUCCESS/CRASHED/etc) and other optional results.

        Args:
            filepointer: a pointer to the file containing the solver execution standard out.
            out_args : a map with {"exit_code" : exit code of target algorithm}
        Returns:
            A map containing the standard AClib run results. The current standard result map as of AClib 2.06 is:
            {
                "status" : <"SAT"/"UNSAT"/"TIMEOUT"/"CRASHED"/"ABORT">,
                "runtime" : <runtime of target algrithm>,
                "quality" : <a domain specific measure of the quality of the solution [optional]>,
                "misc" : <a (comma-less) string that will be associated with the run [optional]>
            }
            ATTENTION: The return values will overwrite the measured results of the runsolver (if runsolver was used). 
        '''
        resultMap = {}
        filepointer.seek(0)
        data = str(filepointer.read(), 'utf-8')
        # print("this is outdata\n" + data)
        lines = data.strip().split('\n')
        resultMap["status"] = "TIMEOUT"
        resultMap["quality"] = -1
        for line in lines:
            line = line.strip()
            if line.startswith('Have hit the optimum') or\
                    line.startswith('Has hit the optimum'):
                resultMap["status"] = "SUCCESS"
                resultMap["quality"] = 1
                break
            if line.startswith('Successful'):
                resultMap["status"] = "SUCCESS"
                resultMap["quality"] = 1
                break
            if line.startswith('Unsuccessful'):
                resultMap['status'] = "TIMEOUT"
                resultMap["quality"] = 1
                break
            if line.startswith('Successes/Runs'):
                rrr = re.search(r'\d+', line)
                if rrr:
                    if int(rrr.group()) > 0:
                        resultMap["status"] = "SUCCESS"
                        resultMap["quality"] = 1
                        break
        self.tmpParamFile.close()
        return resultMap


if __name__ == "__main__":
    wrapper = MAOSWrapper()
    wrapper.main()
