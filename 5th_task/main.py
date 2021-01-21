import os
import time
import numpy as np


os.system('rm job.sh')
os.system('rm glider_10x10.cfg')

N = input()

with open("glider_10x10.cfg", "w") as file_1:
	file_1.write("1000")
	file_1.write("\n1")
	file_1.write("\n" + str(int(np.sqrt(N) * 1000)) + " " + str(int(np.sqrt(N) * 1000)))
	file_1.write("\n0 2")
	file_1.write("\n1 0")
	file_1.write("\n1 2")
	file_1.write("\n2 1")
	file_1.write("\n2 2")

with open("job.sh", "w") as file_1:
    	file_1.write("#!/bin/bash")
    	file_1.write("\n#PBS -l walltime=00:10:00,nodes=7:ppn=4")
    	file_1.write("\n#PBS -N task3_job")
    	file_1.write("\ncd $PBS_O_WORKDIR")
    	file_1.write("\nmpirun --hostfile $PBS_NODEFILE -np " + str(N) + " life2d glider_10x10.cfg")
    	file_1.write("\nhostname")

os.system('chmod +x job.sh')
os.system('qsub job.sh')   
print(str(N) + " processes are in calculating")
print

