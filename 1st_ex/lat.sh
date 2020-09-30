#!/bin/bash

#PBS -l walltime=00:01:00,nodes=1:ppn=2
#PSB -N lat
#PBS -q batch

cd $PBS_0_WORKDIR
for i in `seq 37`; do time mpirun —hostfile $PBS_NODEFILE -np 2 ./hw1 $i; done;
