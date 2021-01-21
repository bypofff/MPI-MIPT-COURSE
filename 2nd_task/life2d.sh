#!/bin/bash
#PBS -l walltime=00:10:00,nodes=7:ppn=4
#PBS -N task4_job
cd $PBS_O_WORKDIR
mpirun --hostfile $PBS_NODEFILE -np 28 life2d p46gun.cfg
hostname