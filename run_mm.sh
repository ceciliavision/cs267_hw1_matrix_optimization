#!/bin/bash -l 
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:10:00 
#SBATCH -J dgemm_job 
#SBATCH -o MydgemmOutput.%j.out

echo "TESTING with: Different Algorithms"
echo "2.4 G frequency, blocked + vec:"
srun -n 1 -N 1 --cpu-freq=1000000 ./benchmark-blocked-vec
echo "2.4 G frequency, blocked + unroll:"
srun -n 1 -N 1 --cpu-freq=1000000 ./benchmark-blocked
echo "2.4 G frequency, naive:"
srun -n 1 -N 1 --cpu-freq=1000000 ./benchmark-naive
echo "2.4 G frequency, blas:"
srun -n 1 -N 1 --cpu-freq=1000000 ./benchmark-blas

printf '%s\n' ---------------------------


