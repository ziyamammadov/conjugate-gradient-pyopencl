#!/bin/bash
#SBATCH -Jp_helm
#SBATCH --partition=gpu
#SBATCH --exclude=falcon2
#SBATCH --gres=gpu:tesla:4
#SBATCH -t 10:00:00
#SBATCH --mem=47000

S=10
N=64
NIT=300

module load python/3.9.12
module load py-numpy/1.22.0
module load py-mpi4py/3.1.4
module load py-scipy/1.8.1
module load gnuplot/5.2.8
 

# export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
# export OMP_NUM_THREADS=1
export PYOPENCL_COMPILER_OUTPUT=1
cd /gpfs/space/home/ziya/conjugate-gradient-multi-gpu
srun hostname
# multiple nodes:
# mpirun ./p_h-PY_C-CL.py ${S} ${N} 2 ${NIT}
# single-node run:
./p_h-PY_C-CL.py ${S} ${N} 3 ${NIT}
