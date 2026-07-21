#!/usr/bin/env bash
#SBATCH --job-name=si-k888
#SBATCH --partition=48cp2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/si-k888-qs-test/slurm-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/si-k888-qs-test/slurm-%j.err

set -e
ABACUS=/data/home/df_iopcas_bhj/software-stack/src/abacus-old-libri/build/abacus_3p
ROOT=/data/home/df_iopcas_bhj/ai-runs/si-k888-qs-test

echo "=== SCF ==="
cd "/scf"
mpirun -np 4 "" > scf.out 2>&1
echo "SCF done"

echo "=== NSCF ==="
cd "/nscf"
mpirun -np 4 "" > nscf.out 2>&1
echo "NSCF done"

echo "ALL DONE"
