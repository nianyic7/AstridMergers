#!/bin/bash
#SBATCH -p RM
#SBATCH -N 2
#SBATCH --ntasks=56
#SBATCH --cpus-per-task=1
#SBATCH --job-name=snr_mm
#SBATCH --time=8:00:00
##SBATCH --mem=100GB
#SBATCH --output=slurm-astrid-mm.out

source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate grsnr_env
conda list
module list
infile='/hildafs/datasets/Asterix/BH_details_bigfile/ASTRID-merger-catalog-z2.npy'
outdir='astrid_snr_raw_z2'
mpirun -n 56 python3 astrid_snr_circ.py --ecc 0 --zdata 'merge' --mdata 'merge' --infile $infile --outdir $outdir
