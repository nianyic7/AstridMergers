#!/bin/bash 
#SBATCH -p small
#SBATCH --time=8:00:00
#SBATCH --nodes=1
##SBATCH --mem=200GB
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=m***snap***
#SBATCH --output=slurm-%x.out
#SBATCH --mail-type=END
#SBATCH --mail-user=nianyic@andrew.cmu.edu

module unload python3
source activate fast-mpi4py
which python

snap=***snap***
mergerdir="$HOME/work/Astrid/mergers/mdata/"
outdir="$HOME/scratch3/Astrid/merger_catalog/"
mkdir -p $outdir $outdir/before $outdir/after

srun python3 ../codes/save_galaxy_info_pig.py --snap $snap --mergerroot $mergerdir --rcut 1.5 --outdir $outdir
