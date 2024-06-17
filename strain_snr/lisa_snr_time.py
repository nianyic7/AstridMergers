import numpy as np
import sys,os
import pickle
import argparse
from mpi4py import MPI
import glob
from gwsnrcalc.utils.waveforms import PhenomDWaveforms, EccentricBinaries
from gwsnrcalc import gw_snr_calculator
from gwsnrcalc.utils.sensitivity import SensitivityContainer
import traceback

#----------- MPI Init ----------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('rank:',rank,'size:',size,flush=True)

#-------------- Cmd line Args ------------------------------
parser = argparse.ArgumentParser(description='calc_lisa_snr')
parser.add_argument('--ecc',required=True,type=float,help='eccentricity of orbits')
parser.add_argument('--zind',required=True,type=int,help='merging redshift to use')
parser.add_argument('--savedir',required=True,type=str,help='directory to save data')

args = parser.parse_args()
ecc = args.ecc
zind = args.zind
savedir = args.savedir

if not os.path.exists('/hildafs/home/nianyic/asterix_bh/'+savedir):
    os.makedirs('/hildafs/home/nianyic/asterix_bh/'+savedir)

#----------- Load data ----------------
root = '/hildafs/home/nianyic/asterix_bh/'
if rank==0:
    print('Loading data...',flush=True)
binaries = np.load(root+'bgood_3up.npy')
binaries = np.array([],dtype = binaries.dtype)
for subdir in sorted(glob.glob(root+'extrapolation_scale=1.5/*.npy')):
    binaries = np.concatenate((binaries,np.load(subdir)))
print(len(binaries))
mask = binaries['sigma']>10
mask &= binaries['gamma'] > 0
binaries = binaries[mask]

if rank==0:
    print('Loaded binaries:',len(binaries),flush=True)


bgood = binaries

zm1 = np.load(root+'zmerge_0.npy').T
zm1 = np.load('/hildafs/home/nianyic/binary/process_asterix/other_data/zmerge_extra.npy').T
zm1 = zm1[mask]

if rank==0:
    print('Finished loading data...',flush=True)
    print(len(bgood),len(zm1),flush=True)
    
#---------------------- Split task -----------------------------
nperr = len(bgood)//size
beg = rank*nperr
end = (rank+1)*nperr
if rank==size-1:
    bgood = bgood[beg:]
    zm1 = zm1[beg:]
else:
    bgood = bgood[beg:end]
    zm1 = zm1[beg:end]
    
print('Rank:',rank,'Beginning index:',beg, 'Ending index:',end,flush=True)

#---------------------- Compute LISA SNR----------------------------

sense =  SensitivityContainer(sensitivity_curves='LPA').noise_interpolants
lpa_all = []
lpa_wd_all = []
unmerg = 0
for i,b in enumerate(bgood):
    ee = b['ecce']
    zm = zm1[i][zind]
    if zm<=0:
        lpa_all.append(-2)
        lpa_wd_all.append(-2)
        unmerg+=1
        continue
    m1 = max(b['mass1'],b['mass2'])
    m2 = min(b['mass1'],b['mass2'])
    
    
    if ecc==0.: # use circ wave
        # use PhenomD
        # m1, m2, chi_1, chi_2, z_or_dist, st, et
        bargs = (m1, m2, 0.8,0.8, zm, 5.,0.)
        if (m1/m2)>1e4:
            print(m1,m2)
            lpa_all.append(-1)
            lpa_wd_all.append(-1)
            continue

        circ_wave = PhenomDWaveforms(disttype='redshift')
        res = gw_snr_calculator.parallel_snr_func(num=0,binary_args=bargs, \
                                                  phenomdwave= circ_wave, signal_type=['all'], \
                                                  noise_interpolants=sense, prefactor=1, verbose=-1)
#         except Exception as exc:
#             print(traceback.format_exc(),flush=True)
#             print(exc,flush=True)
#             print(b['id1'],flush=True)
#             print(m1/m2,flush=True)
#             lpa_all.append(-1)
#             lpa_wd_all.append(-1)
#             continue
            
    else: # use ecc wave
        if ecc < 0: # use measured values with a discount of -ecc
            bargs = (m1, m2, zm, 5.,ee*(-ecc),5.)
        else: # use a constant value
            bargs = (m1, m2, zm, 5.,ecc,5.)
        try:
            ecc_wave = EccentricBinaries(disttype='redshift',\
                                 initial_cond_type='time',n_max=50)
            res = gw_snr_calculator.parallel_ecc_snr_func(num=0,binary_args=bargs,\
                                                eccwave=ecc_wave,signal_type='all',\
                                                noise_interpolants = sense,\
                                                prefactor=1,\
                                               verbose=-1)
        except Exception as exc:
            print(traceback.format_exc(),flush=True)
            print(exc,flush=True)
            print(b['id1'],flush=True)
            lpa_all.append(-1)
            lpa_wd_all.append(-1)
            continue
            
    lpa_all.append(res['LPA_all'])
    lpa_wd_all.append(res['LPA_wd_all'])
comm.Barrier()    
A = comm.reduce(unmerg, op=MPI.SUM, root=0)
if rank == 0:
    print('BHs not merged before z=0:',A,flush=True) 
    
lpa_all = np.array(lpa_all)
lpa_wd_all = np.array(lpa_wd_all)

print('Rank %02d saving to file...'%rank,flush=True)
lpa_all.tofile('/hildafs/home/nianyic/asterix_bh/'+savedir+'/lisa_snr%02d.dat'%rank)
lpa_wd_all.tofile('/hildafs/home/nianyic/asterix_bh/'+savedir+'/lisa_snr_wd%02d.dat'%rank)
    
    
    
    

    
