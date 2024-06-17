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
hh = 0.6774

#----------- MPI Init ----------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('rank:',rank,'size:',size,flush=True)
#-------------- Cmd line Args ------------------------------
parser = argparse.ArgumentParser(description='calc_lisa_snr')
parser.add_argument('--ecc',required=True,type=float,help='eccentricity of orbits')
parser.add_argument('--zdata',required=True,type=str,help='merging redshift to use: merge,delayed')
parser.add_argument('--mdata',required=True,type=str,help='mass data to use: merge,bondi,edd')
parser.add_argument('--infile',required=True,type=str,help='merger catalog file')
parser.add_argument('--outdir',required=True,type=str,help='directory of outputs')

args = parser.parse_args()
ecc = args.ecc
zdata = args.zdata
mdata = args.mdata
infile = args.infile
outdir = args.outdir

savedir = '/hildafs/datasets/Asterix/SNRs/'+outdir


#----------- Load data ----------------
binaries = np.load(infile)
print('Loaded data:',len(binaries),flush=True)

#---------------------- Split task -----------------------------
nperr = len(binaries)//size
beg = rank*nperr
end = (rank+1)*nperr
if rank==size-1:
    binaries = binaries[beg:]
else:
    binaries = binaries[beg:end]
    
print('Rank:',rank,'Beginning index:',beg, 'Ending index:',end,flush=True)

#---------- load redshift -------------
if zdata == 'merge':
    zmerge = binaries['z']
elif zdata == 'delayed':
    zmerge = binaries['zdelayed']
else:
    raise NameError('unrecognized zmerge type')
#---------- load mass ------------------------
if mdata == 'merge':
    m1s = binaries['m1']
    m2s = binaries['m2']
elif mdata == 'bondi':
    m1s = binaries['bondim1']
    m2s = binaries['bondim2']
elif mdata == 'edd':
    m1s = binaries['eddm1']
    m2s = binaries['eddm2']
else:
    raise NameError('unrecognized mass type')
    
print('Loaded data:','mass type:', mdata,'redshift type:', zdata, flush=True)


#---------------------- Compute LISA SNR----------------------------
sense =  SensitivityContainer(sensitivity_curves='LPA').noise_interpolants
features = ['zmerge','m1','m2','fm','hm','fp','hp','snr']
dtype = ['d','d','d','d','d','d','d','d']

lpa_all = []
unmerg = 0
for i,zm in enumerate(zmerge):
    if i%10000 ==1:
        print('Finished %d'%i, flush=True)
        print(lpa_all[-1],flush=True)
    m1 = m1s[i]
    m2 = m2s[i]
    if m2>m1:
        m1,m2 = m2,m1
        
    if ecc==0.: # use circ wave
        # use PhenomD
        # m1, m2, chi_1, chi_2, z_or_dist, st, et

        if (m1/m2)>=1e4:
            print('cannot handle mass ratio >1e4: q=%.1f'%(m1/m2),flush=True)
            unmerg += 1
            lpa_all.append((zm,m1,m2,-1,-1,-1,-1,-1))
            continue
            
        else:
            bargs = (m1, m2, 0.8,0.8, zm, 5.,0.)
            circ_wave = PhenomDWaveforms(disttype='redshift')
            bb = circ_wave(m1,m2,0.8,0.8, zm, 5.0,0.0)
            hm = np.interp(bb.fmrg,bb.freqs,bb.hc)
            hp = np.interp(bb.fpeak,bb.freqs,bb.hc)
        
            res = gw_snr_calculator.parallel_snr_func(num=0,binary_args=bargs, \
                                                      phenomdwave= circ_wave, signal_type=['all'], \
                                                      noise_interpolants=sense, prefactor=1, verbose=-1)
            lpa_all.append((zm,m1,m2,bb.fmrg,hm,bb.fpeak,hp,res['LPA_all']))
    

lpa_all = np.array(lpa_all,dtype=[(f,dtype[i]) for i,f in enumerate(features)])
print('missing pairs:',unmerg,flush=True)
print('Saving to file...',flush=True)
np.save(savedir+'/snr-%02d.npy'%rank,lpa_all)
print('Saved to:',savedir)
    
    
    

    
