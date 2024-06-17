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
parser = argparse.ArgumentParser(description='calc_lisa_strain')
parser.add_argument('--zind',required=True,type=int,help='merger redshift to use')
parser.add_argument('--savedir',required=True,type=str,help='directory to save data')

args = parser.parse_args()
savedir = args.savedir
zind = args.zind

if not os.path.exists('/hildafs/home/nianyic/asterix_bh/'+savedir):
    os.makedirs('/hildafs/home/nianyic/asterix_bh/'+savedir)


#----------- Load data --------------
if rank==0:
    print('Loading data...',flush=True)

# load mergers
root = '/hildafs/home/nianyic/asterix_bh/'
binaries = np.load(root+'bgood_3up.npy')
binaries = np.array([],dtype = binaries.dtype)
for subdir in sorted(glob.glob(root+'extrapolation_scale=1.5/*.npy')):
    binaries = np.concatenate((binaries,np.load(subdir)))
print(len(binaries))
mask = binaries['sigma']>10
mask &= binaries['gamma'] > 0
binaries = binaries[mask]
print(len(binaries))
bgood = binaries

if rank==0:
    print('Loaded binaries:',len(binaries),flush=True)

zm1 = np.load(root+'zmerge_0.npy').T
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
features = ['zmerge','m1','m2','fm','hm','fp','hp','ff','hh']
dtype = ['d','d','d','d','d','d','d','1024d','1024d']

inds = np.random.choice(range(len(bgood)),3000)
bselect = bgood
data = np.zeros(len(bselect),dtype=[(f,dtype[i]) for i,f in enumerate(features)])

for i,b in enumerate(bselect):
    zm = zm1[i][zind]
    if zm<=0:
        for f in features:
            data[i][f] = -1
        continue
    m1 = max(b['mass1'],b['mass2'])
    m2 = min(b['mass1'],b['mass2'])
    data[i]['zmerge'] = zm
    data[i]['m1'] = m1
    data[i]['m2'] = m2
    try:
        circ_save = PhenomDWaveforms(num_points=1024)
        bb = circ_save(m1,m2,0.8,0.8, zm, 5.0,0.0)
        hm = np.interp(bb.fmrg,bb.freqs,bb.hc)
        hp = np.interp(bb.fpeak,bb.freqs,bb.hc)
        data[i]['fm'] = bb.fmrg
        data[i]['fp'] = bb.fpeak
        data[i]['hm'] = hm
        data[i]['hp'] = hp
        data[i]['ff'] = bb.freqs
        data[i]['hh'] = bb.hc
    except Exception as exc:
        print(traceback.format_exc(),flush=True)
        print(exc,flush=True)
        print(b['id1'],flush=True)
        for f in features:
            data[i][f] = -1
        continue

print('Rank %02d saving to file...'%rank,flush=True)
np.save('/hildafs/home/nianyic/asterix_bh/'+savedir+'/lisa_strain%02d.npy'%rank,data)

    
    
    

    
