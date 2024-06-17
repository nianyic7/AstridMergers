import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from bigfile import BigFile
import glob,os,struct
from scipy.ndimage import gaussian_filter1d as gf1d
import scipy.integrate as integrate
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import seaborn as sns
# from bh_tools import *
import pickle
import warnings
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
hh = 0.697
import argparse

    
#------------------- Arguments -------------------------
parser = argparse.ArgumentParser(description='save-binary-orbits')

parser.add_argument('--idx',required=True,type=int,help='index of the merger file')

args = parser.parse_args()
idx = int(args.idx)
#--------------------------------------------------------------------------------
    
# load merger info
print('Loading mergers...')
f = '/hildafs/datasets/Asterix/BH_details_bigfile/mdata-z3/bh-merger-R%03d.npy'%idx
print('Processing mergers in:',f,flush=True)
mergers = np.load(f)
print('number of mergers in this bin:',len(mergers),flush=True)
ids = np.unique(np.concatenate((mergers['ID1'],mergers['ID2'])))
print('Number of unque BHs to read:',len(ids))

#----------------------------------------------------------------------------------

# Figure out which details file we need to read in
# load snapshot info    
root = '/hildafs/datasets/Asterix/'
snapz_all = np.loadtxt(root+'PIG_files/Snapshots.txt',unpack=True)
snapz_all[1] = 1./snapz_all[1]-1
snapz = []
for i in snapz_all[0]:
    if os.path.isdir(root+'BH_details_bigfile/BH-Details-R%03d'%i):
        snapz.append([i,snapz_all[1,int(i)]])
snapz = np.array(snapz).T
print(snapz.T)

zmax = max(mergers['z'])+0.4
zmin = min(mergers['z'])
print(zmax,zmin)
snapind = ((snapz[1]<zmax)&(snapz[1]>=zmin)).nonzero()[0]
snapread = snapz[0][snapind].astype(np.int)
if len(snapread)==0:
    snapread = np.array([idx])
print('bins to read:',snapread)

# now start reading info
all_info = []
for ss in snapread:
    if ss==87:
        continue
    # load IDs
    print('Reading bin:',ss)
    pig = BigFile(root+'BH_details_bigfile/BH-Details-R%03d'%ss)
    data = pig.open('BHID')[:]
    mask = np.in1d(data,ids)
    idsave = data[mask]
    print('total number of data points:',len(idsave),flush=True)
    
    # declare array for storing data
    features = ['ID','z','pos','vel']
    dtype = ['q','d','3d','3d']
    bhlist = np.zeros(len(idsave),dtype=[(f,dtype[i]) for i,f in enumerate(features)])
    # read in all data
    bhlist['ID'] = idsave
    bhlist['pos'] = pig.open('BHpos')[:][mask]
    bhlist['vel'] = pig.open('BHvel')[:][mask]
    bhlist['z'] = pig.open('z')[:][mask]
    all_info.append(bhlist)

all_info = np.concatenate(all_info)
all_info.sort(order=["ID", "z"])
all_info = np.split(all_info, np.where(np.diff(all_info['ID']))[0]+1)
print('total BH processed:',len(all_info),flush=True)
all_info = {b['ID'][0]:b for b in all_info}

# save
print('Saving...')
path = '/hildafs/home/nianyic/asterix_bh/orbits_z34/'
with open(path+'posvel0-%03d.pkl'%idx, 'wb') as f:
    pickle.dump(all_info,f)
    f.close()

print('Done!')


    
    