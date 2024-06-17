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


def calc_pos(bh,z):
    fv = 'extrapolate'
    kind = 'linear'
    box = 250000
#     print(bh['z'])
    dpx = np.diff(bh['pos'][:,0])
    dpy = np.diff(bh['pos'][:,1])
    dpz = np.diff(bh['pos'][:,2])
    
    dpx[dpx>box/2] -= box
    dpx[dpx< - box/2] += box
    dpy[dpy>box/2] -= box
    dpy[dpy< - box/2] += box
    dpz[dpz>box/2] -= box
    dpz[dpz< - box/2] += box
    
    posx = bh['pos'][0,0] + np.cumsum(dpx)
    posy = bh['pos'][0,1] + np.cumsum(dpy)
    posz = bh['pos'][0,2] + np.cumsum(dpz)
    
    fx = interp1d(bh['z'][1:],posx,fill_value=fv,kind = kind)
    fy = interp1d(bh['z'][1:],posy,fill_value=fv,kind = kind)
    fz = interp1d(bh['z'][1:],posz,fill_value=fv,kind = kind)
    return np.array([fx(z),fy(z),fz(z)])

def calc_vel(bh,z):
    fv = 'extrapolate'
    kind = 'linear'
    fx = interp1d(bh['z'],bh['vel'][:,0],fill_value=fv,kind = kind)
    fy = interp1d(bh['z'],bh['vel'][:,1],fill_value=fv,kind = kind)
    fz = interp1d(bh['z'],bh['vel'][:,2],fill_value=fv,kind = kind)
    return np.array([fx(z),fy(z),fz(z)])

def get_last_peak(bh1,bh2,m1,m2,zmerge):
    zz = np.linspace(zmerge+1e-6,zmerge+0.4,3000)
    pos1 = calc_pos(bh1,zz)
    pos2 = calc_pos(bh2,zz)
    pc = m1/(m1+m2)*pos1 + m2/(m1+m2)*pos2
    dp = pos2 - pc
    box = 250000
    dp[dp< -box/2] += box
    dp[dp > box/2] -= box
    dpos = np.linalg.norm(dp, axis=0)/hh/(1+zz) 

    ind1 = argrelextrema(dpos, np.greater)[0]

    # for local minima
    ind2 = argrelextrema(dpos, np.less)[0]
   
    if len(ind1)>0:
        indmax = np.unique(ind1)
        ret1 = zz[indmax]
    else:
        ret1 = np.array([0.])
        print('no local max',flush=True)
    if len(ind2)>0:
        indmin = np.unique(ind2)
        ret2 = zz[indmin]
    else:
        ret2 = np.array([0.])
        print('no local min',flush=True)
    return ret1,ret2


#------------------- Arguments -----------------------------------------------
parser = argparse.ArgumentParser(description='save-binary-orbits')

parser.add_argument('--idx',required=True,type=int,help='index of the merger file')

args = parser.parse_args()
idx = int(args.idx)

#---------------------- Read in Mergers -----------------------------------------
    
# load merger info
print('Loading mergers...')
f = '/hildafs/datasets/Asterix/BH_details_bigfile/mdata-z3/bh-merger-R%03d.npy'%idx
print('Processing mergers in:',f,flush=True)
mergers = np.load(f)
print('number of mergers in this bin:',len(mergers),flush=True)

# load dynamics info
path = '/hildafs/home/nianyic/asterix_bh/orbits_z34/'
with open(path+'posvel0-%03d.pkl'%idx, 'rb') as f:
    info = pickle.load(f)
    f.close()
#------------------------------ Main Loop ---------------------------------------    
orbits = {}
for m in mergers:
    ob = {}
    if m['m1'] > m['m2']:
        id1 = m['ID1']
        id2 = m['ID2']
        m1 = m['m1']
        m2 = m['m2']
    else:
        id1 = m['ID2']
        id2 = m['ID1']
        m1 = m['m2']
        m2 = m['m1']
    try:
        bh1 = info[id1]
        bh2 = info[id2]
    except KeyError:
        print('BH not found:',id1,id2)
        continue
    
    zmerge = m['z']
    zz = np.linspace(zmerge,zmerge+0.4,300)
    try:
        pos1 = calc_pos(bh1,zz)
        pos2 = calc_pos(bh2,zz)
    except ValueError:
        print('Cannot calculate position here:',id1,id2)
    pc = m['m1']/(m['m1']+m['m2'])*pos1 +  m['m2']/(m['m1']+m['m2'])*pos2

    dp2 = (pos2 - pc) 
    dp1 = (pos1 - pc) 
    
    box = 250000
    dp2[dp2< -box/2] += box
    dp2[dp2 > box/2] -= box
    
    dp1[dp1< -box/2] += box
    dp1[dp1 > box/2] -= box
    
    
    vel1 = calc_vel(bh1,zz) * (1+zz)# physical km/s
    vel2 = calc_vel(bh2,zz) * (1+zz)
    vc = m['m1']/(m['m1']+m['m2'])*vel1 +  m['m2']/(m['m1']+m['m2'])*vel2
    dv2 = vel2 - vc
    dv1 = vel1 - vc
#     print(dp1.shape)
#     plt.plot(dp1[0],dp1[1])
#     plt.plot(dp2[0],dp2[1])
#     plt.show()
    ob['zmerge'] = m['z']
    ob['dp2'] = dp2/hh/(1+zz) 
    ob['dp1'] = dp1/hh/(1+zz) 
    ob['pcom'] = pc/hh/(1+zz) 
    ob['dv2'] = dv2
    ob['dv1'] = dv1
    
    zmax,zmin = get_last_peak(bh1,bh2,m1,m2,zmerge)
    ob['zmax'] = zmax
    ob['zmin'] = zmin
    orbits[(m['ID1'],m['ID2'])] = ob
    
# save
print('Saving to file..')
path = '/hildafs/home/nianyic/asterix_bh/orbits_z34/'
with open(path+'orbits-%03d.pkl'%idx, 'wb') as f:
    pickle.dump(orbits,f)
    f.close()
print('Done!')
    