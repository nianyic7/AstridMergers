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
from scipy.interpolate import interp1d
import pickle
import warnings
import argparse
hh=0.697

from colossus.halo.profile_nfw import NFWProfile
from colossus.cosmology import cosmology
params = {'flat': True, 'H0': 69.7, 'Om0': 0.2865, 'Ob0': 0.04628, 'sigma8': 0.82, 'ns': 0.96}
cosmo = cosmology.setCosmology('myCosmo', params)
Mpc_to_m = 3.086e+22
m_to_mpc = 1./Mpc_to_m
s_to_year = 3.17098e-8
c_Mpc_yr = 1e8*m_to_mpc/s_to_year

msun_mks = 1.989e30
pc_mks = 3.086e16
grav_mks = 6.67e-11
km_mks = 1e3
yr_mks = 3.154e+7
c_mks = 3e8


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
binaries = np.load(f)
print('number of mergers in this bin:',len(binaries),flush=True)

# load dynamics info
path = '/hildafs/home/nianyic/asterix_bh/orbits_z34/'
with open(path+'orbits-%03d.pkl'%idx, 'rb') as f:
    orbits = pickle.load(f)
    f.close()
    
#-------------------------------- Main loop --------------------------------------

ecc_list_shape = {}
apo_list_shape = {}
peri_list_shape = {}
i = 0
for m in binaries[:]:
    zmerge = m['z']
    id1 = m['ID1']
    id2 = m['ID2']
    m2 = m['m2']
    m1 = m['m1']
    try:
        ob = orbits[(min(id1,id2),max(id1,id2))]
    except KeyError:
        try:
            ob = orbits[(max(id1,id2),min(id1,id2))]
        except KeyError:
            print('orbit not found:',id1,id2)
            i +=1
            print(i)
            continue
    
    zmax,zmin = ob['zmax'][0],ob['zmin'][0]
    zz = np.linspace(zmerge,zmerge+0.4,300)
    mask1 = np.argmin(np.abs(zz-zmax))
    mask2 = np.argmin(np.abs(zz-zmin))
    dpmax = ob['dp2'][:,mask1]
    dpmin = ob['dp2'][:,mask2]

    apo = np.linalg.norm(dpmax)
    peri = np.linalg.norm(dpmin)
    if peri <= apo:
        ecc = np.sqrt((apo-peri)/(apo+peri))
        ecc_list_shape[(id1,id2)] = ecc
        apo_list_shape[(id1,id2)] = apo
        peri_list_shape[(id1,id2)] = peri
    else:
        i+=1
        continue
#     ob['ecc_shape'] = ecc
#     m['apo'] = apo
#     m['peri'] = peri
print('shape missed binaries:',i,flush=True)
# path = '/hildafs/datasets/Asterix/BH_details/postprocess/'
# with open(path+'ecc_z34/ecc_shape_%03d.pkl'%idx, 'wb') as f:
#     pickle.dump(ecc_list_shape,f)
#     f.close()

# path = '/hildafs/datasets/Asterix/BH_details/postprocess/'
# with open(path+'ecc_z34/apo_shape_%03d.pkl'%idx, 'wb') as f:
#     pickle.dump(apo_list_shape,f)
#     f.close()

# path = '/hildafs/datasets/Asterix/BH_details/postprocess/'
# with open(path+'ecc_z34/peri_shape_%03d.pkl'%idx, 'wb') as f:
#     pickle.dump(peri_list_shape,f)
#     f.close()
    
    
    

    
#--------------------------------------------------------------
print('Start energy calculation...',flush=True)
# load data
path = '/hildafs/datasets/Asterix/BH_details/postprocess/'
r = 'binary_info_post-z3'
with open(path+r+'.pkl', 'rb') as f:
    prof_all = pickle.load(f)
    f.close()  
    
r = 'binary_info_post'
with open(path+r+'.pkl', 'rb') as f:
    prof_all.update(pickle.load(f))
    f.close()  

# compute potential profiles
from scipy.interpolate import interp1d
from scipy.integrate import quad
fphi_list = {}
for k in prof_all.keys():
    # everything in mks unit
    dp = prof_all[k] 
    rr = dp['rr']*1e3*pc_mks
    rho = dp['density0']+dp['density1']+dp['density4']
    rho *= msun_mks/(pc_mks**3)
    fprof = interp1d(rr,rho,bounds_error=False,fill_value='extrapolate')
    rout = np.logspace(-2,1.5,3000)
    rout *= 1e3*pc_mks
    dr = rout[1:] - rout[:-1]
    rc = 0.5*(rout[1:] + rout[:-1])
    rhos = fprof(rc)
    dmr = 4*np.pi*rhos*rc**2*dr
    mr = np.cumsum(dmr)
    dphi1 = dr*rc**2*rhos
    dphi2 = dr*rc*rhos
    first = np.cumsum(dphi1)/rc
    second = np.sum(dphi2) - np.cumsum(dphi2)
    phis = -4*np.pi*grav_mks*(first+second)
    fphi_list[k] = [rc,phis,mr]
print('Finished computing potential profiles!',flush=True)    
    
    
    
#---------------------------------------------------

from scipy.optimize import root_scalar
from scipy.optimize import fsolve
ee_list_en = {}
apo_list_en = {}
peri_list_en = {}

nroots = []
i = 0
for m in binaries[:]:
    zmerge = m['z']
    id1 = m['ID1']
    id2 = m['ID2']
    m2 = m['m2']
    m1 = m['m1']
    try:
        ob = orbits[(min(id1,id2),max(id1,id2))]
    except KeyError:
        try:
            ob = orbits[(max(id1,id2),min(id1,id2))]
        except KeyError:
            print('orbit not found:',id1,id2)
            i +=1
            print(i)
            continue
    
    zmax,zmin = ob['zmax'][0],ob['zmin'][0]
    zz = np.linspace(zmerge,zmerge+0.4,300)
    mask = zz <= max(zmax,zmin,zmerge+0.1)
    dp2 = ob['dp2'][:,mask]
    dv2 = ob['dv2'][:,mask]
    dv2 = dv2*1e3  #m/s
    dp2 = dp2*pc_mks*1e3  #m
    
    JJ = np.mean(np.linalg.norm(np.cross(dv2.T,dp2.T),axis=1)) #mks
    try:
        k = (min(id1,id2),max(id1,id2))
        rr = fphi_list[k][0] # mks
        phi = fphi_list[k][1]

        ob = orbits[(min(id1,id2),max(id1,id2))]
    except KeyError:
        try:
            k = (max(id1,id2),min(id1,id2))
            rr = fphi_list[k][0] # mks
            phi = fphi_list[k][1]
        except KeyError:
            print('orbit not found:',id1,id2,flush=True)
            i +=1
            print(i)
            continue

    phi_of_r = interp1d(rr,phi,bounds_error=False,fill_value='extrapolate')
    EE = np.mean(0.5*(np.linalg.norm(dv2,axis=0)**2) \
                 + phi_of_r(np.linalg.norm(dp2,axis=0)))
    KE = 0.5*(np.linalg.norm(dv2,axis=0)**2)
    ke_of_r = interp1d(np.linalg.norm(dp2,axis=0),KE,bounds_error=False,fill_value='extrapolate')
    def f(r):
#         ret = r**(-2) + 2*(phi_of_r(r)-EE)/JJ**2
        ret = r**(-2) + 2*(-ke_of_r(r))/JJ**2
        return ret
    
    init = np.linspace(0.1e3*pc_mks,10e3*pc_mks,50)

    root = fsolve(f, init)
    posroot = root[root>0]
#     print(set(np.around(posroot/pc_mks/1e3,3)))
#     print(set(np.isclose(f(root), np.zeros(init,dtype=np.float64))))
    posroot = np.array(list(set(np.around(posroot/pc_mks/1e3,3))))
    nroots.append(len(posroot))

    if (len(posroot)>1) & ((id1,id2) in ecc_list_shape.keys()):
        apo = np.sort(posroot)[-1]
        per = np.sort(posroot)[0]
        ee = (apo-per)/(apo+per)
        ees = ecc_list_shape[(id1,id2)]
        apos = apo_list_shape[(id1,id2)]
        pers = peri_list_shape[(id1,id2)]
        ee_list_en[(id1,id2)] = [ee,ees]
        apo_list_en[(id1,id2)] = [apo,apos]
        peri_list_en[(id1,id2)] = [per,pers]
    else:
        continue
        
        
path = '/hildafs/datasets/Asterix/BH_details/postprocess/'

with open(path+'ecc_z34/ecc_%03d.pkl'%idx, 'wb') as f:
    pickle.dump(ee_list_en,f)
    f.close()
with open(path+'ecc_z34/apo_%03d.pkl'%idx, 'wb') as f:
    pickle.dump(apo_list_en,f)
    f.close()
with open(path+'ecc_z34/peri_%03d.pkl'%idx, 'wb') as f:
    pickle.dump(peri_list_en,f)
    f.close()
    
    
    
    
    
    
    
    
    
    
    
    
    