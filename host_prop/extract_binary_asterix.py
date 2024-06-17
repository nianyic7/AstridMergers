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



vcut = 5.0
extrapolate_scale=1.2 # times the gravitational softening length

hh=0.697
from scipy.optimize import curve_fit


def get_obt(pig):
    lbt = pig.open('FOFGroups/LengthByType')[:]
    obt = np.cumsum(lbt,axis=0)
    a1 = np.array([[0,0,0,0,0,0]],dtype=np.uint64)
    obt = np.append(a1,obt,axis=0)
    return obt

def power_law(x, a, b):
    return a*np.power(x, b)

def fit_density_plaw(z,r,rho,count,soft=1.5,scale=1.2):
    soft = soft/(1+z)
    mask = r>soft*scale
    mask &= rho>0
    mask &= count>10
    rr = r[mask][:10]
    pr = rho[mask][:10]
    if (len(pr)<4) or (len(rr)<4):
        return np.array([-1.,1.])
    pars,cov = curve_fit(f=power_law, xdata=rr, ydata=pr, p0=[0, 0], bounds=(-np.inf, np.inf))
    stdevs = np.sqrt(np.diag(cov))
    res = pr - power_law(rr, *pars)
    return pars

def vdisp(arr):
    if len(arr)>4:
        vmean = np.mean(arr,axis=0)
        v2 = np.linalg.norm(arr - vmean,axis=1)**2
        sigma = np.sqrt(np.mean(v2))/np.sqrt(3)
        return sigma
    else:
        return 0.
                
def dist_to_bh(pos,bpos,lbox,scale):
    dr = pos-bpos
    dr[dr > lbox/2.] -= lbox
    dr[dr < -lbox/2.] +=lbox
    return np.linalg.norm(dr*scale/hh,axis=-1)



# load snapshot info    
root = '/hildafs/datasets/Asterix/PIG_files/'
snapz_all = np.loadtxt(root+'Snapshots.txt',unpack=True)
snapz_all[1] = 1./snapz_all[1]-1
snapz = []
j = 0
for i in snapz_all[0]:
    if os.path.isdir(root+'PIG_%03d/'%i):
        snapz.append([i,snapz_all[1,int(i)],j])
        j+=1
snapz = np.array(snapz).T
print(snapz)
mask = snapz[0]>142
mask &= snapz[0]<155
snapz = snapz[:,mask]
print(snapz)


# load merger info
print('Loading mergers...')
mergers = []
for f in sorted(glob.glob('/hildafs/datasets/Asterix/BH_details_bigfile/mdata-z3/bh-merger*.npy')):
    print(f)
    mergers.append(np.load(f))
mergers = np.concatenate(mergers)
print('total number of mergers:',len(mergers))

mergers.sort(order='z')
mbinned = [[]]+[[m for m in mergers\
            if (m[0]>=snapz[1][i]) &  (m[0]<snapz[1][i-1])] for i in range(1,len(snapz[1]))]

# one list for each snapshot
prof_dict = {}
prof_dict1 = {}


#----------------------------------- Loop over the snapshots -------------------------------------------#

for i,zsnap in enumerate(snapz[1]):
    if (mbinned[i] == []) & (i!= len(mbinned)-1):
        if mbinned[i+1] == []:
            print('skipping %d th bin'%i)
            continue
    # load snapshot
    pig_dir = root+'PIG_%03d/'%snapz[0][i]
    if not os.path.isdir(root+'PIG_%03d/'%snapz[0][i]):
        print(root+'PIG_%03d/'%snapz[0][i]+' does not exit!')
        continue

    pig = BigFile(pig_dir)
    # get obt
    obt = get_obt(pig)
    
    # read BH
    bhpos = pig.open('5/Position')[:]
    bhminpos = pig.open('5/BlackholeMinPotPos')[:]
    bhgroup = pig.open('5/GroupID')[:]
    bhid = pig.open('5/ID')[:]
    bhseed = pig.open('5/BlackholeMseed')[:]
    
    try:
        scalefac = pig.open('Header').attrs['Time'][0]
    except KeyError:
        print('header issue')
        scalefac = 1./(1+zsnap)
    print('Loaded snapshot:',snapz[0][i],'Redshift=',zsnap,'Scale factor=',scalefac)


    #---------------------------------------------------------------------------------#
    #-------------------- Binaries in the next bin -----------------------------------#
    print('processing bin (pre-merger):',i+1)
    if i!= len(mbinned)-1:
        print('# bianries in this bin:',len(mbinned[i+1]))
        omit = 0
        for m in mbinned[i+1]:
            id1 = m['ID1']
            id2 = m['ID2']
            #------------- First BH--------------#
            try:
                ibh = bhid==id1
                bpos = bhpos[ibh][0]
                bminpos = bhminpos[ibh][0]
                bseed = bhseed[ibh][0]
            except IndexError:
                print('BH1 not found:',id1)
                omit += 1
                continue
            bgroup = bhgroup[ibh][0]

            # stars in group
            ptype = 4
            spos = pig.open(str(ptype)+'/Position')[obt[bgroup-1][ptype]:obt[bgroup][ptype]]
            svel = pig.open(str(ptype)+'/Velocity')[obt[bgroup-1][ptype]:obt[bgroup][ptype]]
            smass = pig.open(str(ptype)+'/Mass')[obt[bgroup-1][ptype]:obt[bgroup][ptype]]*1e10/hh
            sgroup = pig.open(str(ptype)+'/GroupID')[obt[bgroup-1][ptype]:obt[bgroup][ptype]]

            assert np.all(sgroup==bgroup)

            dr = dist_to_bh(pos=spos,bpos=bminpos,lbox=250000,scale=scalefac)

            mask = dr<vcut
            sig1 = vdisp(svel[mask])
            seed1 = bseed*1e10/hh
            #------------- Second BH--------------#
            try:
                ibh = bhid==id2
                bpos = bhpos[ibh][0]
                bminpos = bhminpos[ibh][0]
                bseed = bhseed[ibh][0]
                
            except IndexError:
                print('BH2 not found:',id2)
                omit += 1
                continue
            bgroup = bhgroup[ibh][0]

            # stars in group
            ptype = 4
            spos = pig.open(str(ptype)+'/Position')[obt[bgroup-1][ptype]:obt[bgroup][ptype]]
            svel = pig.open(str(ptype)+'/Velocity')[obt[bgroup-1][ptype]:obt[bgroup][ptype]]
            smass = pig.open(str(ptype)+'/Mass')[obt[bgroup-1][ptype]:obt[bgroup][ptype]]*1e10/hh
            sgroup = pig.open(str(ptype)+'/GroupID')[obt[bgroup-1][ptype]:obt[bgroup][ptype]]
            assert np.all(sgroup==bgroup)


            dr = dist_to_bh(pos=spos,bpos=bminpos,lbox=250000,scale=scalefac)
            mask = dr<vcut
            sig2 = vdisp(svel[mask])
            seed2 = bseed
            
            prof_dict1[(m['ID1'],m['ID2'])] = [sig1,sig2,seed1,seed2]
    
    print('omitted:',omit)
    
    #----------------------------------------------------------------------------------#
    #-------------------- Binaries in the current bin ---------------------------------#
    # loop over BHs in this bin
    print('processing bin (post-merger):',i)
    omit = 0
    for m in mbinned[i]:
        bres = max(m['ID1'],m['ID2']) # larger id remains
        # find bh group
        try:
            ibh = bhid==bres
            bpos = bhpos[ibh][0]
            bminpos = bhminpos[ibh][0]
        except IndexError:
            print('BH not found:',bres)
            omit += 1
            continue
        bgroup = bhgroup[ibh][0]

        # stars in group
        ptype = 4
        spos = pig.open(str(ptype)+'/Position')[obt[bgroup-1][ptype]:obt[bgroup][ptype]]
        svel = pig.open(str(ptype)+'/Velocity')[obt[bgroup-1][ptype]:obt[bgroup][ptype]]
        smass = pig.open(str(ptype)+'/Mass')[obt[bgroup-1][ptype]:obt[bgroup][ptype]]*1e10/hh
        sgroup = pig.open(str(ptype)+'/GroupID')[obt[bgroup-1][ptype]:obt[bgroup][ptype]]
        
        assert np.all(sgroup==bgroup)

        dr = dist_to_bh(pos=spos,bpos=bminpos,lbox=250000,scale=scalefac)
        bins = np.logspace(-1,1.5,60)
        dv = 4./3*np.pi*np.diff(bins**3)*1e9
        rr = np.exp(0.5*(np.log(bins[1:])+np.log(bins[:-1])))
        
        mask = dr<vcut
        disp = vdisp(svel[mask])
        mstot = np.sum(smass[mask]) # stellar mass
       
        
        
        mr,_ = np.histogram(dr, bins=bins, weights=smass)
        mcum = np.cumsum(mr)
        reff = np.exp(np.interp(0.5*mstot,mcum,np.log(rr))) # in kpc
        
        
        count,_ = np.histogram(dr, bins=bins)
        density = mr/dv

        prof_fit = fit_density_plaw(m['z'],rr,gf1d(density,2),count,scale=extrapolate_scale)
        rho = prof_fit[0] ## density at 1kpc
        gamma = -prof_fit[1] ## positive power law
        
        
        
        if rho.dtype != np.float64:
            print(rho)
            omit += 1
            continue
        try:
            sig1 = prof_dict1[(m['ID1'],m['ID2'])][0]
            sig2 = prof_dict1[(m['ID1'],m['ID2'])][1]
            seed1 = prof_dict1[(m['ID1'],m['ID2'])][2]
            seed2 = prof_dict1[(m['ID1'],m['ID2'])][3]
            prof_dict[(m['ID1'],m['ID2'])] = [m['z'],m['ID1'],m['ID2'],m['m1'],m['m2'],seed1,seed2,sig1,sig2,\
                                              disp,rho,gamma,mstot,reff]
        except KeyError:
            print('skipping:',(m['ID1'],m['ID2']))
            omit += 1
            continue

    print('omitted:',omit)

    print(len(prof_dict.keys()))

    if len(prof_dict.values())>0:
        ids = np.array([[prof_dict[k][1],prof_dict[k][2]] for k in prof_dict.keys()],dtype=np.int64)
        others = np.array([[prof_dict[k][0],prof_dict[k][3],prof_dict[k][4],\
                            prof_dict[k][5],prof_dict[k][6],prof_dict[k][7],prof_dict[k][8],prof_dict[k][9],\
                           prof_dict[k][10],prof_dict[k][11],prof_dict[k][12],prof_dict[k][13]] for k in \
                           prof_dict.keys()],dtype=np.float)
        
        
        f1 = ['id1','id2']
        f2 = ['zmerge','mass1','mass2','seed1','seed2','sig1','sig2','sigma','rho','gamma','mstot','reff']

        dtype1 = ['q','q']
        dtype2 = ['d','d','d','d','d','d','d','d','d','d','d','d']
        dt1 = {'names':f1, 'formats':dtype1}
        dt2 = {'names':f2, 'formats':dtype2}
        ids.dtype=dt1
        others.dtype=dt2

        import numpy.lib.recfunctions as rfn

        newb = [ids[:,0],others[:,0]]
        newb = rfn.merge_arrays(newb, flatten = True, usemask = False)    

        path = '/hildafs/datasets/Asterix/BH_details/postprocess/'
        np.save(path+'binary_z4-comp',newb)
      