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
import argparse

extrapolate_scale=1.2 # times the gravitational softening length
from scipy.optimize import curve_fit


def get_obt(pig):
    lbt = pig.open('FOFGroups/LengthByType')[:]
    obt = np.cumsum(lbt,axis=0)
    a1 = np.array([[0,0,0,0,0,0]],dtype=np.uint64)
    obt = np.append(a1,obt,axis=0)
    return obt


def find_zrange(sthis):
    isnap = (allsnaps["snap_num"] == sthis).nonzero()[0][0]
    
    if isnap == 0: # this is the first snap with 4/5
        snext = allsnaps[1]["snap_num"]
        znext = allsnaps[1]["redshift"]
        sprev = 0
        zprev = 100.
    elif isnap == len(allsnaps) - 1: # this is the last snap with 4/5
        snext = 1024
        znext = 0.
        sprev = allsnaps[-2]["snap_num"]
        zprev = allsnaps[-2]["redshift"]
    else:
        snext = allsnaps[isnap + 1]["snap_num"]
        sprev = allsnaps[isnap - 1]["snap_num"]
        znext = allsnaps[isnap + 1]["redshift"]
        zprev = allsnaps[isnap - 1]["redshift"]
    return zprev, znext


def select_mergers(mergers, zmax, zmin):
    print('Selecting mergers to process...')
    mask = mergers['z'] >= zmin
    mask &= mergers['z'] < zmax
    merger = mergers[mask]
    return merger


def load_all_mergers(mroot):
    flist = sorted(glob.glob(mroot + "bh-merger-*.npy"))
    merger_list = []
    for ff in flist:
        merger = np.load(ff)
        merger_list.append(merger)
    mergers = np.concatenate(merger_list)
    return mergers


def get_snaps_with_stars(roots):
    f4_snaps = []
    for root in roots:
        ffs = sorted(glob.glob(root + "PIG_*"))
        for ff in ffs:
            if os.path.exists(ff + "/4/Position") and os.path.exists(ff + "/5/Position"):
                pig       = BigFile(ff)
                battr     = pig["Header"].attrs
                scale_fac = pig["Header"].attrs["Time"][0]
                redshift  = 1./scale_fac - 1
                snap      = int(ff.split('_')[-1])

                f4_snaps.append((snap, redshift))
            
    dtype    = np.dtype([("snap_num", int), ("redshift", np.float32)])
    f4_snaps = np.array(f4_snaps, dtype=dtype)
    f4_snaps = np.sort(f4_snaps, order=["snap_num"])
    print("All snapshots with stars:", f4_snaps, flush=True)
    
    return f4_snaps
        

        
def load_cosmology(pig):
    # you can check the redshift by reading the attributes of the snapshot
    battr = pig["Header"].attrs
    scale_fac = battr["Time"][0]
    redshift = 1./battr["Time"][0] - 1
    Lbox = battr['BoxSize']
    hh = battr['HubbleParam']
    om0 = battr['Omega0']
    omb = battr['OmegaBaryon']
    oml = battr['OmegaLambda']
    Nfof = battr['NumFOFGroupsTotal']
    sigma8 = 0.82

    return Lbox, redshift, hh


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
        v2 = np.linalg.norm(arr - vmean,axis = 1) ** 2
        sigma = np.sqrt(np.mean(v2))/np.sqrt(3)
        return sigma
    else:
        return 0.
           
        
def dist_to_bh(pos, bpos, lbox, scale):
    dr = pos - bpos
    dr[dr > lbox/2.] -= lbox
    dr[dr < -lbox/2.] +=lbox
    return np.linalg.norm(dr * scale / hh,axis=-1)



def galaxy_info(bidx, profile = False):
    gidx = pig['5/GroupID'][bidx] - 1
    bpos = pig['5/Position'][bidx]
    beg = Offset[gidx]
    end = beg + Length[gidx]
    mseed = pig['5/BlackholeMseed'][bidx] * 1e10/hh
    aseed = pig['5/StarFormationTime'][bidx]
    zseed = 1./ aseed - 1

    pos4  = pig['4/Position'][beg[4] : end[4]]
    mass4 = pig['4/Mass'][beg[4] : end[4]]
    dr_p = dist_to_bh(pos = pos4,bpos = bpos,lbox = Lbox,scale = 1./(1 + redshift))
    mask = dr_p < rcut
    
    try:
        vel4  = pig['4/Velocity'][beg[4] : end[4]]
        sigma = vdisp(vel4[mask]) * (1 + redshift) # peculiar velocity
    except:
        sigma = -1.

    m4tot = np.sum(mass4[mask]) * 1e10/hh
    mhtot = pig['FOFGroups/MassByType'][gidx][1] * 1e10/hh
    
    if not profile:
        return mseed, zseed, m4tot, mhtot, sigma
    
    bins = np.logspace(-1, 1.5, 60)
    dv = 4. / 3 * np.pi * np.diff(bins ** 3) * 1e9
    rr = np.exp(0.5 * (np.log(bins[1:]) + np.log(bins[:-1]))) 
    mr, _ = np.histogram(dr_p, bins = bins, weights = mass4)
    
    count, _ = np.histogram(dr_p, bins = bins)
    density = mr / dv

    prof_fit = fit_density_plaw(redshift, rr, gf1d(density, 2), count, scale = extrapolate_scale)
    rho = prof_fit[0] ## density at 1kpc
    gamma = - prof_fit[1] ## positive power law
    
    return mseed, zseed, m4tot, mhtot, sigma, rho, gamma


def get_galaxy_info_pre(merger):
    id1, id2 = merger["ID1"], merger["ID2"]
    m1, m2   = merger["m1"], merger["m2"]
    zmerge   = merger["z"]
    try:
        bidx1 = id2idx[id1]
        mseed1, zseed1, m4tot1, mhtot1, sigma1 = galaxy_info(bidx1)
    except KeyError:
        mseed1, zseed1, m4tot1, mhtot1, sigma1 = -1, -1, -1, -1, -1
    try:
        bidx2 = id2idx[id2]
        mseed2, zseed2, m4tot2, mhtot2, sigma2 = galaxy_info(bidx2)
    except KeyError:
        mseed2, zseed2, m4tot2, mhtot2, sigma2 = -1, -1, -1, -1, -1

    return (zmerge, id1, id2, m1, m2, m4tot1, m4tot2, sigma1, sigma2, mseed1, mseed2, zseed1, zseed2, redshift, snap)
    
    
    
def get_galaxy_info_post(merger):
    id1, id2 = merger["ID1"], merger["ID2"]
    m1, m2   = merger["m1"], merger["m2"]
    zmerge = merger["z"]
    idr = max(id1, id2)
    try:
        bidx1 = id2idx[idr]
        mseed1, zseed1, m4tot1, mhtot1, sigma1, rho1, gamma1 = galaxy_info(bidx1, profile=True)
    except KeyError:
        mseed1, zseed1, m4tot1, mhtot1, sigma1, rho1, gamma1 = -1, -1, -1, -1, -1, -1, -1
        
    return (zmerge, id1, id2, m1, m2, m4tot1, mhtot1, sigma1, rho1, gamma1, redshift, snap)


if __name__ == "__main__":

    #-------------- Cmd line Args ------------------------------
    parser = argparse.ArgumentParser(description='subgrpID-subfind')
    parser.add_argument('--mergerroot',required=True,type=str,help='path of the raw merger file directory')
    parser.add_argument('--snap',required=True, type=int,help='pig snapshot number')
    parser.add_argument('--rcut',required=True,type=float,help='radial cut for galaxy info, in kpc')
    parser.add_argument('--outdir',required=True,type=str,help='path of the output file directory')
    args = parser.parse_args()
    
    #--------------------------
    
    snap = int(args.snap)
    mroot = args.mergerroot
    outdir = args.outdir
    rcut = args.rcut
    
    roots = ["/home1/08942/nianyic/ASTRID2_PIG/", "/home1/08942/nianyic/asterix/PIG/"]
    allsnaps = get_snaps_with_stars(roots)
    
    if snap not in allsnaps['snap_num']:
        raise Exception("no star/bh data in snapshot %d"%(snap))
    try:
        pig = BigFile(roots[0] + "PIG_%03d"%snap)
    except:
        pig = BigFile(roots[1] + "PIG_%03d"%snap)
    
    # --------- prepare some data --------------
    BHIDs = pig['5/ID'][:]
    id2idx = {bhid : i for i,bhid in enumerate(BHIDs)}
    del BHIDs
    
    Offset = get_obt(pig)
    Length = pig['FOFGroups/LengthByType']
    
    #--------------- load mergers relavant to this snapshot ------------------
    Lbox, redshift, hh = load_cosmology(pig)
    mergers = load_all_mergers(mroot)
    zprev, znext = find_zrange(snap)
    
    #------------------ Pre-merger ------------------------------
    # mergers happening after the current snapshot (i.e. this snapshot is "pre" merger)
    # need znext to cutoff the boundary caes
    
    mergers_pre  = select_mergers(mergers, zmax = redshift, zmin = znext)
    print('Loaded total number of pre-merger events:', len(mergers_pre), flush=True)
    

    f1 = ["zmerge", "id1", "id2", "m1", "m2", "m4tot1", "m4tot2", \
          "sigma1", "sigma2", "mseed1", "mseed2", "zseed1", "zseedo2", "zsnap", "snap_num"]
    d1 = ["d", "q", "q", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "i"]
    dtype1 = {'names':f1, 'formats':d1}
    
    savedir = outdir + "before/before-merger-gal-S%03d.npy"%(snap)
    if os.path.isfile(savedir):
        print('skipping as already processed in %s'%savedir, flush=True)
        
    info_pre = [get_galaxy_info_pre(m) for m in mergers_pre]
    info_pre = np.array(info_pre, dtype = dtype1)
    
    print("Saving to:", savedir, flush=True)
    np.save(savedir, info_pre)
        
        
    #------------------ Post-merger ------------------------------
    # mergers happening before the current snapshot (i.e. this snapshot is "post" merger)
    mergers_post  = select_mergers(mergers, zmax = zprev, zmin = redshift)
    print('Loaded total number of post-merger events:', len(mergers_post), flush=True)
    
    f2 = ["zmerge", "id1", "id2", "m1", "m2", "m4tot", "mhtot", "sigma", "rho", "gamma", "zsnap", "snap_num"]
    d2 = ["d", "q", "q", "d", "d", "d", "d", "d", "d", "d", "d", "i"]
    dtype2 = {'names':f2, 'formats':d2}
    
    #--------------- fetch galaxy info ------------------
    savedir = outdir + "after/after-merger-gal-S%03d.npy"%(snap)
    if os.path.isfile(savedir):
        print('skipping as already processed in %s'%savedir, flush=True)

    info_post = [get_galaxy_info_post(m) for m in mergers_post]
    info_post = np.array(info_post, dtype = dtype2)
    
    print("Saving to:", savedir, flush=True)
    np.save(savedir, info_post)
        

    print('Done!', flush=True)
    
    
    
    
        
    
    

















