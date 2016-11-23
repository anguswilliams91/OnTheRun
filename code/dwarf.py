from __future__ import division, print_function 

import numpy as np, gus_utils as gu
from scipy.optimize import brentq

def Phi(r,params):

    """
    SPL potential
    """

    vesc_Rsun,alpha = params
    return -.5*vesc_Rsun**2.*(r/8.5)**-alpha

def q_gaussian(v,vesc,sigma):

    """
    qGaussian function to represent the velocity distribution of 
    the dwarfs
    """

    beta = np.sqrt( 0.5/sigma**2. - 1.5/vesc**2.)
    q = 1. - (beta*vesc)**-2.
    Cq = np.sqrt((1.-q)/np.pi)*gamma((5.-3*q)/(2.*(1.-q)))/gamma((2.-q)/(1.-q))
    return beta*Cq*(1. - (1-q)*(beta*v)**2.)**(1./(1.-q))

def rperi_rapo_eq(r,E,L,params):

    """equation for finding rp, ra"""

    return E  - .5*L**2. / r**2. - Phi(r,params)

def rp_ra_getstart(r,E,L,params,apo=False):
    """Find suitable place to start the minimization routine"""
    if apo:
        rtry = 1.1*r
    else:
        rtry = 0.9*r
    while rperi_rapo_eq(rtry,E,L,params)>=0. and rtry>1e-9:
        if apo:
            if rtry>1e10:
                raise Exception("This orbit looks unbound.")
            rtry*=1.1
        else:
            rtry*=0.9
    if rtry<1e-9:
        return 0.
    return rtry

def get_rp_ra(r,vr,vT,params):
    """Find the limits for the radial action integral"""
    E = .5*(vr**2. + vT**2.) + Phi(r,params)
    L = vT*r
    eps=1e-8
    if np.abs(rperi_rapo_eq(r,E,L,params))<1e-7: #we are at peri or apo
        peps = rperi_rapo_eq(r+eps,E,L,params)
        meps = rperi_rapo_eq(r-eps,E,L,params)
        if peps<0. and meps>0.: #we are at apo
            ra = r
            rstart = rp_ra_getstart(r-eps,E,L,params)
            if rstart==0.: rp = 0.
            else:
                rp = brentq(rperi_rapo_eq,rstart,r-eps,(E,L,params),maxiter=200)
        elif peps>0. and meps<0.: #we are at peri
            rp = r
            rend = rp_ra_getstart(r,E,L,params,apo=True)
            ra = brentq(rperi_rapo_eq,r+eps,rend,rend)
        else: #circular orbit
            rp = r
            ra = r
    else:
        rstart = rp_ra_getstart(r,E,L,params)
        if rstart==0.: rp=0.
        else:
            rp = brentq(rperi_rapo_eq,rstart,rstart/0.9,(E,L,params),maxiter=200)
        rend = rp_ra_getstart(r,E,L,params,apo=True)
        ra = brentq(rperi_rapo_eq,rend/1.1,rend,(E,L,params),maxiter=200)
    return (rp,ra)

def sample_rp_ra_e_distributions(r,vr,chain,thin_by=100,burnin=200):

    """
    sample vr distribution given chain and measurement
    """

    c = gu.reshape_chain(chain)[:,burnin::thin_by,:]
    c = np.reshape(c, (c.shape[0]*c.shape[1],c.shape[2]))
    k = c[:,2]
    samples = c[:,-2:]
    rp,ra,e = np.ones(len(samples))*np.nan,np.ones(len(samples))*np.nan,np.ones(len(samples))*np.nan

    for i,params in enumerate(samples):
        try:
            vesc_Rsun,alpha = params
            vesc = vesc_Rsun*(r/8.5)**(-alpha/2.)
            v = vr + (vesc - vr)*(1. - np.random.power(k[i]+1) )
            vT = np.sqrt(v**2. - vr**2.)
            rp[i],ra[i] = get_rp_ra(r,vr,vT,params)
            e[i] = (ra[i] - rp[i])/(ra[i] + rp[i])
        except:
            pass

    return rp[~np.isnan(rp)],ra[~np.isnan(rp)],e[~np.isnan(rp)]







