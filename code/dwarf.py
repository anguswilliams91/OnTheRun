from __future__ import division, print_function 

import numpy as np, gus_utils as gu
from scipy.optimize import brentq

def Phi(r,params):

    """
    SPL potential.

    Arguments
    ---------

    r: array_like
        radii at which to evaluate the potential 

    params: list 
        [vesc(Rsun), alpha] the potential parameters
    """

    vesc_Rsun,alpha = params
    return -.5*vesc_Rsun**2.*(r/8.5)**-alpha

def rperi_rapo_eq(r,E,L,params):

    """equation for finding rp, ra

    Arguments
    ---------

    r: array_like
        radii at which to evalute the equation

    E: float 
        the energy of the orbit 

    L: float 
        the angular momentum of the orbit 

    params: list 
        the potential parameters

    Returns
    -------

    eq: array_like
        values of the LHS of the equation at r

    """

    return E  - .5*L**2. / r**2. - Phi(r,params)

def rp_ra_getstart(r,E,L,params,apo=False):
    """Find suitable place to start the root finding

    Arguments
    ---------

    r: array_like
        radii at which to evalute the equation

    E: float 
        the energy of the orbit 

    L: float 
        the angular momentum of the orbit 

    params: list 
        the potential parameters

    Returns
    -------

    rtry: float
        a good place to start using a root finder

    """
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

    """Find the limiting radii of the orbit

    Arguments
    ---------

    r: float 
        the current radius of the dwarf 

    vr: float 
        the current radial velocity of the dwarf 

    vT: float 
        the current tangential velocity of the dwarf

    params: list 
        the potential parameters 
    """

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
    sample rp,ra,epsilon distribution given chain and measurement

    Arguments
    ---------

    r: float 
        galactocentric radius of the dwarf

    vr: float 
        the radial velocity of the dwarf 

    chain: array_like
        mcmc chain of the potential parameters and k 

    thin_by: (=100) int 
        the thinning factor for the mcmc chain 

    burnin: (=200) int
        the number of steps per walker to discard for burn in 

    Returns
    -------

    rp: array_like
        posterior samples on rp 

    ra: array_like
        posterior samples on ra

    e: array_like
        posterior samples on the eccentricity
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

def test_rpra_solver(params):

    """
    Given some potential parameters, test the peri and apo solver 
    by passing it orbits with known rp, ra.

    Arguments
    ---------

    params: list
        potential parameters [vesc_Rsun, alphas]

    Returns
    -------

    delta_rp: array_like 
        the differences between the true and solver pericentres

    delta_ra: array_like 
        the differences between the true and solver apocentres
    """

    ra = np.random.uniform(1.,1000.,size=100)
    rp_frac = np.random.uniform(size=100)
    rp = rp_frac*ra
    E = -(rp**2.*Phi(rp,params) - ra**2.*Phi(ra,params))/(ra**2. - rp**2.)
    L = np.sqrt(2.*(Phi(ra,params) - Phi(rp,params))/(rp**-2. - ra**-2.))
    dr = ra-rp 
    f = np.random.uniform(size=100)
    rs = rp + f*dr
    vT = L/rs 
    vr = np.sqrt(2.*(E - Phi(rs,params)) - vT**2.)
    rp_s,ra_s = np.zeros(100), np.zeros(100)
    for i in np.arange(100):
        rp_s[i],ra_s[i] = get_rp_ra(rs[i],vr[i],vT[i],params)
        
    return rp - rp_s, ra - ra_s

