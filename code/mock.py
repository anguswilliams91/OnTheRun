from __future__ import division, print_function

import numpy as np, models as m, emcee, gus_utils as gu, warnings


def generate_mock(N,params,vmin=200.):

    vesc_Rsun, alpha, k = params 

    r1 = np.random.normal(loc=10.,scale=1.7,size=np.int(.75*N))
    r2 = np.random.normal(loc=20.,scale=8.,size=np.int(.25*N))
    r = np.clip( np.hstack((r1,r2)), 1., 50.)
    vesc = m.vesc_model(r,0,0,[vesc_Rsun,alpha],"spherical_powerlaw")
    v = np.zeros_like(r)

    for i in np.arange(len(r)):

        v[i] = vmin + (vesc[i] - vmin)*(1. - np.random.power(k+2) )

    v[0] = 700.

    return np.stack((v,r))

def log_likelihood_errorfree(params,data,vmin):

    vesc_Rsun, alpha, k, f = params 
    v,r = data

    vesc = m.vesc_model(r,0.,0.,[vesc_Rsun,alpha],"spherical_powerlaw")
    out = np.zeros_like(v)
    outlier_normalisation = ( .5*m.erfc( vmin / (np.sqrt(2.)*1000.) ) )**-1.
    with warnings.catch_warnings():
        #deliberately getting NaNs here so stop python from telling us about it
        warnings.simplefilter("ignore",category=RuntimeWarning)
        out = (1.-f)*(k+2)*(vesc - v)**(k+1.) / (vesc - vmin)**(k+2.) +\
                f*outlier_normalisation*m.Gaussian(v,0.,1000.)
        out[np.isnan(out)] = f*outlier_normalisation*m.Gaussian(v[np.isnan(out)],0.,1000.)
        ll = np.sum(np.log(out))
    return ll

def log_priors(params,vmin):

    vesc_Rsun, alpha, k, f = params 

    if k<0. or k>10.:

        return -np.inf

    elif vesc_Rsun<vmin:

        return -np.inf

    elif alpha<0. or alpha>1.:

        return -np.inf

    elif f<0. or f>1.:

        return -np.inf

    else:

        return -np.log(vesc_Rsun)

def log_posterior(params,data,vmin):

    lnprior = log_priors(params,vmin)

    if ~np.isfinite(lnprior):
        return lnprior
    else:
        return lnprior + log_likelihood_errorfree(params,data,vmin)

def setup_mcmc(N,params=[530.,0.4,4.],vmin=200.,fname="/data/aamw3/mcmc/escape_chains/mock_run.dat"):

    data = generate_mock(N,params,vmin)

    sampler = emcee.EnsembleSampler(30,4,log_posterior,args=(data,vmin),threads=8)
    guess = [np.random.uniform(low=1.5*np.min(data[0]),high=1.5*params[0]),np.random.uniform(0.1,1.),np.random.uniform(1.,10.),0.1] 
    std = [10.,0.01,0.01,0.001]
    p0 = emcee.utils.sample_ball(guess,std,30)

    gu.write_to_file(sampler,fname,p0,Nsteps=10000)

    return data




