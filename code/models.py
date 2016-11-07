from __future__ import division, print_function

import numpy as np, pandas as pd , gus_utils as gu, emcee

def Ivesic_estimator(g,r,i,feh):

    """distance estimator from Ivesic et al. (2008)

    Arguments
    ---------

    g: array_like
        SDSS g band de-reddened apparent magnitude
    r: array_like
        SDSS r band de-reddened apparent magnitude
    i: array_like
        SDSS i band de-reddened apparent magnitude
    feh: array_like
        SDSS SSPP [Fe/H] estimate

    Returns
    -------

    distance: array_like
        distances assuming the Ivesic estimator
    """

    delta_Mr = 4.50 - 1.11*feh - 0.18*feh**2.
    x = g-i
    Mr0 = -5.06 + 14.32*x - 12.97*x**2. + 6.127*x**3. - 1.267*x**4. + 0.0967*x**5. 
    Mr = Mr0 + delta_Mr 

    mu = r - Mr 
    return 10.**(mu/5. - 2.)

def sample_distances(data,n_samples=10000,tracer='main_sequence'):

    """Given a Pandas dataframe, compute distance samples and add new column to the DataFrame.

    Arguments
    ---------

    data: pandas.DataFrame
        a DataFrame object with columns for g,r,i band magnitudes and [Fe/H] + their uncertainties.

    n_samples: (=10000) int
        the number of Monte-Carlo samples to draw from the uncertainties

    Returns
    -------

    distances: array_like[n_stars,n_samples]
        an array containing n_samples distances estimates for each star.

    """

    n_stars = len(data)
    distances = np.zeros((n_stars,n_samples))
    l_oversampled = np.zeros((n_stars,n_samples))
    b_oversampled = np.zeros((n_stars,n_samples))
    v_oversampled = np.zeros((n_stars,n_samples))

    for i in np.arange(n_stars):

        if tracer=='main_sequence':
            g_samples = np.random.normal(loc=data.g[i],scale=data.g_err[i],size=n_samples)
            r_samples = np.random.normal(loc=data.r[i],scale=data.r_err[i],size=n_samples)
            i_samples = np.random.normal(loc=data.i[i],scale=data.i_err[i],size=n_samples)
            feh_samples = np.random.normal(loc=data.feh[i],scale=data.feh_err[i],size=n_samples)
            distances[i] = Ivesic_estimator(g_samples,r_samples,i_samples,feh_samples)
        if tracer=='bhb':
            g_samples = np.random.normal(loc=data.g[i],scale=data.g_err[i],size=n_samples)
            r_samples = np.random.normal(loc=data.r[i],scale=data.r_err[i],size=n_samples)
            distances[i] = gu.BHB_distance(g_samples,r_samples)
        if tracer=='kgiant':
            dm_samples = np.random.normal(loc=data.dm[i],scale=data.dm_err[i],size=n_samples)
            distances[i] = 10.**(.2*dm_samples - 2.)


        l_oversampled[i] = data.l[i]*np.ones(n_samples)
        b_oversampled[i] = data.b[i]*np.ones(n_samples)
        v_oversampled[i] = data.vgsr[i]*np.ones(n_samples)

    median_distances = np.median(distances, axis=1)
    uncertainty_distances = np.std(distances,axis=1)
    frac_err = uncertainty_distances/median_distances
    if tracer=='main_sequence':
        accept = (frac_err<0.2)&(median_distances<15.)
    elif tracer=='bhb':
        accept = (frac_err<0.3)&(median_distances<50.)
    elif tracer=='kgiant':
        accept = (frac_err<0.3)&(median_distances<50.)
    l_oversampled,b_oversampled,v_oversampled,distances = l_oversampled[accept],b_oversampled[accept],\
                                              v_oversampled[accept],distances[accept]

    return (l_oversampled,b_oversampled,v_oversampled,distances)

def sample_distances_multiple_tracers():

    bhb = pd.read_csv("/data/aamw3/SDSS/bhb.csv")
    kgiant = pd.read_csv("/data/aamw3/SDSS/kgiant.csv")
    ms = pd.read_csv("/data/aamw3/SDSS/main_sequence.csv")

    bhb_s = sample_distances(bhb,n_samples=1000,tracer='bhb')
    kgiant_s = sample_distances(kgiant,n_samples=1000,tracer='kgiant')
    ms_s = sample_distances(ms,n_samples=1000,tracer='main_sequence')

    return [bhb_s, kgiant_s, ms_s]

def Gaussian(v,mu,sigma):
    return (2.*np.pi*sigma**2.)**-.5 * np.exp(-.5 * (v-mu)**2. / sigma**2.)

def likelihood_constant(params,data,vmin):

    """Likelihood function for the data given a power-law model where vesc is fixed"""

    vesc,k,f = params

    out = np.zeros_like(data.vgsr.values)
    out[data.vgsr.values<vesc] = f*Gaussian(data.vgsr.values[data.vgsr.values<vesc],0.,1000.) + \
                                (1.-f)*(vesc - np.abs(data.vgsr.values[data.vgsr.values<vesc]))**(k+1) * (k+2) / (vesc - vmin)**(k+2)
    out[data.vgsr.values>vesc] = f*Gaussian(data.vgsr.values[data.vgsr.values>vesc],0.,1000.)

    return np.sum( np.log(out) )

def likelihood_powerlaw(params,data,vmin):

    """Likelihood function when the escape speed is a power law in radius"""

    vesc_8,alpha,k,f = params

    l,b,v,s = data 
    x,y,z = gu.galactic2cartesian(s,b,l)
    r = np.sqrt(x**2.+y**2.+z**2.)

    vesc = vesc_8*(r/8.)**(.5*alpha)

    out = np.zeros_like(v)
    out = (1.-f)*(k+2)*(vesc - np.abs(v))**(k+1.) / (vesc - vmin)**(k+2.) + f*Gaussian(v,0.,1000.)
    out[np.isnan(out)] = f*Gaussian(v[np.isnan(out)],0.,1000.)
    out = np.mean(out, axis=1)

    return np.sum( np.log(out) )

def priors_powerlaw(params,vmin):

    vesc_8,alpha,k,f = params

    if vesc_8<vmin or vesc_8>1000.:
        return -np.inf
    if alpha < -1. or alpha>0.:
        return -np.inf
    if k<2.7 or k>4.7:
        return -np.inf
    if f<0. or f>1.:
        return -np.inf
    else:
        return -.5*(vesc_8 - 533.)**2. / 50.**2.

def log_posterior_powerlaw(params,data,vmin):

    pr = priors_powerlaw(params,vmin)

    if pr == -np.inf:
        return -np.inf
    else:
        return likelihood_powerlaw(params,data,vmin)

def likelihood_powerlaw_multiple_tracers(params,data,vmin):

    bhb,kgiant,ms = data
    vesc_8,alpha,kbhb,kkgiant,kms,f = params 

    return likelihood_powerlaw([vesc_8,alpha,kbhb,f],bhb,vmin) +\
           likelihood_powerlaw([vesc_8,alpha,kkgiant,f],kgiant,vmin) + \
           likelihood_powerlaw([vesc_8,alpha,kms,f],ms,vmin)

def priors_powerlaw_multiple_tracers(params,vmin):

    vesc_8,alpha,kbhb,kkgiant,kms,f = params

    k = np.array([kbhb,kkgiant,kms])

    if vesc_8<vmin or vesc_8>1000.:
        return -np.inf
    if alpha < -1. or alpha>0.:
        return -np.inf
    if any(k<2.7) or any(k>4.7):
        return -np.inf
    if f<0. or f>1.:
        return -np.inf
    else:
        return -.5*(vesc_8 - 533.)**2. / 50.**2.

def log_posterior_powerlaw_multiple_tracers(params,data,vmin):

    pr = priors_powerlaw_multiple_tracers(params,vmin)

    if pr == -np.inf:
        return -np.inf
    else:
        return likelihood_powerlaw_multiple_tracers(params,data,vmin)

def mcmc_multiple_tracers():
    data = sample_distances_multiple_tracers()
    params = [500., -0.25,3.5,3.5,3.5,0.1]
    std = [10.,0.01,0.1,0.1,0.1,0.01]
    p0 = emcee.utils.sample_ball(params,std,size=80)
    sampler = emcee.EnsembleSampler(80,6,log_posterior_powerlaw_multiple_tracers,args=(data,200.),threads=20)
    gu.write_to_file(sampler,"/data/aamw3/mcmc/escape_chains/ms_powerlaw_pifflprior_multiple_tracers.dat",p0,Nsteps=10000)  

def main():

    # data = pd.read_csv("/data/aamw3/SDSS/main_sequence.csv")
    # samples = sample_distances(data,n_samples=1000)
    # params = [500., -0.25,3.5,0.1]
    # std = [10.,0.01,0.1,0.01]
    # p0 = emcee.utils.sample_ball(params,std,size=32)
    # sampler = emcee.EnsembleSampler(32,3,log_posterior_powerlaw,args=(samples,200.),threads=16)
    # gu.write_to_file(sampler,"/data/aamw3/mcmc/escape_chains/ms_powerlaw_pifflprior_200cut.dat",p0,Nsteps=10000)
    mcmc_multiple_tracers()


if __name__ == "__main__":
    main()









