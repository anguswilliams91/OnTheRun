from __future__ import division, print_function

import numpy as np, pandas as pd , gus_utils as gu, emcee, warnings, sys

from scipy.optimize import minimize
from scipy.special import hyp2f1, erf

def sample_distances(data,n_samples=10000,tracer='main_sequence'):

    """Given a Pandas dataframe, compute distance samples and return a numpy array with distance samples and 
    clones of other measurements for convenience.

    Arguments
    ---------

    data: pandas.DataFrame
        a DataFrame object with columns for g,r,i band magnitudes and [Fe/H] + their uncertainties.

    n_samples: (=10000) int
        the number of Monte-Carlo samples to draw from the uncertainties

    Returns
    -------

    samples: tuple
        tuple of arrays (l,b,v,s) each of shape [n_stars,n_samples] 
        containing n_samples of distances estimates for each star and 
        copies of (l,b,v) for convenience.

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
            distances[i] = gu.Ivesic_estimator(g_samples,r_samples,i_samples,feh_samples)
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

def sample_distances_multiple_tracers(n_samples=1000):

    """
    Sample from the uncertainties on distance for all three data sets.

    Arguments
    ---------

    n_samples: int
        the number of samples to draw from the uncertainties per star.


    Returns
    -------

    data: list 
        [bhb,kgiant,ms], a list of tuples (l,b,v,s), each tuple contains arrays of shape [n_stars,n_samples] 
        containing n_samples of distance estimates for each star and copies of the galactic coordinates and 
        galactocentric line of sight velocities. 
    """

    bhb = pd.read_csv("/data/aamw3/SDSS/bhb.csv")
    kgiant = pd.read_csv("/data/aamw3/SDSS/kgiant.csv")
    ms = pd.read_csv("/data/aamw3/SDSS/main_sequence.csv")

    # #try a different circular speed
    # bhb.vgsr = gu.helio2galactic(bhb.vhel,bhb.l,bhb.b,vcirc=220.)
    # kgiant.vgsr = gu.helio2galactic(kgiant.vhel,kgiant.l,kgiant.b,vcirc=220.)
    # ms.vgsr = gu.helio2galactic(ms.vhel,ms.l,ms.b,vcirc=220.)

    bhb = bhb[np.abs(bhb.vgsr)>200.].reset_index(drop=True)
    kgiant = kgiant[np.abs(kgiant.vgsr)>200.].reset_index(drop=True)
    ms = ms[np.abs(ms.vgsr)>200.].reset_index(drop=True)


    bhb_s = sample_distances(bhb,n_samples=n_samples,tracer='bhb')
    kgiant_s = sample_distances(kgiant,n_samples=n_samples,tracer='kgiant')
    ms_s = sample_distances(ms,n_samples=n_samples,tracer='main_sequence')

    return [bhb_s, kgiant_s, ms_s]

def compute_galactocentric_radii(data,tracer,append_dataframe=True):

    """
    Compute galactocentric radius for a given tracer.

    Arguments
    ---------

    data: DataFrame
        pandas dataframe containing the relevant information for this tracer

    tracer: string 
        tracer type

    append_dataframe: bool 
        if True, append a column called 'rgc' to the DataFrame, if False then 
        return the radii explicitly

    Returns
    -------

    r: array_like
        list of galactocentric radii (if append_dataframe is False)
    """

    if tracer == "main_sequence":

        s = gu.Ivesic_estimator(data.g.values,data.r.values,data.i.values,data.feh.values)

    elif tracer == "bhb":

        s = gu.BHB_distance(data.g.values,data.r.values)

    elif tracer == "kgiant":

        if append_dataframe is True:
            print("Radii should be included in K giant DataFrame already.")
            return None
        else: return data.rgc.values

    x,y,z = gu.galactic2cartesian(s,data.b.values,data.l.values)
    r = np.sqrt(x**2.+y**2.+z**2.)

    if append_dataframe is True: 

        data.loc[:,'rgc'] = pd.Series(r, index=data.index)

    else:

        return r


def Gaussian(v,mu,sigma):

    """Gaussian distribution.

    Arguments
    ---------

    v: array_like
        the independent variable.

    mu: float 
        the mean of the distribution 

    sigma: float 
        the dispersion of the distribution

    Returns
    -------

    pdf: array_like
        the pdf at each value of v
    """

    return (2.*np.pi*sigma**2.)**-.5 * np.exp(-.5 * (v-mu)**2. / sigma**2.)

def log_likelihood(params, data, vmin, model):

    """
    Likelihood function template for given model. 

    Arguments
    ---------

    params: array_like
        the model parameters

    data: list
        the output of sample_distances_multiple_tracers

    vmin: float 
        the minimum radial velocity considered 

    Returns
    -------

    logL: array_like
        the sum of the log-likelihoods for this set of parameters.
    """

    kbhb,kkgiant,kms,f = params[:4]
    pot_params = params[4:]
    tracer_likelihoods = np.zeros(3)
    k = [kbhb,kkgiant,kms]
    outlier_normalisation = 2.37676

    for i,tracer in enumerate(data):
        try:
            l,b,v,s = tracer
            x,y,z = gu.galactic2cartesian(s,b,l)
            vesc = vesc_model(x,y,z,pot_params,model)
            out = np.zeros_like(v)
            with warnings.catch_warnings():
                #deliberately getting NaNs here so stop python from telling us about it
                warnings.simplefilter("ignore",category=RuntimeWarning)
                out = (1.-f)*(k[i]+2)*(vesc - np.abs(v))**(k[i]+1.) / (vesc - vmin)**(k[i]+2.) + \
                        f*outlier_normalisation*Gaussian(np.abs(v),0.,1000.)
                out[np.isnan(out)] = f*outlier_normalisation*Gaussian(np.abs(v[np.isnan(out)]),0.,1000.)
            tracer_likelihoods[i] = np.sum( np.log(np.mean(out, axis=1) ) )
        except:
            pass

    return np.sum(tracer_likelihoods)


def TFpotential(r,params):

    """
    TF model from Gibbons et al. 2014

    Arguments
    ---------

    r: array_like
        spherical radius in kpc

    params: array_like
        model parameters (v0, rs, alpha)

    Returns
    -------

    Phi: array_like
        the potential at r

    """

    v0,rs,alpha = params

    prefactor = -(r**-(alpha+2.) * rs**alpha * (r**2. + rs**2.)**(-.5*alpha)\
                * v0**2.) / (alpha*(2. + alpha))
    term_1 = r**(2.+alpha) * (2. + alpha)
    term_2 = alpha * rs**2. * (r**2. + rs**2.)**(.5*alpha) * hyp2f1(.5*(alpha+2),\
             .5*(alpha+2), .5*(alpha+4), -(rs/r)**2.)

    return prefactor*(term_1 + term_2)


def vesc_model(x,y,z,params,model):

    """
    A set of models for the escape velocity.

    Arguments
    ---------

    x,y,z: array_like:
        Galactic cartesian coordinates 

    params: array_like:
        Model parameters

    model: string 
        name of model being used

    Returns
    -------

    vesc: array_like
        escape velocity in this model at (x,y,z)

    """

    if model == "spherical_powerlaw":

        r = np.sqrt(x**2.+y**2.+z**2.)
        vesc_Rsun, alpha = params 
        return vesc_Rsun * (r/8.5)**(-.5*alpha)

    elif model == "flattened_powerlaw":

        vesc_Rsun, alpha, q = params 
        rq = np.sqrt(x**2. + y**2. + (z/q)**2.)
        return vesc_Rsun * (rq/8.5)**(-.5*alpha) 

    elif model == "TF":

        r = np.sqrt(x**2.+y**2.+z**2.)
        return np.sqrt(-2.*TFpotential(r,params))

    else: 

        raise ValueError(model+" not found.")

    return None

def get_numparams(model):

    """Get the number of parameters associated with a model

    Arguments
    ---------

    model: string
        the name of the model 

    Returns
    -------

    n_params: int 
        the number of parameters in the model

    """

    if model == "spherical_powerlaw":

        return 2

    elif model == "flattened_powerlaw" or model == "TF":

        return 3

    else: 

        raise ValueError(model+" not found.")

    return None


def log_priors_global(params):

    """Priors on parameters common to all models.

    Arguments
    ---------

    params: array_like:
        parameters common to all models (k values and outlier fraction)

    Returns
    -------

    log_prior: float
        the log of the prior on the parameters

    """

    kbhb,kkgiant,kms,f = params 
    k = np.array([kbhb,kkgiant,kms])

    if any(k<0.) or any(k>10.):
        #cosmological sim priors on K from Smith et al.
        return -np.inf
    elif f<0. or f>1.:
        return -np.inf
    else:
        return 0.    

def log_priors_model(params,vmin,model):

    """
    Model-specific priors.

    Arguments
    ---------

    params: array_like:
        model parameters

    vmin: float
        the minimum radial velocity considered

    model: string
        model name

    Returns
    -------

    log_prior: float
        the log of the prior on the parameters.
    """

    if model == "spherical_powerlaw":

        vesc_Rsun, alpha = params 
        
        if vesc_Rsun<vmin or vesc_Rsun>1000.:
            return -np.inf
        elif alpha<0. or alpha>1.:
            return -np.inf
        else:
            return -np.log(vesc_Rsun) 

    elif model == "flattened_powerlaw":

        vesc_Rsun, alpha, q = params

        if vesc_Rsun<vmin or vesc_Rsun>1000.:
            return -np.inf
        elif alpha<0. or alpha>1.:
            return -np.inf
        elif q<0. or q>4.:
            return -np.inf
        else:
            #prior from Bowden, Evans, Williams on q
            return -np.log(vesc_Rsun) \
                    -np.log(1. + q**2.)

    elif model == "TF":

        v0, rs, alpha = params

        if alpha<0. or alpha>1.:
            return -np.inf
        elif v0<150. or v0>400.:
            return -np.inf
        elif rs<0. or rs>100.:
            return -np.inf
        elif vesc_model(50.,0.,0.,params,"TF")<vmin:
            return -np.inf 
        else:
            return -np.log(v0) \
                    -.5*(rs-15.)**2./7**2.

    else: 

        raise ValueError(model+" not found.")

    return None

def sample_priors_model(model,n_walkers):

    """
    Sample from the prior on a model to use as 
    starting points for optimizations.

    Arguments
    ---------

    model: string
        the name of the model 

    n_walkers: int 
        the number of walkers (draws)

    Returns
    -------

    samples: array_like
        samples from the model priors

    """

    if model == "spherical_powerlaw":

        #vesc_Rsun_samples = np.random.normal(loc=533.,scale=25.,size=n_walkers)
        vesc_Rsun_samples = np.random.uniform(low=np.log(200.),high=np.log(1000.),size=n_walkers)
        alpha_samples = np.random.uniform(low=0., high=1., size=n_walkers)

        return np.vstack((np.exp(vesc_Rsun_samples),alpha_samples)).T

    elif model == "flattened_powerlaw":

        vesc_Rsun_samples = np.random.uniform(low=np.log(200.),high=np.log(1000.),size=n_walkers)
        alpha_samples = np.random.uniform(low=0., high=1., size=n_walkers)
        #sample flattening in variable where prior is uniform then transform to q
        u_samples = np.random.uniform(low=0., high=(2./np.pi)*np.arctan(4.), size=n_walkers)
        q_samples = np.tan(np.pi*u_samples/2.)

        return np.vstack((np.exp(vesc_Rsun_samples),alpha_samples,q_samples)).T

    elif model == "TF":

        v0_samples = np.random.uniform(low=np.log(200.),high=np.log(300.),size=n_walkers)
        rs_samples = np.clip(np.random.normal(loc=15.,scale=7.,size=n_walkers),0.,np.inf)
        alpha_samples = np.random.uniform(low=0.,high=1.,size=n_walkers)

        return np.vstack((np.exp(v0_samples),rs_samples,alpha_samples)).T

    else: 

        raise ValueError(model+" not found.")

    return None

def sample_priors_global(n_walkers):

    """
    Sample priors on global parameters

    Arguments
    ---------

    n_walkers: int
        the number of walkers (draws)

    Returns
    -------

    samples: array_like 
        samples from the prior

    """

    kBHB = np.random.uniform(low=0., high=10., size=n_walkers)
    kkgiant = np.random.uniform(low=0., high=10., size=n_walkers)
    kms = np.random.uniform(low=0., high=10., size=n_walkers)
    f = np.random.uniform(low=0.,high=1.,size=n_walkers)

    return np.vstack((kBHB,kkgiant,kms,f)).T



def log_posterior(params, data, vmin, model):

    """
    Likelihood function template for given model. 

    Arguments
    ---------

    params: array_like
        the model parameters

    data: list
        the output of sample_distances_multiple_tracers

    vmin: float 
        the minimum radial velocity considered 

    Returns
    -------

    logP: float
        the log-posterior for this set of parameters.
    """

    params_global = params[:4]
    params_model = params[4:]
    logprior = log_priors_global(params_global)+log_priors_model(params_model,vmin,model)

    if logprior == -np.inf:
        return logprior
    else:
        return log_likelihood(params,data,vmin,model)+logprior


def run_mcmc(model,filename,vmin=200.,n_walkers=80,n_steps=3000,n_threads=20,n_samples=200,seed=0,tracer=None):

    """
    Set up and run an MCMC for a given model. The chain will run and 
    write the parameters at each step to a file.

    Arguments
    ---------

    model: string
        the name of the model being fit

    filename: string
        path to file where the chain will be written

    vmin: (=200) float
        the minimum radial velocity considered

    n_walkers: (=80) int
        the number of emcee walkers to use

    n_steps: (=3000) int 
        the number of steps each walker will take 

    n_threads: (=20) int 
        the number of threads over which to parallelise computation

    n_samples: (=200) int
        the number of samples to draw from the uncertainties on the distances 

    """

    np.random.seed(seed)

    data = sample_distances_multiple_tracers(n_samples=n_samples)
    #sometimes we want to test with a subset of the tracers
    if tracer == "kgiant_only":
        data[0] = None 
        data[2] = None
    elif tracer == "main_sequence_only":
        data[0] = None
        data[1] = None

    #run a minimization for each walker to get starting points
    n_params = 4 + get_numparams(model)
    p0 = np.zeros((n_walkers,n_params))
    prior_samples_model = sample_priors_model(model,n_walkers)
    prior_samples_global = sample_priors_global(n_walkers)
    p0 = np.hstack((prior_samples_global,prior_samples_model))

    sampler = emcee.EnsembleSampler(n_walkers,n_params,log_posterior,args=(data,vmin,model),threads=n_threads)

    gu.write_to_file(sampler,filename,p0,Nsteps=n_steps)

    return None

def main():

    model, filename, tracer, nwalkers, nsteps, nthreads  = sys.argv[1:]

    nwalkers,nsteps,nthreads = int(nwalkers),int(nsteps),int(nthreads)

    run_mcmc(model,filename,tracer=tracer,n_walkers=nwalkers,n_steps=nsteps,n_threads=nthreads)


if __name__ == "__main__":
    main()
