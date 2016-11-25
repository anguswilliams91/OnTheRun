from __future__ import division, print_function

import numpy as np, models as m, sql_utils as sql, pandas as pd,\
        multiprocessing as mp, gus_utils as gu

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import fixed_quad
from functools import partial
from scipy.special import gamma

def construct_interpolator(data,tracer):

    """
    Construct an interpolator for p(r) of the data

    Arguments
    ---------

    data: DataFrame 
        pandas dataframe with relevant info for the tracer 

    tracer: string 
        name of tracer 

    Returns 
    -------

    endpoints: list
        the centres of the extremal bins 

    spline: InterpolatedUnivariateSpline
        spline object for p(r)

    """

    r = m.compute_galactocentric_radii(data,tracer,append_dataframe=False)
    if tracer=="kgiant" or tracer=="bhb":
        r = r[r<50.]
    elif tracer=="main_sequence":
        r = r[r<20.]
    pdf,bins = np.histogram(r,10,normed=True)
    r_nodes = np.array([.5*(bins[i]+bins[i+1]) for i in np.arange(10)])

    return ([np.min(r_nodes), np.max(r_nodes)],InterpolatedUnivariateSpline(r_nodes,pdf))


def vLOS_probability(v,vmin,k,spline,limits,params,model):

    """Calculate the probability density at a line of sight velocity given a model 
    with a particular set of parameters, and a selection function p(r).

    Arguments
    ---------

    v: float
        line of sight velocity at which to evaluate the probability

    vmin: float
        the minimum line of sight velocity in the sample 

    k: float 
        the power law index of the speed distribution 

    spline: InterpolatedUnivariateSpline
        a spline object for p(r)

    limits: list 
        the upper and lower limits of p(r)

    params: array_like
        model parameters 

    model: string 
        the name of the model

    Returns
    -------

    pdf: float
        the probability density at v

    """

    if v<vmin: return 0.

    def numerator_integrand(r):

        out = np.zeros_like(r)
        vesc = m.vesc_model(r,0.,0.,params,model)
        out[vesc<=v] = 0.
        out[vesc>v] = (m.vesc_model(r[vesc>v],0.,0.,params,model) - v)**(k+1.) * spline(r[vesc>v])
        return out

    numerator = fixed_quad(numerator_integrand,limits[0],limits[1],n=12)[0]

    def denominator_integrand(r):

        return spline(r)*(m.vesc_model(r,0.,0.,params,model) - vmin)**(k+2.) / (k+2.)

    denominator = fixed_quad(denominator_integrand,limits[0],limits[1],n=12)[0]

    return numerator/denominator

def draw_samples(N,vmin,k,spline,limits,params,model):

    """
    Given a model, draw a sample of size N from p(vLOS).

    Arguments
    ---------

    N: int 
        the number of points to draw 

    vmin: float 
        the minimum speed considered 

    k: float 
        the power law index of the speed distribution

    spline: InterpolatedUnivariateSpline 
        spline of p(r) for this tracer 

    limits: list
        the upper and lower limits of the spline 

    params: array_like
        model parameters 

    model: string 
        name of model

    Returns
    -------

    v: array_like
        list of velocities sampled from p(vLOS)
    """

    v = np.linspace(vmin,600.,100)
    pdf = np.array([vLOS_probability(vi,vmin,k,spline,limits,params,model) for vi in v])
    v_spline = InterpolatedUnivariateSpline(v,pdf)
    cdf = np.array([fixed_quad(v_spline,vmin,vi)[0] for vi in v])
    try:
        idx = np.where(np.diff(cdf)<0.)[0][0]+1
    except:
        idx = None
    v,cdf = v[:idx],cdf[:idx]
    inv_cdf = InterpolatedUnivariateSpline(cdf,v)
    u = np.random.uniform(size=N)
    return inv_cdf(u)

def posterior_predictive_check(chain,tracer,model,vmin,nbins=20,thin_by=1,burnin=200):
    
    """
    For every set of parameters in an MCMC chain, generate a mock data set of the same 
    size as the data.

    Arguments
    ---------

    chain: array_like [nsamples,ndim]
        mcmc chain 

    tracer: string 
        type of tracer

    model: string 
        the name of the model 

    vmin: float
        the minimum speed considered 

    nbins: int (=20)
        the number of bins in vLOS to use 

    thin_by: int(=1)
        thin the chains by this factor 

    burnin: int(=200)
        number of steps per walker to burn in 

    Returns
    -------

    bin_centres: array_like
        centres of bins in vLOS 

    counts: array_like
        the number counts of the data in each of the above bins 

    model_counts: array_like[nsamples,nstars]
        the counts generated in each of the above bins in each mock sample
    """

    n = m.get_numparams(model)
    c = gu.reshape_chain(chain)[:,burnin::thin_by,:]
    c = np.reshape(c, (c.shape[0]*c.shape[1],c.shape[2]))
    samples = c[:,-n:]

    if tracer == "main_sequence":
        k = c[:,2]
        data = pd.read_csv("/data/aamw3/SDSS/main_sequence.csv")
        data = data[data.vgsr!=np.max(data.vgsr)].reset_index(drop=True) #remove the one outlier
    elif tracer == "kgiant":
        k = c[:,1]
        data = pd.read_csv("/data/aamw3/SDSS/kgiant.csv")
    else:
        k = c[:,0]        
        data = pd.read_csv("/data/aamw3/SDSS/bhb.csv")       
    lims,spline = construct_interpolator(data,tracer)
    data = data[np.abs(data.vgsr)>vmin].reset_index(drop=True)
    N = len(data)

    counts,bin_edges = np.histogram(np.abs(data.vgsr.values),nbins)
    bin_centres = np.array([.5*(bin_edges[i] + bin_edges[i+1]) for i in np.arange(nbins)])
    model_counts = np.zeros((len(k), nbins))
    for i,theta in enumerate(samples):
        v = draw_samples(N,vmin,k[i],spline,lims,theta,model)
        model_counts[i,:],_ = np.histogram(v,bin_edges)
    return bin_centres,counts,model_counts

def ppc_alltracers(fname,chain,model,vmin,nbins=[20,20,10],thin_by=1,burnin=200):

    """
    generate mock samples for all of our tracer groups. Save all of the information 
    to file.

    Arguments
    ---------

    fname: string 
        name of file to write dictionaries to 

    chain: array_like [nsamples,ndim]
        mcmc chain 

    model: string 
        the name of the model 

    vmin: float
        the minimum speed considered 

    nbins: list(=[20,20,10])
        the number of bins to use for each of the three tracers 

    thin_by: int(=1)
        thin the chains by this factor 

    burnin: int(=200)
        number of steps per walker to burn in 

    """
    
    tracers = ["main_sequence","kgiant","bhb"]

    for i,tracer in enumerate(tracers):
        bin_centres,data_counts,model_counts = posterior_predictive_check(chain,tracer,\
                                                        model,vmin,nbins=nbins[i],thin_by=thin_by,burnin=burnin)
        summary = {'bin_centres': bin_centres, 'data_counts': data_counts, 'model_counts': model_counts}
        np.save(fname+"_"+tracer,summary)

    return None

def posterior_predictive(v,vmin,k_samples,spline,limits,param_samples,model):

    """
    Compute the posterior predictive distribution at v given samples from the posterior 
    from an MCMC.

    Arguments
    ---------

    v: float
        the line-of-sight velocity at which to compute the posterior predictive distribution 

    vmin: float
        cut off speed 

    k_samples: array_like 
        MCMC samples of the slope of the speed distribution 

    spline: InterpolatedUnivariateSpline 
        a spline object for p(r)

    limits: list
        [rmin,rmax] for the spline 

    param_samples: array_like [n_params, n_samples]
        samples of the potential parameters

    model: string 
        name of model 

    """


    return np.mean(np.array([ vLOS_probability(v,vmin,k_samples[i],spline,limits,param_samples[i],model) \
                for i in np.arange(len(k_samples))]))

def posterior_predictive_grid(v_grid,vmin,chain,model,tracer,burnin=200,pool_size=8):

    """
    Compute the posterior predictive distribution given an MCMC chain and a model. Parallelise 
    over a given number of threads to speed up computation.

    Arguments
    ---------

    v_grid: array_like
        an array of speeds at which to evaluate the posterior predictive distribution

    vmin: float 
        the minimum speed considered

    chain: array_like [nsamples,ndim]
        an MCMC chain of model parameters 

    model: string 
        the name of the model 

    tracer: string 
        the type of tracer 

    burnin: int (=200)
        the number of steps per walker to disregard as burn-in 

    pool_size: int (=8)
        the size of the multiprocessing pool over which to distribute computation

    Returns
    -------

    ppd: array_like
        array of the same shape as v_grid, containing the posterior predictive probabilities 
        at each speed in v_grid
    """
    #reshape the chain according to which model we're looking it
    n = m.get_numparams(model)
    c = gu.reshape_chain(chain)[:,burnin:,:]
    c = np.reshape(c, (c.shape[0]*c.shape[1],c.shape[2]))
    samples = c[:,-n:]

    if tracer == "main_sequence":
        k = c[:,2]
        data = pd.read_csv("/data/aamw3/SDSS/main_sequence.csv")
        lims,spline = construct_interpolator(data,"main_sequence")
    elif tracer == "kgiant":
        k = c[:,1]
        data = pd.read_csv("/data/aamw3/SDSS/kgiant.csv")
        lims,spline = construct_interpolator(data,"kgiant")
    elif tracer == "bhb":
        k = c[:,0]
        data = pd.read_csv("/data/aamw3/SDSS/bhb.csv")
        lims,spline = construct_interpolator(data,"bhb")


    parfun = partial(posterior_predictive,vmin=vmin,k_samples=k,spline=spline,limits=lims\
                    ,param_samples=samples,model=model)
    pool = mp.Pool(pool_size)
    output = pool.map(parfun,v_grid)
    pool.close()
    return output

def outlier_probabilities(params, data, vmin, model):

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
    outlier_probabilities = [None,None,None]
    k = [kbhb,kkgiant,kms]
    outlier_normalisation = ( .5*m.erfc( vmin / (np.sqrt(2.)*1000.) ) )**-1.

    for i,tracer in enumerate(data):
        l,b,v,s = tracer
        x,y,z = gu.galactic2cartesian(s,b,l)
        vesc = m.vesc_model(x,y,z,pot_params,model)
        out = np.zeros_like(v)
        with m.warnings.catch_warnings():
            #deliberately getting NaNs here so stop python from telling us about it
            m.warnings.simplefilter("ignore",category=RuntimeWarning)
            out = (1.-f)*(k[i]+2)*(vesc - np.abs(v))**(k[i]+1.) / (vesc - vmin)**(k[i]+2.) + \
                        f*outlier_normalisation*m.Gaussian(np.abs(v),0.,1000.)
            out[np.isnan(out)] = f*outlier_normalisation*m.Gaussian(np.abs(v[np.isnan(out)]),0.,1000.)
            outlier = f*outlier_normalisation*m.Gaussian(np.abs(v),0.,1000.)
        outlier_probabilities[i] = np.mean(outlier,axis=1) / np.mean(out, axis=1)

    return outlier_probabilities

def check_outliers(chain,vmin,model,burnin=200):

    """
    Compute the probabilities that stars are outliers using our MCMC chains. We are 
    being lazy and not marginalising over the posterior because this is a quick check.
    """

    res = gu.ChainResults(chain,burnin=200)[:,0]
    n = m.get_numparams(model)
    data = m.sample_distances_multiple_tracers(n_samples=200,vmin=vmin)
    return outlier_probabilities(res,data,vmin,model)


def gaia_crossmatch():
    """
    Cross-match our MS targets to TGAS and check that they have small tangential motions
    """

    ms = pd.read_csv("/data/aamw3/SDSS/main_sequence.csv")

    query_str = "select ss.pmra_new,ss.pmdec_new from mytable as t\
                left join lateral (select g.pmra_new,g.pmdec_new \
                from gaia_dr1_aux.gaia_source_sdssdr9_xm_new as g \
                where g.objid=t.objid order by g.dist \
                asc limit 1) as ss on true"

    pmra,pmdec = sql.local_join(query_str,'mytable',(ms.objid.values,),('objid',))

    ms.loc[:,'pmra'] = pd.Series(pmra,index=ms.index)
    ms.loc[:,'pmdec'] = pd.Series(pmdec,index=ms.index)

    return ms


def main():

    fname = "/data/aamw3/SDSS/model_comparison"
    chain = np.genfromtxt("/data/aamw3/mcmc/escape_chains/spherical_powerlaw.dat")
    ppc_alltracers(fname,chain,"spherical_powerlaw",200.,nbins=[20,20,10],thin_by=1,burnin=200)

if __name__ == "__main__":
    main()