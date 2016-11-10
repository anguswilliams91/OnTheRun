from __future__ import division, print_function

import numpy as np, matplotlib.pyplot as plt, pandas as pd

import plotting as pl, gus_utils as gu, models as m

from scipy.special import gammaincinv

def Vesc_posterior(chain,model,burnin=200,pos_cmap="Greys",dat_cmap="Blues"):

    """
    Plot the posterior of Vesc(r) as a function of r using 
    Monte Carlo samples of the parameters.

    Arguments
    ---------

    chain: array_like
        MCMC chain of parameters 

    model: string 
        name of model to be plotted 

    burnin: (=200) int 
        number of steps to discard as burn-in


    Returns
    -------

    ax: matplotlib axes 
        axes object with plot 

    """

    #reshape the chain according to which model we're looking it
    n = m.get_numparams(model)
    c = gu.reshape_chain(chain)[:,burnin:,-n:]
    samples = np.reshape(c, (c.shape[0]*c.shape[1],n)).T

    #load the data, extract vgsr and r from it
    bhb = pd.read_csv("/data/aamw3/SDSS/bhb.csv")
    kgiant = pd.read_csv("/data/aamw3/SDSS/kgiant.csv")
    msto = pd.read_csv("/data/aamw3/SDSS/main_sequence.csv")

    bhb_dist = gu.BHB_distance(bhb.g.values,bhb.r.values,feh=bhb.feh.values)
    bx,by,bz = gu.galactic2cartesian(bhb_dist,bhb.b.values,bhb.l.values)
    br = np.sqrt(bx**2.+by**2.+bz**2.)
    bvgsr = bhb.vgsr.values[(br<50.)&np.isfinite(br)]
    br = br[(br<50.)&np.isfinite(br)]


    msto_dist = gu.Ivesic_estimator(msto.g.values,msto.r.values,msto.i.values,msto.feh.values)
    mx,my,mz = gu.galactic2cartesian(msto_dist,msto.b.values,msto.l.values)
    mr = np.sqrt(mx**2.+my**2.+mz**2.)
    mvgsr = msto.vgsr.values[(mr<50.)&np.isfinite(mr)]
    mr = mr[(mr<50.)&np.isfinite(mr)]

    kr = kgiant.rgc.values 
    kvgsr = kgiant.vgsr.values

    #plot the data and the posterior
    def vesc(r,params):
        return m.vesc_model(r,0.,0.,params,model)

    def minus_vesc(r,params):
        return -m.vesc_model(r,0.,0.,params,model)

    fig,ax = plt.subplots()
    r = np.linspace(4.,50.,1000)
    pl.posterior_1D(samples,r,vesc,cmap=pos_cmap,ax=ax)
    pl.posterior_1D(samples,r,minus_vesc,cmap=pos_cmap,ax=ax)

    # ax.plot(mr,mvgsr,'o',c='k',mec='none',ms=5,alpha=0.8,label="MSTO",rasterized=True)
    # ax.plot(kr,kvgsr,'o',c='r',mec='none',ms=5,alpha=0.8,label="K giant",rasterized=True)
    # ax.plot(br,bvgsr,'o',c='b',mec='none',ms=5,alpha=0.8,label="BHB",rasterized=True)
    pl.kde_smooth(np.hstack((mr,kr,br)),np.hstack((mvgsr,kvgsr,bvgsr)),scatter_outside=True,cmap=dat_cmap,ax=ax,markersize=4,\
            fill=True)

    ax.set_xlabel("$r/\\mathrm{kpc}$")
    ax.set_ylabel("$v_\\mathrm{||}/\\mathrm{kms^{-1}}$")

    ax.set_ylim((-700.,700.))
    ax.set_xlim(np.min(np.hstack((mr,kr,br))),50.)

    return ax


def v_distribution_plot(chain,model,tracer,burnin=200,cmap="Greys"):

    """
    Plot the inferred line of sight velocity distribution for stars, normalised 
    to the escape velocity at their position.

    Arguments
    ---------
    chain: array_like
        MCMC chain of parameters 

    model: string 
        name of model to be plotted 

    burnin: (=200) int 
        number of steps to discard as burn-in

    Returns
    -------

    ax: matplotlib axes 
        axes object with plot 

    """

    n = m.get_numparams(model)
    params = gu.ChainResults(chain,burnin=200)[-n:,0] #median potential parameters

    #get the samples for each k
    c = gu.reshape_chain(chain)[burnin:,:3]
    samples = np.reshape(c, (c.shape[0]*c.shape[1],n)).T

    if tracer=="bhb":
        data = pd.read_csv("/data/aamw3/SDSS/bhb.csv")
    elif tracer=="kgiant":
        data = pd.read_csv("/data/aamw3/SDSS/kgiant.csv")
    elif tracer=="main_sequence":
        data = pd.read_csv("/data/aamw3/SDSS/main_sequence.csv")

    data = data[np.abs(data.vgsr)>200.].reset_index(drop=True)

    if tracer=="bhb":
        dist = gu.BHB_distance(bhb.g.values,bhb.r.values,feh=bhb.feh.values)
    elif tracer=="kgiant":
        dist = kgiant.s.values
    elif tracer=="main_sequence":
        dist = gu.Ivesic_estimator(msto.g.values,msto.r.values,msto.i.values,msto.feh.values)
    idx = np.isfinite(dist)&(dist<50.)
    data,dist = data[idx].reset_index(drop=True),dist[idx]
    x,y,z = gu.galactic2cartesian(dist,data.b.values,data.l.values)
    r = np.sqrt(x**2.+y**2.+z**2.)[idx]
    vesc = m.vesc_model(x[idx],y[idx],z[idx],params,model)
    v_norm = np.abs(bhb.vgsr.values[idx])/bvesc


    return 