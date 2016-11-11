from __future__ import division, print_function

import numpy as np, matplotlib.pyplot as plt, pandas as pd

import plotting as pl, gus_utils as gu, models as m, fits as f

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


def v_distribution_plot(chain,model,burnin=200,cmap="Greys",thin_by=10,nbins=[20,20,10]):

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
    major_formatter = pl.FuncFormatter(pl.my_formatter)

    #reshape the chain according to which model we're looking it
    n = m.get_numparams(model)
    c = gu.reshape_chain(chain)[:,burnin::thin_by,:]
    c = np.reshape(c, (c.shape[0]*c.shape[1],c.shape[2]))
    samples = c[:,-n:]

    #get the samples for k
    k_bhb = c[:,0]
    k_kgiant = c[:,1]
    k_ms = c[:,2]
    k = [k_ms,k_kgiant,k_bhb]
    tracer_names = ["main_sequence","kgiant","bhb" ]

    bhb = pd.read_csv("/data/aamw3/SDSS/bhb.csv")
    kgiant = pd.read_csv("/data/aamw3/SDSS/kgiant.csv")
    msto = pd.read_csv("/data/aamw3/SDSS/main_sequence.csv")
    msto = msto[msto.vgsr!=np.max(msto.vgsr)].reset_index(drop=True)
    data = (msto,kgiant,bhb)

    fig,ax = plt.subplots(3,2,figsize=(10.,15.))
    for a in ax.ravel():
        a.yaxis.set_major_formatter(major_formatter)
    cm = plt.cm.get_cmap(cmap)

    v = np.linspace(200.,550.,200)
    n_samples = len(samples)
    n_v = len(v)

    for i in np.arange(len(data)):
        lims,spline = f.construct_interpolator(data[i],tracer_names[i])
        function_samples = np.array([f.compute_posterior(vi,200.,k[i][j],spline,lims\
                                        ,samples[j],model) for vi in v for j in \
                                        np.arange(n_samples)]).reshape((n_v,n_samples))

        confs = np.zeros((n_v,5))
        for j in np.arange(n_v):
            confs[j,0] = np.percentile(function_samples[j],3)
            confs[j,1] = np.percentile(function_samples[j],16)
            confs[j,2] = np.percentile(function_samples[j],50)
            confs[j,3] = np.percentile(function_samples[j],84)
            confs[j,4] = np.percentile(function_samples[j],97)

        for a in ax[i,:]:
            a.plot(v,confs[:,2],c=cm(1.))
            a.fill_between(v,confs[:,1],confs[:,3],facecolor=cm(0.25),lw=0)
            a.fill_between(v,confs[:,3],confs[:,4],facecolor=cm(0.75),lw=0)
            a.fill_between(v,confs[:,0],confs[:,1],facecolor=cm(0.75),lw=0)
            a.set_xlim(np.min(v),np.max(v))

    for i, tracer in enumerate(data):
        tracer = tracer[np.abs(tracer.vgsr)>200.].reset_index(drop=True)
        counts,bins = np.histogram(np.abs(tracer.vgsr.values),nbins[i])
        v_centres = np.array([.5*(bins[j]+bins[j+1]) for j in np.arange(nbins[i])])
        nf = (len(tracer)*(bins[1]-bins[0]))**-1.
        lowers,medians,uppers = np.zeros_like(v_centres),np.zeros_like(v_centres),np.zeros_like(v_centres)

        for j in np.arange(nbins[i]):
            lowers[j] = gammaincinv(counts[j]+1.,0.1)*nf
            medians[j] = gammaincinv(counts[j]+1.,0.5)*nf
            uppers[j] = gammaincinv(counts[j]+1.,0.9)*nf

        ax[i,0].errorbar(v_centres,medians,yerr=[medians-lowers,uppers-medians],fmt='o',c='k')
        ax[i,1].errorbar(v_centres,medians,yerr=[medians-lowers,uppers-medians],fmt='o',c='k')
        ax[i,1].set_yscale("log")
        ax[i,1].set_xscale("log")

    fig.text(0.5,0.,"$v_{||}/\\mathrm{kms^{-1}}$")
    fig.text(0.,0.5,"$p(v_{||})/\\mathrm{km^{-1}s}$",rotation=90)

    return ax