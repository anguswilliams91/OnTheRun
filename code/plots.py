from __future__ import division, print_function

import numpy as np, matplotlib.pyplot as plt, pandas as pd

import plotting as pl, gus_utils as gu, models as m, fits as f, corner_plot as  cp 

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

    ax.plot(mr,mvgsr,'o',c="k",mec='none',ms=3,alpha=0.8,label="MSTO",rasterized=True)
    ax.plot(kr,kvgsr,'o',c="crimson",mec='none',ms=3,alpha=0.8,label="K giant",rasterized=True)
    ax.plot(br,bvgsr,'o',c="royalblue",mec='none',ms=3,alpha=0.8,label="BHB",rasterized=True)

    ax.set_xlabel("$r/\\mathrm{kpc}$")
    ax.set_ylabel("$v_\\mathrm{||}/\\mathrm{kms^{-1}}$")
    ax.legend(loc='upper right',numpoints=1,markerscale=3)

    ax.set_ylim((-700.,1000.))
    ax.set_xlim(np.min(np.hstack((mr,kr,br))),50.)

    return ax


def posterior_predictive_check(chain,model,burnin=200,cmap="Greys",thin_by=10,nbins=[20,20,10]):

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

    #load the data, extract vgsr and r from it
    bhb = pd.read_csv("/data/aamw3/SDSS/bhb.csv")
    kgiant = pd.read_csv("/data/aamw3/SDSS/kgiant.csv")
    msto = pd.read_csv("/data/aamw3/SDSS/main_sequence.csv")
    msto = msto[msto.vgsr!=np.max(msto.vgsr)].reset_index(drop=True)
    data = (msto,kgiant,bhb)
    tracer_title = ["MSTO","K-giant","BHB"]
    colors = ["k","crimson","royalblue"]

    fig,ax = plt.subplots(3,2,figsize=(10.,15.),sharex=True)
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

        for a in ax[i,:]:
            a.plot(v,np.mean(function_samples,axis=1),c='slategray',zorder=0,lw=2)
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

        ax[i,0].errorbar(v_centres,medians,yerr=[medians-lowers,uppers-medians],fmt='o',ecolor=colors[i],ms=4.,mfc=colors[i],mec='none',elinewidth=1)
        ax[i,1].errorbar(v_centres,medians,yerr=[medians-lowers,uppers-medians],fmt='o',ecolor=colors[i],ms=4.,mfc=colors[i],mec='none',elinewidth=1)
        ax[i,1].set_yscale("log")
        ymin,ymax = ax[i,1].get_ylim()
        ax[i,1].set_ylim((1e-6,ymax))
        ax[i,0].text(400.,0.75*ax[i,0].get_ylim()[1],tracer_title[i],fontsize=20)

    fig.text(0.5,0.,"$v_{||}/\\mathrm{kms^{-1}}$")
    fig.text(0.,0.5,"$p(v_{||}\\, | \\, \\mathrm{data})/\\mathrm{km^{-1}s}$",rotation=90)

    return ax

def mass_enclosed(chain,model,burnin=200,cmap="Greys",fontsize=30,tickfontsize=20,**kwargs):

    """
    Plot the mass enclosed implied by a spherically symmetric model given 
    an MCMC chain of parameter samples and radial limits. Median and 68 percent, 
    94 percent credible intervals

    Arguments 
    ---------

    chain: array_like [n_samples,n_parameters]
        MCMC chain of parameter samples 

    model: string 
        model name

    Returns
    -------

    ax: pyplot.axes 
        matplotlib.pyplot axes object  
    """

    G = 43010.795338751527 #in km^2 s^-2 kpc (10^10 Msun)^-1
    n = m.get_numparams(model)
    c = gu.reshape_chain(chain)[:,burnin:,:]
    c = np.reshape(c, (c.shape[0]*c.shape[1],c.shape[2]))
    samples = c[:,-n:].T

    rlims=[0.1,50.]

    if model == "spherical_powerlaw":

        model_name = "SPL"

        def mass_enclosed(r,params):

            vesc_Rsun, alpha = params
            v0 = np.sqrt(.5*alpha)*vesc_Rsun
            return v0**2.*r*(r/8.5)**(-alpha) / G 

    elif model == "TF":

        model_name = "TF"

        def mass_enclosed(r,params):

            v0, rs, alpha = params

            return r * v0**2. * rs**alpha / (G * (rs**2.+r**2.)**(.5*alpha) )

    def rotation_curve(r,params):

        return np.sqrt(mass_enclosed(r,params)*G/r)

    r1 = np.linspace(rlims[0],rlims[1],200)
    r2 = np.linspace(8.5,rlims[1],200)
    width,height = plt.rcParams.get('figure.figsize')
    fig,ax = plt.subplots(1,2,figsize=(2*width,height))

    pl.posterior_1D(samples,r1,mass_enclosed,cmap=cmap,ax=ax[0],fontsize=fontsize,tickfontsize=tickfontsize,**kwargs)
    pl.posterior_1D(samples,r2,rotation_curve,cmap=cmap,ax=ax[1],fontsize=fontsize,tickfontsize=tickfontsize,**kwargs)
    ymin,ymax = ax[0].get_ylim()
    ax[0].text(rlims[0]+.1*(rlims[1]-rlims[0]),ymin+.75*(ymax-ymin),model_name,fontsize=1.5*fontsize)
    ax[0].set_ylabel("$M(r)/\\mathrm{10^{11}M_\\odot}$",fontsize=fontsize)
    ax[0].set_xlabel("$r/\\mathrm{kpc}$",fontsize=fontsize)
    ax[1].set_xlabel("$r/\\mathrm{kpc}$",fontsize=fontsize)
    ax[1].set_ylabel("$v_c(r)/\\mathrm{kms^{-1}}$",fontsize=fontsize)
    ax[1].set_ylim((0.,400.))

    return ax

def plot_tracers(**kwargs):

    """
    Plot the full distributions of vLOS against galactocentric r for 
    our three tracer groups.
    """

    bhb = pd.read_csv("/data/aamw3/SDSS/bhb.csv")
    m.compute_galactocentric_radii(bhb,"bhb",append_dataframe=True)
    kgiant = pd.read_csv("/data/aamw3/SDSS/kgiant.csv")
    m.compute_galactocentric_radii(kgiant,"kgiant",append_dataframe=True)
    msto = pd.read_csv("/data/aamw3/SDSS/main_sequence.csv")
    m.compute_galactocentric_radii(msto,"main_sequence",append_dataframe=True)
    bhb = bhb[(np.abs(bhb.vgsr)>200.)].reset_index(drop=True)
    kgiant = kgiant[(np.abs(kgiant.vgsr)>200.)].reset_index(drop=True)
    msto = msto[(np.abs(msto.vgsr)>200.)].reset_index(drop=True)
    data = (msto,kgiant,bhb)
    tracer_title = ["MSTO","K-giant","BHB"]
    colors = ["k","crimson","royalblue"]
    width,height = plt.rcParams.get('figure.figsize')
    fig,ax = plt.subplots(1,3,figsize=(3*width,height),sharey=True,sharex=True)

    for i,tracer in enumerate(data):

        ax[i].scatter(tracer.rgc.values,np.abs(tracer.vgsr.values),c=colors[i],\
                                                        edgecolors='none',**kwargs)
        
        ax[i].set_xlim((0.,50.))
        xmin,xmax = ax[i].get_xlim()
        ymin,ymax = ax[i].get_ylim()
        ax[i].set_ylim((200.,ymax))
        ax[i].text(xmin+.65*(xmax-xmin),200.+.8*(ymax-200.),tracer_title[i]+"\n $N={}$".format(len(tracer)),fontsize=30)

    
    fig.text(0.5,0.,"$r/\\mathrm{kpc}$")
    fig.text(0.,0.5,"$v_{||}/\\mathrm{kms^{-1}}$",rotation=90)
    fig.subplots_adjust(bottom=0.3,left=0.2)

    return None

def median_vT_plot(fehcut=2.,s_cut=2.,tgas_bins=20,sdss_bins=10):

    #first tgas-rave
    data = pd.read_csv("/data/aamw3/gaia_dr1/tgas_raveon_nodups.csv")
    data = data[data.feh<fehcut]

    l,b = gu.radec2galactic(data.ra.values,data.dec.values)
    vgsr = gu.helio2galactic(data.vhel.values,l,b)

    x,y,z,vx,vy,vz = gu.obs2cartesian(data.pmra.values,data.pmdec.values,data.ra.values,data.dec.values,\
                                                    1./data.parallax.values,data.vhel.values,radec_pms=True)
    vT = np.sqrt(vx**2.+vy**2.+vz**2.-vgsr**2.)

    counts,bin_edges = np.histogram(np.abs(vgsr),tgas_bins)
    bin_centres = np.array([.5*(bin_edges[i] + bin_edges[i+1]) for i in np.arange(tgas_bins)])
    median_vTs = np.zeros(tgas_bins)
    plus_vTs = np.zeros(tgas_bins)
    minus_vTs = np.zeros(tgas_bins)
    for i in np.arange(tgas_bins):
        idx  = (np.abs(vgsr)>bin_edges[i])&(np.abs(vgsr)<bin_edges[i+1])
        if counts[i]>10.:
            median_vTs[i] = np.median(vT[idx])
            plus_vTs[i] = np.percentile(vT[idx],84.) - np.median(vT[idx]) 
            minus_vTs[i] = np.median(vT[idx]) - np.percentile(vT[idx],16.) 

    i1 = counts>10

    #fig,ax = plt.subplots(1,2,figsize=(14.,7.),sharey=True)
    fig,ax = plt.subplots()
    ax.errorbar(bin_centres[i1],median_vTs[i1],yerr=[minus_vTs[i1],plus_vTs[i1]],fmt='-o',label="$v_T$")
    ax.plot(bin_centres[i1],np.sqrt(median_vTs[i1]**2.+bin_centres[i1]**2.),label="$v_\\mathrm{total}$")
    ax.set_ylabel("$v_i/\\mathrm{kms^{-1}}$")
    ax.set_xlabel("$v_{||}/\\mathrm{kms^{-1}}$")
    ax.text(50.,50.,"TGAS-RAVE",fontsize=25)
    ax.legend(loc='best')
    ax.axvline(200.,c='0.5',ls='--',zorder=0.)

    # #now do sdss-gaia
    # ms = f.gaia_crossmatch()
    # ms = ms[(ms.pmra==ms.pmra)&(ms.pmdec==ms.pmdec)]
    # ms.loc[:,'s'] = pd.Series(gu.Ivesic_estimator(ms.g.values,ms.r.values,ms.i.values,ms.feh.values),\
    #                             index=ms.index)
    # if s_cut is not None:
    #     ms = ms[ms.s<s_cut]
    # x,y,z,vx,vy,vz = gu.obs2cartesian(ms.pmra.values,ms.pmdec.values,ms.ra.values,ms.dec.values,\
    #                                     ms.s.values,ms.vhel.values,radec_pms=True)
    # vT = np.sqrt(vx**2.+vy**2.+vz**2.-ms.vgsr.values**2.)
    # counts,bin_edges = np.histogram(np.abs(ms.vgsr.values),sdss_bins)
    # bin_centres = np.array([.5*(bin_edges[i] + bin_edges[i+1]) for i in np.arange(sdss_bins)])
    # median_vTs = np.zeros(sdss_bins)
    # plus_vTs = np.zeros(sdss_bins)
    # minus_vTs = np.zeros(sdss_bins)
    # for i in np.arange(sdss_bins):
    #     idx  = (np.abs(ms.vgsr.values)>bin_edges[i])&(np.abs(ms.vgsr.values)<bin_edges[i+1])
    #     if counts[i]>10.:
    #         median_vTs[i] = np.median(vT[idx])
    #         plus_vTs[i] = np.percentile(vT[idx],84.) - np.median(vT[idx]) 
    #         minus_vTs[i] = np.median(vT[idx]) - np.percentile(vT[idx],16.)    

    # i1 = counts>10

    # ax[1].errorbar(bin_centres[i1],median_vTs[i1],yerr=[minus_vTs[i1],plus_vTs[i1]],fmt='-o')
    # ax[1].set_xlim((0.,300.))
    # ax[1].text(50.,310.,"SDSS-{\\it Gaia}",fontsize=25)
    # ax[1].set_xlabel("$v_{||}/\\mathrm{kms^{-1}}$")
    # fig.subplots_adjust(bottom=0.3)

    return fig,ax








