from __future__ import division, print_function

import numpy as np, matplotlib.pyplot as plt, pandas as pd

import plotting as pl, gus_utils as gu, models as m, fits as f, corner_plot as  cp,\
        matplotlib as mpl

from scipy.special import gammaincinv
from palettable.colorbrewer.qualitative import Set1_6

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


def posterior_predictive_check(chain,model,burnin=200,cmap="Greys",thin_by=10,nbins=[20,20,10],pool_size=8):

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

    v = np.linspace(200.,550.,100)
    for i in np.arange(len(data)):
        ppd = f.posterior_predictive_grid(v,200.,chain,model,tracer_names[i],burnin=200,pool_size=pool_size)

        for a in ax[i,:]:
            a.plot(v,ppd,c='slategray',zorder=0,lw=2)
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

    return fig,ax

def mass_enclosed(chain,model,burnin=200,cmap="Blues",fontsize=30,tickfontsize=20,thin_by=1,**kwargs):

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
    c = gu.reshape_chain(chain)[:,burnin::thin_by,:]
    c = np.reshape(c, (c.shape[0]*c.shape[1],c.shape[2]))
    samples = c[:,-n:].T

    rlims=[0.1,70.]

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

    pl.posterior_1D(samples,r1,mass_enclosed,cmap=cmap,ax=ax[0],tickfontsize="small",fontsize=mpl.rcParams['font.size'],**kwargs)
    pl.posterior_1D(samples,r2,rotation_curve,cmap=cmap,ax=ax[1],tickfontsize="small",fontsize=mpl.rcParams['font.size'],**kwargs)
    ymin,ymax = ax[0].get_ylim()
    ax[0].set_ylabel("$M(r)/\\mathrm{10^{11}M_\\odot}$")
    ax[0].set_xlabel("$r/\\mathrm{kpc}$")
    ax[1].set_xlabel("$r/\\mathrm{kpc}$")
    ax[1].set_ylabel("$v_c(r)/\\mathrm{kms^{-1}}$")
    ax[1].set_ylim((100.,350.))

    colors = (Set1_6.mpl_colors[0],Set1_6.mpl_colors[2],Set1_6.mpl_colors[3],Set1_6.mpl_colors[4],'k')

    #now plot other people's masses...
    ax[0].errorbar( 60.0, 40.0, yerr = 7.0, fmt='o', label = "X08", c=colors[0], ms=9, mec='none', zorder=100,elinewidth=2) #Xue 2008
    ax[0].errorbar( 50.0 + 1, 54.0, yerr = [[36.0],[2.0]], fmt='^', label = "W99", c=colors[1], ms=9, mec='none', zorder=100,elinewidth=2) #Wilkinson 1999
    ax[0].errorbar( 50.0 - 1, 42.0, yerr = 4.0, fmt='v', label = "D12", c=colors[2] ,ms=9, mec='none', zorder=100,elinewidth=2) #Deason Broken Degneracies
    ax[0].errorbar(50.0, 44.8, yerr=[[1.4],[1.5]], fmt='d', label = "WE15", c=colors[3], ms=9, mec='none', zorder=100,elinewidth=2) #me and wyn 2015
    ax[0].errorbar(50.0, 29., yerr=[[5.],[5.]], fmt='s', label = "G14", c=colors[4], ms=9, mec='none', zorder=100,elinewidth=2) #simon 2014
    ax[0].legend(loc='best',numpoints=1)

    #...and circular speeds
    circspeed = lambda mass,radius: np.sqrt(mass*G/radius)
    ax[1].errorbar( 60.0, circspeed(40.,60.), yerr = [[circspeed(40.,60.)-circspeed(40.-7.,60.)],[circspeed(40.+7.,60.)-circspeed(40.,60.)]],\
                             fmt='o', label = "X08", c=colors[0], ms=9, mec='none', zorder=100,elinewidth=2) #Xue 2008
    ax[1].errorbar( 50.0 + 1, circspeed(54.,50.), yerr = [[circspeed(54.,50.)-circspeed(53.-36.,50.)],[circspeed(54+2.,50.)-circspeed(54.,50.)]],\
                             fmt='^', label = "W99", c=colors[1], ms=9, mec='none', zorder=100,elinewidth=2) #Wilkinson 1999
    ax[1].errorbar( 50.0 - 1, circspeed(42.,50.), yerr = [[circspeed(42.,50.)-circspeed(42.-4.,50.)],[circspeed(42+4.,50.)-circspeed(42.,50.)]]\
                            , fmt='v', label = "D12", c=colors[2], ms=9, mec='none', zorder=100,elinewidth=2) #Deason Broken Degneracies
    ax[1].errorbar(50.0, 198.2, yerr=[[3.2],[3.4]]\
                            , fmt='d', label = "WE15", c=colors[3], ms=9, mec='none', zorder=100,elinewidth=2) #me and wyn 2015
    ax[1].errorbar(50.0, circspeed(29.,50.), yerr=[[circspeed(29.,50.)-circspeed(29.-5.,50.)],[circspeed(29+5.,50.)-circspeed(29.,50.)]]\
                            , fmt='s', label = "G14", c=colors[4], ms=9, mec='none', zorder=100,elinewidth=2) #simon 2014


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
        ax[i].tick_params(labelsize=20)

    
    ax[1].set_xlabel("$r/\\mathrm{kpc}$",fontsize=25)
    ax[0].set_ylabel("$v_{||}/\\mathrm{kms^{-1}}$",fontsize=25)
    fig.subplots_adjust(bottom=0.3,left=0.2)

    return None

def Vesc_posterior(burnin=200):

    chain = np.genfromtxt("/data/aamw3/mcmc/escape_chains/spherical_powerlaw.dat")

    fig,ax = plt.subplots()
    r = np.linspace(4.,50.,200)

    n = m.get_numparams("spherical_powerlaw")
    c = gu.reshape_chain(chain)[:,burnin:,:]
    c = np.reshape(c, (c.shape[0]*c.shape[1],c.shape[2]))
    samples = c[:,-n:].T

    def vesc(r,params):
        return m.vesc_model(r,0.,0.,params,"spherical_powerlaw")

    pl.posterior_1D(samples,r,vesc,cmap="Blues",ax=ax,tickfontsize="small",fontsize=mpl.rcParams['font.size'])

    ax.set_xlabel("$r/\\mathrm{kpc}$")
    ax.set_ylabel("$v_\\mathrm{esc}(r)/\\mathrm{kms^{-1}}$")

    ax.errorbar(8.5,533.,yerr=[[41.],[54.]],fmt='o',c='k',markersize=10.,zorder=100000)
    ax.text(8.9,600.,"P14",fontsize=25)

    return fig,ax

def main():

    filename, model, pool_size  = m.sys.argv[1:]
    pool_size = np.int(pool_size)
    chain = np.genfromtxt(filename)
    fig,ax = posterior_predictive_check(chain,model,burnin=200,cmap="Greys",thin_by=10,nbins=[20,20,10],pool_size=pool_size)
    fig.savefig("/data/aamw3/OnTheRunWriteup/plots/ppc.pdf")

if __name__ == "__main__":
    main()










