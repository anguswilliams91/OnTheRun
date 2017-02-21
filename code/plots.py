from __future__ import division, print_function

import numpy as np, matplotlib.pyplot as plt, pandas as pd

import plotting as pl, gus_utils as gu, models as m, fits as f, corner_plot as  cp,\
        matplotlib as mpl, dwarf as d, matplotlib.ticker as ticker,matplotlib.gridspec as gridspec

from scipy.special import gammaincinv
from palettable.colorbrewer.qualitative import Set1_6
from matplotlib.patches import Ellipse

def posterior_predictive_distribution(chain,model,burnin=200,cmap="Greys",thin_by=10,nbins=[20,20,10],pool_size=8):

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

def posterior_predictive_checks():

    """
    Graphical check of the model using posterior predictive samples.
    """

    main_sequence = np.load("/data/aamw3/SDSS/model_comparison_main_sequence.npy").item()
    kgiant = np.load("/data/aamw3/SDSS/model_comparison_kgiant.npy").item()
    bhb = np.load("/data/aamw3/SDSS/model_comparison_bhb.npy").item()

    tracers = (main_sequence,kgiant,bhb)
    tracer_title = ["MSTO","K-giant","BHB"]
    cmaps = ["YlOrBr","Reds","Blues"]

    fig,ax = plt.subplots(3,2,figsize=(10.,15.))

    for i,data in enumerate(tracers):
        binwidth = np.diff(data['bin_centres'])[0]
        for axi in ax[i]:
            # axi.boxplot(data['model_counts'],showfliers=False,boxprops=dict(c='0.5',linewidth=2),\
            #     whiskerprops=dict(c='0.5',linewidth=2,linestyle='-'),medianprops=dict(c='0.5',linewidth=2),\
            #     capprops=dict(c='0.5',linewidth=2),positions=data['bin_centres'],whis=[0.,100.],widths=.5*binwidth)
            cm = plt.cm.get_cmap(cmaps[i])
            p0 = np.percentile(data['model_counts'],0,axis=0)
            p2p5 = np.percentile(data['model_counts'],2.5,axis=0)
            p16 = np.percentile(data['model_counts'],16,axis=0)
            p50 = np.median(data['model_counts'],axis=0)
            p84 = np.percentile(data['model_counts'],84,axis=0)
            p97p5 = np.percentile(data['model_counts'],97.5,axis=0)
            p100 = np.percentile(data['model_counts'],100,axis=0)
            axi.plot(data['bin_centres'],p50,c=cm(1.))
            axi.fill_between(data['bin_centres'],p16,p84,facecolor=cm(0.7),lw=.1)
            axi.fill_between(data['bin_centres'],p2p5,p16,facecolor=cm(0.45),lw=.1)
            axi.fill_between(data['bin_centres'],p84,p97p5,facecolor=cm(0.45),lw=.1)
            axi.fill_between(data['bin_centres'],p0,p2p5,facecolor=cm(0.2),lw=.1)
            axi.fill_between(data['bin_centres'],p97p5,p100,facecolor=cm(0.2),lw=.1)
            axi.plot(data['bin_centres'],data['data_counts'],'o',mfc='k',mec='none',label="Data")
            axi.set_xlim((data['bin_centres'][0]-2.,data['bin_centres'][-1]+2.))
            axi.xaxis.set_major_locator(ticker.MaxNLocator(5))

    for i,axi in enumerate(ax[:,0]):
        ymin,ymax = axi.get_ylim()
        xmin,xmax = axi.get_xlim()
        axi.text(xmin+0.6*(xmax-xmin),ymin+0.7*(ymax-ymin),tracer_title[i],fontsize=30)

    for i, axi in enumerate(ax[:,1]):
        ymin,ymax = axi.get_ylim()
        axi.set_yscale("log")
        axi.set_ylim((10.**-1.,ymax))


    fig.text(0.5,0.,"$v_{||}/\\mathrm{kms^{-1}}$",fontsize=20)
    fig.text(0.,0.5,"$N$",fontsize=20)
    fig.subplots_adjust(bottom=0.3,left=0.3)

    return None


def mass_enclosed(chain,model,burnin=200,cmap="Blues",thin_by=1,**kwargs):

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

    burnin: (=200) int 
        the number of steps per walker to discard as burn in

    cmap: (="Blues") string 
        matpotlib colormap to use

    thin_by: (=1) int 
        thinning factor for MCMC chains

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
    ax[0].set_ylabel("$M(r)/\\mathrm{10^{10}M_\\odot}$")
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

def plot_tracers(mark_outliers=False,**kwargs):

    """
    Plot the full distributions of vLOS against galactocentric r for 
    our three tracer groups.
    """

    bhb = pd.read_csv("/data/aamw3/SDSS/bhb.csv")
    m.compute_galactocentric_radii(bhb,"bhb",append_dataframe=True)
    kgiant = pd.read_csv("/data/aamw3/SDSS/kgiant.csv")
    m.compute_galactocentric_radii(kgiant,"kgiant",append_dataframe=True)
    msto = pd.read_csv("/data/aamw3/SDSS/main_sequence_paper.csv")
    m.compute_galactocentric_radii(msto,"main_sequence",append_dataframe=True)
    bhb = bhb[(np.abs(bhb.vgsr)>200.)].reset_index(drop=True)
    kgiant = kgiant[(np.abs(kgiant.vgsr)>200.)].reset_index(drop=True)
    msto = msto[(np.abs(msto.vgsr)>200.)].reset_index(drop=True)
    data = (msto,kgiant,bhb)
    tracer_title = ["MSTO","K-giant","BHB"]
    colors = ["darkorange","crimson","royalblue"]
    width,height = plt.rcParams.get('figure.figsize')
    fig = plt.figure(figsize=(2*width,2*height))
    gs = gridspec.GridSpec(2,4)
    ax = [fig.add_subplot(gs[0,:2]),fig.add_subplot(gs[0,2:]),fig.add_subplot(gs[1,1:3])]
    # fig,ax = plt.subplots(1,3,figsize=(3*width,height),sharey=True,sharex=True)

    for i,tracer in enumerate(data):

        ax[i].scatter(tracer.rgc.values,np.abs(tracer.vgsr.values),c=colors[i],\
                                                        edgecolors='none',**kwargs)
        
        ax[i].set_xlim((0.,50.))
        ax[i].set_ylim((200.,550.))
        xmin,xmax = ax[i].get_xlim()
        ymin,ymax = ax[i].get_ylim()
        ax[i].text(xmin+.65*(xmax-xmin),200.+.8*(ymax-200.),tracer_title[i]+"\n $N={}$".format(len(tracer)),fontsize=30)
        ax[i].tick_params(labelsize=20)
        ax[i].set_xlabel("$r/\\mathrm{kpc}$",fontsize=25)

    ax[1].tick_params(labelleft='off')
    ax[0].set_ylabel("$v_{||}/\\mathrm{kms^{-1}}$",fontsize=25)
    ax[2].set_ylabel("$v_{||}/\\mathrm{kms^{-1}}$",fontsize=25)
    fig.subplots_adjust(bottom=0.3,left=0.2)

    if mark_outliers:
        sids = np.array([611488477824968704,687975180215019520,814161184382019584,1154154111483013120,1220628099725551616\
        ,1511087739091576832,2473759385139046400,2507504500953081856,2837293955551356928,2982662855519660032,3027555364468975616])
        msto_outliers = msto[msto.specobjid.isin(sids)]
        ax[0].scatter(msto_outliers.rgc.values,np.abs(msto_outliers.vgsr),c='k',edgecolors='none',s=50)


    return None

def Vesc_posterior(chain,model,burnin=200):

    """
    Plot the posterior distribution on the escape speed as a function of radius.

    Arguments
    ---------

    chain: array_like[nsamples,ndims]
        mcmc output

    model: string 
        the name of the model to plot

    burnin: int(=200)
        number of steps from the default chain to disregard

    """

    fig,ax = plt.subplots()
    r = np.linspace(4.,50.,200)

    n = m.get_numparams(model)
    c = gu.reshape_chain(chain)[:,burnin:,:]
    c = np.reshape(c, (c.shape[0]*c.shape[1],c.shape[2]))
    samples = c[:,-n:].T

    def vesc(r,params):
        return m.vesc_model(r,0.,0.,params,model)

    pl.posterior_1D(samples,r,vesc,cmap="Blues",ax=ax,tickfontsize="small",fontsize=mpl.rcParams['font.size'])

    ax.set_xlabel("$r/\\mathrm{kpc}$")
    ax.set_ylabel("$v_\\mathrm{esc}(r)/\\mathrm{kms^{-1}}$")

    ax.errorbar(8.5,533.,yerr=[[41.],[54.]],fmt='o',c='k',markersize=10.,zorder=100000)
    ax.text(8.9,600.,"P14",fontsize=25)

    return fig,ax

def halo_distribution(chain,model,burnin=200):

    """
    Plot the mother sample line-of-sight velocity distributions and our inference on the escape speed. 

    Arguments
    ---------

    chain: array_like[nsamples,ndims]
        mcmc output

    model: string 
        the name of the model to plot

    burnin: int(=200)
        number of steps from the default chain to disregard

    """

    fig,ax = plt.subplots()
    msto_data = pd.read_csv("/data/aamw3/SDSS/main_sequence.csv")
    kgiant_data = pd.read_csv("/data/aamw3/SDSS/kgiant.csv")
    bhb_data = pd.read_csv("/data/aamw3/SDSS/bhb.csv")
    s_MSTO = gu.Ivesic_estimator(msto_data.g.values,msto_data.r.values\
                                    ,msto_data.i.values,msto_data.feh.values)
    x_MSTO,y_MSTO,z_MSTO = gu.galactic2cartesian(s_MSTO,msto_data.b.values,msto_data.l.values)
    r_MSTO = np.sqrt(x_MSTO**2.+y_MSTO**2.+z_MSTO**2.)
    s_BHB = gu.BHB_distance(bhb_data.g.values,bhb_data.r.values)
    x_BHB,y_BHB,z_BHB = gu.galactic2cartesian(s_BHB,bhb_data.b.values,bhb_data.l.values)
    r_BHB = np.sqrt(x_BHB**2.+y_BHB**2.+z_BHB**2.)
    r_halo = np.hstack((r_BHB,kgiant_data.rgc.values,r_MSTO))
    v_halo = np.hstack((bhb_data.vgsr.values,kgiant_data.vgsr.values,msto_data.vgsr.values))
    r = np.linspace(1.,50.,100)

    n = m.get_numparams(model)
    c = gu.reshape_chain(chain)[:,burnin:,:]
    c = np.reshape(c, (c.shape[0]*c.shape[1],c.shape[2]))
    samples = c[:,-n:].T

    def vesc(r,params):
        return m.vesc_model(r,0.,0.,params,model)
    def m_vesc(r,params):
        return -m.vesc_model(r,0.,0.,params,model)

    pl.posterior_1D(samples,r,vesc,cmap="Blues",ax=ax,tickfontsize="small",fontsize=mpl.rcParams['font.size'])
    pl.posterior_1D(samples,r,m_vesc,cmap="Blues",ax=ax,tickfontsize="small",fontsize=mpl.rcParams['font.size'])
    ax.plot(r_halo,v_halo,'o',ms=5,c='0.3',mec='none')
    ax.set_ylabel("$v_{||}/\\mathrm{kms^{-1}}$")
    ax.set_xlabel("$r/\\mathrm{kpc}$")
    ax.set_ylim((-600.,600.))
    ax.axhline(-200.,ls='--',c='k')
    ax.axhline(+200.,ls='--',c='k')


    return fig,ax

def dwarf_galaxies(chain,model,burnin=200):

    """
    Plot the dwarf galaxy line of sight velocities multiplied by sqrt(3) and our inference on the escape speed.

    Arguments
    ---------

    chain: array_like[nsamples,ndims]
        mcmc output

    model: string 
        the name of the model to plot

    burnin: int(=200)
        number of steps from the default chain to disregard

    """

    fig,ax = plt.subplots()
    dwarf_data = pd.read_csv("/data/aamw3/satellites/r_vgsr_dwarfs.csv")
    r = np.linspace(np.min(dwarf_data['r']),np.max(dwarf_data['r'])+10,300)

    n = m.get_numparams(model)
    c = gu.reshape_chain(chain)[:,burnin:,:]
    c = np.reshape(c, (c.shape[0]*c.shape[1],c.shape[2]))
    samples = c[:,-n:].T

    def vesc(r,params):
        return m.vesc_model(r,0.,0.,params,model)
    def m_vesc(r,params):
        return -m.vesc_model(r,0.,0.,params,model)

    pl.posterior_1D(samples,r,vesc,cmap="Blues",ax=ax,tickfontsize="small",fontsize=mpl.rcParams['font.size'])
    pl.posterior_1D(samples,r,m_vesc,cmap="Blues",ax=ax,tickfontsize="small",fontsize=mpl.rcParams['font.size'])

    ax.plot(dwarf_data['r'],np.sqrt(3.)*dwarf_data['vgsr'],'o',mec='none',ms=10,c='0.5')
    ax.plot(53,-np.sqrt(3.)*211.,'o',ms=10,c='y',mec='none')
    ax.plot(116.,-np.sqrt(3.)*189,'o',ms=10,c='y',mec='none')
    ax.plot(46.,np.sqrt(3.)*244,'o',ms=10,c='r',mec='none')
    ax.plot(37.,-np.sqrt(3.)*247,'o',ms=10,c='r',mec='none')
    ax.plot(126.,np.sqrt(3.)*154.3,'o',ms=10,c='r',mec='none')
    ax.annotate("Tuc 2", xy=(53,-np.sqrt(3.)*211.),xytext=(75.,-485.),arrowprops=dict(facecolor='black',width=1,shrink=0.15)) #tuc2
    ax.annotate("Gru 1", xy=(116.,-np.sqrt(3.)*189),xytext=(130,-465.),arrowprops=dict(facecolor='black',width=1,shrink=0.15)) #gru1
    ax.annotate("Boo III", xy=(46.,np.sqrt(3.)*244),xytext=(70,480.),arrowprops=dict(facecolor='black',width=1,shrink=0.15)) #booIII
    ax.annotate("Tri II", xy=(37.,-np.sqrt(3.)*247),xytext=(55.,-564),arrowprops=dict(facecolor='black',width=1,shrink=0.15)) #triII
    ax.annotate("Herc", xy=(126.,np.sqrt(3.)*154.3),xytext=(140.,440.),arrowprops=dict(facecolor='black',width=1,shrink=0.15)) #herc

    ax.set_ylabel("$\sqrt{3}\,v_{||}/\\mathrm{kms^{-1}}$")
    ax.set_xlabel("$r/\\mathrm{kpc}$")


    return fig,ax


def dwarf_posteriors(chain,burnin=200):

    """
    Plot posteriors on rp,ra,e for BooIII, TriII and Herc

    Arguments
    ---------

    chain: array_like[nsamples,ndims]
        mcmc output from a spherical_powerlaw chain

    burnin: int(=200)
        number of steps from the default chain to disregard

    """

    data = pd.read_csv("/data/aamw3/satellites/r_vgsr_dwarfs.csv")
    names = ["BootesIII","TriangulumII","Hercules"]
    plot_names = ["Bootes III", "Triangulum II", "Hercules"]
    fig,ax = plt.subplots(1,3,figsize=(15,5))

    for a in ax.ravel():
        a.yaxis.set_visible(False)

    for i,name in enumerate(names):
        r = data.r[data.name==name].values[0]
        vgsr = np.abs(data.vgsr[data.name==name].values[0])
        rp,ra,e = d.sample_rp_ra_e_distributions(r,vgsr,chain,thin_by=1,burnin=burnin)
        ax[0].hist(rp,100,histtype="step",range=(0.,100.),color=Set1_6.mpl_colors[i],normed=True,label=plot_names[i])
        ax[1].hist(ra,100,histtype="step",range=(0.,800.),color=Set1_6.mpl_colors[i],normed=True,label=plot_names[i])
        ax[2].hist(e,100,histtype="step",range=(0.5,1.),color=Set1_6.mpl_colors[i],normed=True,label=plot_names[i])

    ax[0].set_xlim((0.,100.))
    ax[1].set_xlim((0.,800.))
    ax[2].set_xlim((0.5,1.))
    ax[0].set_xlabel("$r_\\mathrm{peri}/\\mathrm{kpc}$",fontsize=20)
    ax[1].set_xlabel("$r_\\mathrm{apo}/\\mathrm{kpc}$",fontsize=20)
    ax[2].set_xlabel("$\\epsilon$",fontsize=20)
    ax[2].legend(loc='upper left',fontsize=25)

    return fig,ax

def corner_plots():

    """
    Make the corner plots in the paper
    """

    chain = np.genfromtxt("/data/aamw3/mcmc/escape_chains/spherical_powerlaw.dat")
    chain = chain[80*200:,[3,2,1,4,5,6]]
    chain[:,-3] *= 1960.
    axis_labels = ["$k_\\mathrm{MSTO}$","$k_\\mathrm{K-giant}$","$k_\\mathrm{BHB}$","$N_*f$","$v_\\mathrm{esc}(R_\\odot)/\\mathrm{kms^{-1}}$","$\\alpha$"]
    cp.corner_plot(chain,axis_labels=axis_labels,fontsize=15,tickfontsize=12,figsize=(10.,10.))
    plt.gcf().text(0.75,0.75,"SPL",fontsize=30)

    chain = np.genfromtxt("/data/aamw3/mcmc/escape_chains/flattened_powerlaw.dat")
    chain = chain[80*200:,-3:]
    axis_labels = ["$v_\\mathrm{esc}(R_\\odot)/\\mathrm{kms^{-1}}$","$\\beta$","$q$"]
    cp.corner_plot(chain,axis_labels=axis_labels)
    plt.gcf().text(0.75,0.75,"EPL",fontsize=30)

    chain = np.genfromtxt("/data/aamw3/mcmc/escape_chains/TF.dat")
    chain = chain[80*200:,-3:]
    axis_labels = ["$v_0/\\mathrm{kms^{-1}}$","$r_s/\\mathrm{kpc}$","$\\gamma$"]
    cp.corner_plot(chain,axis_labels=axis_labels)
    plt.gcf().text(0.75,0.75,"TF",fontsize=30)

    chain = np.genfromtxt("/data/aamw3/mcmc/escape_chains/spherical_powerlaw.dat")
    chain_1 = np.genfromtxt("/data/aamw3/mcmc/escape_chains/spherical_powerlaw_nogal.dat")
    chain = chain[80*200:,[3,2,1,4,5,6]]
    chain_1 = chain_1[80*200:,[3,2,1,4,5,6]]
    axis_labels = ["$k_\\mathrm{MSTO}$","$k_\\mathrm{K-giant}$","$k_\\mathrm{BHB}$","$N_*f$","$v_\\mathrm{esc}(R_\\odot)/\\mathrm{kms^{-1}}$","$\\alpha$"]
    chain_labels = ["Main analysis", "Extra morphological cut"]
    cp.multi_corner_plot((chain,chain_1),axis_labels=axis_labels,chain_labels=chain_labels,fontsize=15,tickfontsize=12,figsize=(10.,10.),linecolors=['k','r'],\
                            linewidth=2.)

    return None


def main():

    filename, model, pool_size  = m.sys.argv[1:]
    pool_size = np.int(pool_size)
    chain = np.genfromtxt(filename)
    fig,ax = posterior_predictive_check(chain,model,burnin=200,cmap="Greys",thin_by=10,nbins=[20,20,10],pool_size=pool_size)
    fig.savefig("/data/aamw3/OnTheRunWriteup/plots/ppc.pdf")

if __name__ == "__main__":
    main()










