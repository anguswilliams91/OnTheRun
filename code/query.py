import sql_utils as sql
import pandas as pd 
import numpy as np 
import gus_utils as gu

def main_sequence_query():

    """
    Retrieve MSTO stars from SDSS DR9 with some quality cuts. Saves the data 
    as a Pandas DataFrame
    """

    getstr = "select spa.ra, spa.dec, spa.dered_u, spa.psfmagerr_u,spa.dered_g,\
     spa.psfmagerr_g,spa.dered_r, spa.psfmagerr_r,spa.dered_i, spa.psfmagerr_i,\
     spa.dered_z, spa.psfmagerr_z,spp.elodiervfinal,spp.elodiervfinalerr,\
     spp.fehadop,spp.fehadopunc,spp.loggadop,spp.loggadopunc,spp.teffadop,spp.teffadopunc, spa.objid \
     from sdssdr9.specphotoall as spa, sdssdr9.sppparams as spp where spp.specobjid=spa.specobjid \
     and spp.scienceprimary=1 and spa.class='STAR' and spa.extinction_r<0.3 and spa.dered_g-spa.dered_r between 0.2 and 0.6 \
     and spa.psfmagerr_g<0.04 and spa.psfmagerr_r<0.04 and spa.psfmagerr_i<0.04 and spp.fehadopunc<0.1\
     and spa.dered_r between 14.5 and 20. and spp.fehadop between -4. and 2. and spp.loggadop between 3.5 and 4. \
     and spp.elodiervfinal between -1000. and 1000. and spp.teffadop between 4800. and 8000. and \
     spa.psfmagerr_g>0. and spa.psfmagerr_r>0. and spa.psfmagerr_i>0. and (spp.zwarning=0 or spp.zwarning=16)\
     and (spp.snr > 20.)"

    res = sql.get(getstr) 
    data = pd.DataFrame(np.array(res).T,columns=['ra','dec','u','u_err','g','g_err','r','r_err','i','i_err',\
                                                    'z','z_err','vhel','vhel_err','feh','feh_err','logg',\
                                                    'logg_err','teff','teff_err', 'objid'])
    data['objid'] = data['objid'].astype(int)
    l,b = gu.radec2galactic(data.ra.values,data.dec.values)
    vgsr = gu.helio2galactic(data.vhel.values,l,b)
    data.loc[:,'l'] = pd.Series(l,index=data.index)
    data.loc[:,'b'] = pd.Series(b,index=data.index)
    data.loc[:,'vgsr'] = pd.Series(vgsr,index=data.index)
    s = gu.Ivesic_estimator(data.g.values,data.r.values,data.i.values,data.feh.values)
    data = data[(np.abs(data.b)>np.radians(20.))&(data.feh<-0.9)&(data.vhel_err<20.)&(s<15.)].reset_index(drop=True)

    data.to_csv("/data/aamw3/SDSS/main_sequence.csv")

    return None

def bhb_query():

    """
    Retrieve BHB stars from SDSS DR9 with some quality cuts. Saves the data 
    as a Pandas DataFrame
    """

    getstr = "select g.ra,g.dec,g.psfmag_g-g.extinction_g,g.psfmag_r-g.extinction_r,g.psfmagerr_g,g.psfmagerr_r,\
              spp.loggadop,spp.fehadop,spp.teffadop,spp.elodiervfinal,spp.elodiervfinalerr from \
              sdssdr9.sppparams as spp, sdssdr9.specphotoall as g where \
              (spp.loggadop between 3. and 3.5) and (spp.teffadop between 8300. and 9300.) and \
              (spp.fehadop between -2. and -1.) and \
              (g.psfmag_g-g.extinction_g-g.psfmag_r+g.extinction_r between -0.25 and 0.) and \
              (g.psfmag_u-g.extinction_u-g.psfmag_g+g.extinction_g between 0.9 and 1.4) and \
              spp.specobjid=g.specobjid and spp.scienceprimary=1 and spp.snr>20. and \
              (spp.zwarning=0 or spp.zwarning=16)"

    res = sql.get(getstr)
    data = pd.DataFrame(np.array(res).T, columns=['ra','dec','g','r','g_err','r_err','logg','feh','teff','vhel',\
                        'vhel_err'])
    l,b = gu.radec2galactic(data.ra.values,data.dec.values)
    vgsr = gu.helio2galactic(data.vhel.values,l,b)
    data.loc[:,'l'] = pd.Series(l,index=data.index)
    data.loc[:,'b'] = pd.Series(b,index=data.index)
    data.loc[:,'vgsr'] = pd.Series(vgsr,index=data.index)
    data = data[(np.abs(data.b)>np.radians(20.))&(data.feh<-0.9)&(data.vhel_err<20.)].reset_index(drop=True)
    data.to_csv("/data/aamw3/SDSS/bhb.csv")

    return None


"""
To make the K giant sample, I downloaded the data from Xue et al. and then cut so that [Fe/H]<-0.9, |b| > 20 degrees and rgc<50kpc. 
"""