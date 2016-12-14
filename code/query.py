import sql_utils as sql
import pandas as pd 
import numpy as np 
import gus_utils as gu

def main_sequence_query():

    """
    Retrieve MSTO stars from SDSS DR9 with some quality cuts. Saves the data 
    as a Pandas DataFrame
    """

    getstr = "SELECT spa.ra, spa.dec, spa.dered_u, spa.psfmagerr_u,spa.dered_g,\
    spa.psfmagerr_g,spa.dered_r, spa.psfmagerr_r,spa.dered_i, spa.psfmagerr_i,\
    spa.dered_z, spa.psfmagerr_z,spp.elodiervfinal,spp.elodiervfinalerr,\
    spp.fehadop,spp.fehadopunc,spp.loggadop,spp.loggadopunc,spp.teffadop,spp.teffadopunc, spa.objid \
    \
    FROM sdssdr9.specphotoall AS spa,\
         sdssdr9.sppparams AS spp \
    \
    WHERE spp.specobjid=spa.specobjid \
    AND spp.scienceprimary=1 \
    AND spa.class='STAR' \
    AND spa.extinction_r<0.3\
    AND spa.dered_g-spa.dered_r BETWEEN 0.2 AND 0.6 \
    AND spa.dered_r BETWEEN 14.5 AND 20. \
    AND spp.fehadop BETWEEN -4. AND -0.9 \
    AND spp.loggadop BETWEEN 3.5 AND 4. \
    AND spp.teffadop BETWEEN 4500. AND 8000.\
    AND spa.psfmagerr_g BETWEEN 0. AND 0.04 \
    AND spa.psfmagerr_r BETWEEN 0. AND 0.04 \
    AND spa.psfmagerr_i BETWEEN 0. AND 0.04 \
    AND spp.fehadopunc < 0.1 \
    AND (spp.zwarning=0 OR spp.zwarning=16) \
    AND spp.snr > 20."

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
    data = data[(np.abs(data.b)>np.radians(20.))&(data.feh<-0.9)&(s<15.)].reset_index(drop=True)

    data.to_csv("/data/aamw3/SDSS/main_sequence.csv")

    return None

def bhb_query():

    """
    Retrieve BHB stars from SDSS DR9 with some quality cuts. Saves the data 
    as a Pandas DataFrame
    """

    getstr = "SELECT spa.ra,spa.dec,spa.psfmag_g-spa.extinction_g,spa.psfmag_r-spa.extinction_r,spa.psfmagerr_g,spa.psfmagerr_r, \
    spp.loggadop,spp.fehadop,spp.teffadop,spp.elodiervfinal,spp.elodiervfinalerr \
    \
    FROM sdssdr9.specphotoall AS spa, \
    sdssdr9.sppparams AS spp \
    \
    WHERE spp.specobjid=spa.specobjid \
    AND spp.scienceprimary=1 \
    AND spa.class='STAR' \
    AND spa.psfmag_g-spa.extinction_g-spa.psfmag_r \
        +spa.extinction_r BETWEEN -0.25 AND 0. \
    AND spa.psfmag_u-spa.extinction_u-spa.psfmag_g \
        +spa.extinction_g BETWEEN 0.9 AND 1.4 \
    AND spp.fehadop BETWEEN -2. AND -1. \
    AND spp.loggadop BETWEEN 3. AND 3.5 \
    AND spp.teffadop BETWEEN 8300. AND 9300. \
    AND (spp.zwarning=0 OR spp.zwarning=16) \
    AND spp.snr>20."

    res = sql.get(getstr)
    data = pd.DataFrame(np.array(res).T, columns=['ra','dec','g','r','g_err','r_err','logg','feh','teff','vhel',\
                        'vhel_err'])
    l,b = gu.radec2galactic(data.ra.values,data.dec.values)
    vgsr = gu.helio2galactic(data.vhel.values,l,b)
    data.loc[:,'l'] = pd.Series(l,index=data.index)
    data.loc[:,'b'] = pd.Series(b,index=data.index)
    data.loc[:,'vgsr'] = pd.Series(vgsr,index=data.index)
    data = data[(np.abs(data.b)>np.radians(20.))].reset_index(drop=True)
    data.to_csv("/data/aamw3/SDSS/bhb.csv")

    return None


"""
To make the K giant sample, I downloaded the data from Xue et al. and then cut so that [Fe/H]<-0.9, |b| > 20 degrees and rgc<50kpc. 
"""