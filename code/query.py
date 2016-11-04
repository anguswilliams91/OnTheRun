import sql_utils as sql
import pandas as pd 
import numpy as np 
import gus_utils as gu

def main_sequence_query():

	getstr = "select spa.ra, spa.dec, spa.dered_u, spa.psfmagerr_u,spa.dered_g,\
	 spa.psfmagerr_g,spa.dered_r, spa.psfmagerr_r,spa.dered_i, spa.psfmagerr_i,\
	 spa.dered_z, spa.psfmagerr_z,spp.elodiervfinal,spp.elodiervfinalerr,\
	 spp.fehadop,spp.fehadopunc,spp.loggadop,spp.loggadopunc,spp.teffadop,spp.teffadopunc \
	 from sdssdr9.specphotoall as spa, sdssdr9.sppparams as spp where spp.specobjid=spa.specobjid \
	 and spp.scienceprimary=1 and spa.class='STAR' and spa.extinction_r<0.3 and spa.dered_g-spa.dered_r<0.6 \
	 and spa.dered_r between 14.5 and 20. and spp.fehadop between -4. and 2. and spp.loggadop between 3.5 and 4. \
	 and spp.elodiervfinal between -1000. and 1000. and spp.teffadop between 4000. and 10000."

	res = sql.get(getstr) 
	data = pd.DataFrame(np.array(res).T,columns=['ra','dec','u','u_err','g','g_err','r','r_err','i','i_err',\
													'z','z_err','vhel','vhel_err','feh','feh_err','logg',\
													'logg_err','teff','teff_err'])

	l,b = gu.radec2galactic(data.ra.values,data.dec.values)
	data.loc[:,'l'] = pd.Series(l,index=data.index)
	data.loc[:,'b'] = pd.Series(b,index=data.index)
	data = data[np.abs(data.b)>np.radians(20.)].reset_index(drop=True)

	data.to_csv("/data/aamw3/SDSS/main_sequence.csv")

	return None