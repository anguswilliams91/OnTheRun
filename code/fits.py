from __future__ import division, print_function

import numpy as np, models as m

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import fixed_quad

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
    r = r[r<50.]
    pdf,bins = np.histogram(r,10,normed=True)
    r_nodes = np.array([.5*(bins[i]+bins[i+1]) for i in np.arange(10)])

    return ([np.min(r_nodes), np.max(r_nodes)],InterpolatedUnivariateSpline(r_nodes,pdf))


def compute_posterior(v,vmin,k,spline,limits,params,model):

    if v<vmin or v>m.vesc_model(limits[1],0.,0.,params,model): return 0.

    def numerator_integrand(r):

        return (m.vesc_model(r,0.,0.,params,model) - v)**(k+1.) * spline(r)

    numerator = fixed_quad(numerator_integrand,limits[0],limits[1],n=12)[0]

    def denominator_integrand(r):

        return spline(r)*(m.vesc_model(r,0.,0.,params,model) - vmin)**(k+2.) / (k+2.)

    denominator = fixed_quad(denominator_integrand,limits[0],limits[1],n=12)[0]

    return numerator/denominator


def compute_confidence_intervals(v,vmin,chain,model,tracer):

    """
    Use the above function to compute confidence intervals on the pdf of 
    the data for a range of v.

    """

    return None




