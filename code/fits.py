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
    if tracer=="kgiant" or tracer=="bhb":
        r = r[r<50.]
    elif tracer=="main_sequence":
        r = r[r<20.]
    pdf,bins = np.histogram(r,10,normed=True)
    r_nodes = np.array([.5*(bins[i]+bins[i+1]) for i in np.arange(10)])

    return ([np.min(r_nodes), np.max(r_nodes)],InterpolatedUnivariateSpline(r_nodes,pdf))


def compute_posterior(v,vmin,k,spline,limits,params,model):

    """Calculate the probability desnity at a line of sight velocity given a model 
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

    if v<vmin or v>m.vesc_model(limits[1],0.,0.,params,model): return 0.

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



