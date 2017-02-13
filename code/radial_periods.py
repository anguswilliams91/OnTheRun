from __future__ import division,print_function 

import numpy as np, matplotlib.pyplot as plt 
from scipy.integrate import quad

Omega_0 = 0.3
delta_th = 340. 
H = 65./1000. #in kms^-1 kpc^-1 
G = 43010.795338751527 #in km^2 s^-2 kpc (10^10 Msun)^-1
rho_crit = 3.*H**2. / (8.*np.pi*G)

smith_params = [6.5,257.,24.3]

def Phi_NFW(r,params):
    rvir,c = params 
    rho_s = rho_crit * Omega_0 * delta_th * c**3. / (3.*(np.log(1.+c) - c/(1.+c)))
    return -(4.*np.pi*G*rho_s*rvir**3.)/(c**3.*r) * np.log( 1. + c*r/rvir)

def Phi_total(r,params):
    Mbaryon,rvir,c = params 
    return -G*Mbaryon/r + Phi_NFW(r,[rvir,c])

def integrand(theta,rapo,params):
    r = .5*rapo*(np.sin(theta) + 1.)
    return rapo*np.cos(theta)/np.sqrt(2.*(Phi_total(rapo,params)-Phi_total(r,params)))

def radial_period(rapo,params):
    Tr = quad(integrand,-np.pi/2.,np.pi/2.,args=(rapo,params))[0]
    Tr *= 3e16/(365*24*60*60*1e9)
    return Tr