from __future__ import division
import time
import os
import numpy as np
import peakutils
import pyfits
import cv2
import scipy.ndimage
import math
from scipy import stats

def confidence_band(x, dfdp, confprob, fitobj, f, abswei=False):
    #----------------------------------------------------------
    # Given a value for x, calculate the error df in y = model(p,x)
    # This function returns for each x in a NumPy array, the
    # upper and lower value of the confidence interval.
    # The arrays with limits are returned and can be used to
    # plot confidence bands.
    #
    #
    # Input:
    #
    # x        NumPy array with values for which you want
    #          the confidence interval.
    #
    # dfdp     A list with derivatives. There are as many entries in
    #          this list as there are parameters in your model.
    #
    # confprob Confidence probability in percent (e.g. 90% or 95%).
    #          From this number we derive the confidence level
    #          (e.g. 0.05). The Confidence Band
    #          is a 100*(1-alpha)% band. This implies
    #          that for a given value of x the probability that
    #          the 'true' value of f falls within these limits is
    #          100*(1-alpha)%.
    #
    # fitobj   The Fitter object from a fit with kmpfit
    #
    # f        A function that returns a value y = f(p,x)
    #          p are the best-fit parameters and x is a NumPy array
    #          with values of x for which you want the confidence interval.
    #
    # abswei   Are the weights absolute? For absolute weights we take
    #          unscaled covariance matrix elements in our calculations.
    #          For unit weighting (i.e. unweighted) and relative
    #          weighting, we scale the covariance matrix elements with
    #          the value of the reduced chi squared.
    #
    # Returns:
    #
    # y          The model values at x: y = f(p,x)
    # upperband  The upper confidence limits
    # lowerband  The lower confidence limits
    #
    # Note:
    #
    # If parameters were fixed in the fit, the corresponding
    # error is 0 and there is no contribution to the condidence
    # interval.
    #----------------------------------------------------------
    from scipy.stats import t
    # Given the confidence probability confprob = 100(1-alpha)
    # we derive for alpha: alpha = 1 - confprob/100
    alpha = 1 - confprob/100.0
    prb = 1.0 - alpha/2
    tval = t.ppf(prb, fitobj.dof)

    C = fitobj.covar
    n = len(fitobj.params)              # Number of parameters from covariance matrix
    p = fitobj.params
    N = len(x)
    if abswei:
        covscale = 1.0
    else:
        covscale = fitobj.rchi2_min
        df2 = np.zeros(N)
    for j in range(n):
        for k in range(n):
            df2 += dfdp[j]*dfdp[k]*C[j,k]

    df = np.sqrt(fitobj.rchi2_min*df2)
    y = f(p, x)
    delta = tval * df
    upperband = y + delta
    lowerband = y - delta
    return y, upperband, lowerband

def lorentzian(x, pars):
    A1 = pars[0] ; x01 = pars[1] ; w1 = pars[2]
    f = A1*w1*2/((x-x01)**2+w1**2)
    return f

def ThreeLorentzian(x,*pars):
    A1 = pars[0] ; x01 = pars[1] ; w1 = pars[2]
    A2 = pars[3] ; x02 = pars[4] ; w2 = pars[5]
    A3 = pars[6] ; x03 = pars[7] ; w3 = pars[8]
    p1 = lorentzian(x, [A1, x01, w1])
    p2 = lorentzian(x, [A2, x02, w2])
    p3 = lorentzian(x, [A3, x03, w3])
    return p1 + p2 + p3

def lorentzianresiduals(p, data):
    x,y = data
    A1, x01, w1, A2, x02, w2, A3, x03, w3 = p
    return y - ThreeLorentzian(x, *p)

def residuals(p, data):     # Residuals function needed by kmpfit
    x, y, yerr = data       # Data arrays is a tuple given by programmer
    a = p                   # Parameters which are adjusted by kmpfit
    w = yerr**2
    wi = np.sqrt(np.where(w==0.0, 0.0, 1.0/(w)))
    return wi*(y-(a+math.radians(20)*x))

def residuals1(p, data):    # Residuals function needed by kmpfit
    x, y, yerr = data       # Data arrays is a tuple given by programmer
    a = p                   # Parameters which are adjusted by kmpfit
    w = yerr**2
    wi = np.sqrt(np.where(w==0.0, 0.0, 1.0/(w)))
    return wi*(y-(a+0*x))

def residuals2(p, data):    # Residuals function needed by kmpfit
    x, y, yerr = data       # Data arrays is a tuple given by programmer
    a = p                   # Parameters which are adjusted by kmpfit
    w = yerr**2
    wi = np.sqrt(np.where(w==0.0, 0.0, 1.0/(w)))
    return wi*(y-(a-math.radians(20)*x))

def linfit(p, x):
    a,b = p
    return a + b*x

def linfitresiduals(p, data):
    a, b = p
    x, y = data
    #w = yerr**2
    #wi = np.sqrt(np.where(w==0.0, 0.0, 1.0/(w)))
    return (y - linfit(p,x))
