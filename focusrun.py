#!/usr/bin/env python

# ====================================================================================
#            Bahtinov Mask Analysis Software to Determine Best M2 Offset
# ====================================================================================

# Import required modules
from __future__ import division
import time
import datetime
import sep
import sewpy
from tiptilt import image
from pylab import rcParams
import pyfits
from matplotlib.pyplot import figure, show, rc
from matplotlib import patches
# Set figure parameters and date
rcParams['figure.figsize'] = 14, 8
rc('font', size=12)
rc('legend', fontsize=12)
start = time.time()
today_utc_date = datetime.datetime.utcnow().strftime('%Y-%m-%d')
today_utc_time = time.strftime('%c', time.gmtime(time.time()))

print '=============== Focus Run Analysis Started ==============='
print today_utc_time

def select_sources(file, threshold_factor = 70, minimum_flux = 1e5, window_size = 100):
    objects = []
    name = file.split('/')[-1].split('.')[0]
    fitsfile = pyfits.open(file, uint=False)
    fitsfile.info()
    original = fitsfile[1].data
    data, background =  image.subtract_background(original)
    sources = image.select_sources(data, background, threshold_factor = threshold_factor, minimum_flux = minimum_flux, window_size = window_size)
    objects.append([sources['x'], sources['x2'], sources['y'], sources['y2']])

    X,XX,Y,YY = np.loadtxt('SExtractor/' + str(name) + '_reduced.txt', usecols = (0,1,2,3), unpack = True)
    ratio = data.shape[0] * 1.0 / data.shape[1]
    fig = figure(figsize=(10,ratio *10))
    axis = fig.add_subplot(111)
    axis.imshow(data, cmap = 'gray', interpolation = 'nearest', origin = 'lower')
    axis.scatter(X,Y)
    for i,x in enumerate(sources):
        window = image.source_window(x, window_size)
        axis.add_patch(
            patches.Rectangle(
                (window[1].start, window[0].start),
                window[1].stop - window[1].start,
                window[0].stop - window[0].start,
                color = 'r',
                fill=False
            )
        )
    plt.close()
    #show()

    return np.asarray(objects)[0]



def FocusRunResults():
    data = np.loadtxt(directory_prefix +'bahtinov_results/Focusrun/' + today_utc_date + '/Results/FocusResults.txt')
    image = data[:,0] ; defocus = data[:,1] ; focus = data[:,2] ; focuserr = data[:,3]
    xp = np.linspace(min(defocus), max(defocus), 100)
    z = np.polyfit(defocus, focus, 1)
    fitobj = kmpfit.Fitter(residuals=functions.linfitresiduals, data=(defocus, focus, focuserr),
                           xtol=1e-12, gtol=1e-12)
    fitobj.fit(params0=z)
    print "\n=================== Results linear fit ==================="
    print "Fitted parameters:      ", fitobj.params
    print "Covariance errors:      ", fitobj.xerror
    print "Standard errors:        ", fitobj.stderr
    print "Chi^2 min:              ", round(fitobj.chi2_min, 2)
    print "Reduced Chi^2:          ", round(fitobj.rchi2_min, 2)
    print "Iterations:             ", fitobj.niter
    print "Number of free pars.:   ", fitobj.nfree
    print "Degrees of freedom      ", fitobj.dof
    print "Status Message:         ", fitobj.message

    dfdp = [1, xp]
    confprob = 95.0
    ydummy, upperband, lowerband = functions.confidence_band(xp, dfdp, confprob, fitobj, functions.linfit)
    verts = zip(xp, lowerband) + zip(xp[::-1], upperband[::-1])
    bestfocus = -fitobj.params[0]/fitobj.params[1]
    bestfocusvar = bestfocus**2 * ( (fitobj.stderr[0] / fitobj.params[0])**2 + (fitobj.stderr[1] / fitobj.params[1])**2 - 2 * (fitobj.covar[0,1]/(fitobj.params[0]*fitobj.params[1])))
    fig, axis = plt.subplots()
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.errorbar(defocus,focus, yerr = focuserr, fmt = 'ko')
    axis.annotate('Best offset M2 = %.2f $\pm$ %.3f $\\mu m$' %(bestfocus, bestfocusvar**.5),
        xy=(0, 1), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='top')
    axis.plot(xp, functions.linfit(fitobj.params,xp), color = 'r', ls='--', lw=2,
        label = 'linear fit \n $\chi ^2_{reduced}$ = %.3f' %(fitobj.rchi2_min))
    axis.set_title('Focus run M2 vs calculated defocus', fontsize = 14)
    axis.set_xlabel('Offset M2 [$\mu m$]')
    axis.set_ylabel('Calculated defocus [$\mu m$]')
    axis.set_xlim(90,210)
    axis.grid(True)
    axis.legend(loc=4,fancybox=True, shadow=True, ncol=4, borderpad=1.01)
    fig.tight_layout()
    fig.savefig(directory_prefix + 'bahtinov_results/Focusrun/' + today_utc_date + '/Results/Focusrun_defocus_results.png')
    #show()


# ====================================================================================
#                                   Start script
# ====================================================================================
if __name__ == '__main__':
    import os
    import subprocess
    import glob
    import numpy as np
    from tqdm import *
    import matplotlib
    import matplotlib.pyplot as plt
    import cv2
    from matplotlib.lines import Line2D
    from shapely.geometry import LineString
    import scipy.optimize
    from matplotlib.patches import Polygon
    from mpl_toolkits.mplot3d import Axes3D
    from astropy.io import fits
    from matplotlib.ticker import AutoMinorLocator
    from kapteyn import kmpfit
    import functools
    import functions
    import Bahtinov
    Offset = [100,110,120,130,140,150,160,170,180,190,200,200,190,180,170,160,150,140,130,120,110,100]  # Offsets of M2
    directory_prefix = '/media/data/'
    directory = directory_prefix + 'Bahtinov/'                 # Directory containing images
    files = sorted(glob.glob(directory + '*test.fits'))

    window_size = 400
    for k in tqdm(xrange(0,len(files))):
        name = files[k].split('/')[-1].split('.')[0]
        fitsfile = pyfits.open(files[k], uint=False)
        original = fitsfile[1].data
        data, background =  image.subtract_background(original)
        x, x2, y, y2 = np.loadtxt('SExtractor/' + str(name) + '_reduced.txt', usecols = (0,1,2,3), unpack = True)
        ratio = data.shape[0] * 1.0 / data.shape[1]
        fig = figure(figsize=(10,ratio *10))
        axis = fig.add_subplot(111)
        axis.imshow(data-background, cmap = 'gray', interpolation = 'nearest', origin = 'lower')
        axis.scatter(x,y)
        plt.close()
        #show()

        subprocess.call(('rm ' + directory_prefix + 'bahtinov_results/Focusrun/' + today_utc_date + '/Results/' + str(name) + '/FocusResults.txt').format(directory_prefix), shell=True)
        subprocess.call(('rm ' + directory_prefix + 'bahtinov_results/Focusrun/' + today_utc_date + '/Plots/' + str(name) + '/*').format(directory_prefix), shell=True)
        subprocess.call(('rm ' + directory_prefix + 'bahtinov_results/Focusrun/' + today_utc_date + '/Results/' + str(name) + '/FocusCCDResults_' + str(name)+ '.txt').format(directory_prefix), shell=True)
        for p in xrange(0,len(x)):
            Image = Bahtinov.Bahtinov(files[k], x[p], x2[p] ,y[p], y2[p], 0, k, p, window_size)
            Image.main()

    # ====================================================================================

    #FocusRunResults()
    #subprocess.call(('python CCD.py'), shell=True)
    period = time.time() - start
    print '\nThe computation time was %.3f seconds\n' %(period)
    print '=============== Focus Run Analysis Ended ==============='


    '''
    # Single image:
    file = directory + 'temp_12000x10600_890_test.fits'
    name = file.split('/')[-1].split('.')[0]
    window_size = 400
    offset = 100
    x, x2, y, y2 = np.loadtxt('SExtractor/' + str(name) + '_reduced.txt', usecols = (0,1,2,3), unpack = True)#select_sources(file)#, window_size = window_size)
    for p in xrange(0,len(x)):
        single_file = Bahtinov.Bahtinov(file, x[p], x2[p] ,y[p], y2[p], offset, 0, p, window_size)
        single_file.main()
    '''
