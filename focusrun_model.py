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
from matplotlib.pyplot import figure, show, rc, cm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import AutoMinorLocator
from kapteyn import kmpfit
from astropy.io import fits
from scipy import stats
import scipy

# Set figure parameters and date
rcParams['figure.figsize'] = 14, 8
rc('font', size=12)
rc('legend', fontsize=12)
today_utc_date = datetime.datetime.utcnow().strftime('%Y-%m-%d')
today_utc_time = time.strftime('%c', time.gmtime(time.time()))

print 'UTC:', today_utc_time



def FocusRunResults():
    data = np.loadtxt(workdir + 'Focusrun/' + today_utc_date + '/Results/FocusResults.txt')
    image = data[:,0] ; defocus = data[:,1] ; focus = data[:,2] ; focuserr = data[:,3]
    xp = np.linspace(min(defocus), max(defocus), 100)
    z = np.polyfit(defocus, focus, 1)
    fitobj = kmpfit.Fitter(residuals=functions.linfitresiduals, data=(defocus, focus, focuserr), xtol=1e-12, gtol=1e-12)
    fitobj.fit(params0=z)
    print "\n=================== Results linear fit best defocus M2 ==================="
    print '{:25s} {:1s} {:3s}'.format("File", ":", name)
    print '{:25s} {:1s} {:3s}'.format("Fitted parameters kmpfit", ":", fitobj.params)
    print '{:25s} {:1s} {:3s}'.format("Covariance errors", ":", fitobj.xerror)
    print '{:25s} {:1s} {:3s}'.format("Standard errors", ":", fitobj.stderr)
    print '{:25s} {:1s} {:3.4f}'.format("Chi^2 min", ":", round(fitobj.chi2_min, 2))
    print '{:25s} {:1s} {:3.4f}'.format("Reduced Chi^2", ":", round(fitobj.rchi2_min, 2))
    print '{:25s} {:1s} {:3.0f}'.format("Iterations", ":", fitobj.niter)
    print '{:25s} {:1s} {:3.0f}'.format("Number of free pars.", ":", fitobj.nfree)
    print '{:25s} {:1s} {:3.0f}'.format("Degrees of freedom", ":", fitobj.dof)
    print '{:25s} {:1s} {:3s}'.format("Status Message", ":", fitobj.message)

    bestfocus = -fitobj.params[0]/fitobj.params[1]
    #print bestfocus
    bestfocusvar = bestfocus**2 * ( (fitobj.stderr[0] / fitobj.params[0])**2 + (fitobj.stderr[1] / fitobj.params[1])**2 - 2 * (fitobj.covar[0,1]/(fitobj.params[0]*fitobj.params[1])))
    bestfocusvar = (fitobj.params[1] / fitobj.params[0]**2)**2 * fitobj.stderr[1]**2 + (-1/fitobj.params[1])**2 * fitobj.stderr[0]**2
    #print (fitobj.stderr[1] / fitobj.params[1])**2, 2 * (fitobj.covar[0,1]/(fitobj.params[0]*fitobj.params[1]))
    #print bestfocusvar
    fig, axis = plt.subplots()
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.errorbar(defocus,focus, yerr = focuserr, fmt = 'ko')
    axis.annotate('Best offset M2 = %.2f $\pm$ %.3f $\\mu m$' %(bestfocus, bestfocusvar**.5),
        xy=(0, 1), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='top')
    axis.plot(xp, functions.linfit(fitobj.params,xp), color = 'r', ls='--', lw=2, label = 'linear fit \n $\chi ^2_{reduced}$ = %.3f' %(fitobj.rchi2_min))
    axis.set_title('Focus run M2 vs calculated axial defocus', fontsize = 14)
    axis.set_xlabel('Piston M2 [$\mu m$]')
    axis.set_ylabel('Calculated axial defocus [$\mu m$]')
    axis.set_xlim(-5,205)
    axis.grid(True)
    axis.legend(loc=4,fancybox=True, shadow=True, ncol=4, borderpad=1.01)
    fig.tight_layout()
    fig.savefig(workdir + 'Focusrun/' + today_utc_date + '/Results/Focusrun_defocus.png')

def select_sources(file, workdir, threshold = 25, SNR_limit = 3):
    objects = []
    name = file.split('/')[-1].split('.')[0]
    fitsfile = pyfits.open(file, uint=False)
    original = fitsfile[1].data
    original = np.asarray(original, dtype = np.float)
    bias = fits.open('/media/data/Bahtinov/2017_03_15/Bias/Masterbias.fits')
    bias = bias[0].data
    original = original
    data, background =  image.subtract_background(original)
    sources = sep.extract(data, threshold, err = background.globalrms, gain = 1.0, minarea = 50)
    sources = np.sort(sources, order = 'flux')
    #print len(sources)
    BG = sep.sum_ellipann(data, sources['x'], sources['y'], sources['a'], sources['b'], sources['theta'], sources['b']*1.5, sources['b']*4.5, err = background.globalrms)[0]
    SNR = sources['flux'] / BG
    objects.append([sources['x'][np.where(SNR > SNR_limit)],
                sources['x2'][np.where(SNR > SNR_limit)],
                sources['y'][np.where(SNR > SNR_limit)],
                sources['y2'][np.where(SNR > SNR_limit)],
                SNR[np.where(SNR > SNR_limit)]])
    objects = objects[0]
    ratio = data.shape[0] * 1.0 / data.shape[1]
    fig = figure(figsize=(10,ratio *10))
    axis = fig.add_subplot(111)
    axis.imshow(data, cmap=cm.gray, norm = matplotlib.colors.LogNorm(vmin=0.01, vmax = np.max(data)), origin = 'lower')
    axis.scatter(sources['x'], sources['y'], color = 'b', s = 2.5)
    axis.scatter(objects[0], objects[2], color = 'r', s = 2.5)
    fig.savefig(workdir + 'Focusrun/' + today_utc_date + '/' + name + '.png')
    plt.close()
    return np.asarray(objects)


# ====================================================================================
#                                   Start script
# ====================================================================================
if __name__ == '__main__':
    import os
    import subprocess
    import glob
    import argparse
    import numpy as np
    from tqdm import *
    import matplotlib
    import functions
    import Bahtinov_Hough

    start = time.time()
    directory_prefix = '/media/data/'

    print '-'*75
    params = argparse.ArgumentParser(description = 'User parameters.')
    params.add_argument('--workdir', required = True, help = 'Set directory where to save data')
    params.add_argument('--image', default = None, help = 'Single image to run script on.')
    params.add_argument('--path', default = None, help = 'Path to images to run script on.')
    params.add_argument('--offset_image', default = None, help = 'Offset of M2 used with single image')
    params.add_argument('--offset_path', default = None, nargs='+', type=int, help = 'Offsets of M2 used with set of images, example: 1 2 3 4 5')
    params.add_argument('--offset_path_single', default = None, help = 'Offset of M2 used for all images in directory')
    params.print_help()
    args = params.parse_args()
    workdir = args.workdir
    print '-'*75

    if args.path is not None:
        workdir = args.workdir
        subprocess.call(('rm ' + workdir + 'Focusrun/' + today_utc_date + '/Results/FocusResults.txt').format(workdir), shell=True)

        # Creates new directories if non-existing
        if not os.path.exists(workdir + 'Focusrun'):
            subprocess.call(('mkdir ' + workdir + 'Focusrun').format(workdir), shell=True)
        if not os.path.exists(workdir + 'Focusrun/' + today_utc_date):
            subprocess.call(('mkdir ' + workdir + 'Focusrun/' + today_utc_date ).format(workdir), shell=True)

        print 'Current work directory:', workdir
        files = sorted(glob.glob(args.path + '*.fits'))
        offset = np.zeros(len(files))
        if args.offset_path is not None:
            offset = args.offset_path
        if args.offset_path_single is not None:
            offset = np.full(len(files), float(args.offset_path_single))
        window_size = 400
        print '\n'+ '='*25 + ' Focus Run Analysis Started ' +  '='*25
        print 'Number of files:', len(files)

        for i in tqdm(xrange(0, len(files))):
            star_counter = 0
            name = files[i].split('/')[-1].split('.')[0]
            x, x2, y, y2, SNR = select_sources(files[i], workdir)

            #x, x2, y, y2, SNR = [5154,9453], [0,0], [5727,6650], [0,0], [20,20]
            subprocess.call(('rm ' + workdir + 'Focusrun/' + today_utc_date + '/Plots/' + str(name) + '/*').format(workdir), shell=True)
            print 'Number of candidates found:', len(x)
            for j in xrange(0,len(x)):
                Image = Bahtinov_Hough.Bahtinov(files[i], name, x[j], x2[j] ,y[j], y2[j], SNR[j], offset[i], i, j, window_size, workdir)
                star_counter = Image.houghtransform(star_counter)
                #Image.check_error()
                #star_counter = Image.main(star_counter)

            #if (len(x) - star_counter) == 0 or len(x) == 0:
            #    star_counter = 0
            #    print 'No sources used, new threshold value.'
            #    x, x2, y, y2, SNR = select_sources(files[i], workdir, SNR_limit = 1)
            #    print 'Number of candidates found:', len(x)
            #    for j in xrange(0,len(x)):
            #        Image = Bahtinov_Hough.Bahtinov(files[i], name, x[j], x2[j] ,y[j], y2[j], SNR[j], offset[i], i, j, window_size, workdir)
            #        star_counter = Image.houghtransform(star_counter)

            print 'Number of sources used: %.f. Detection percentage: %.2f %%' %((len(x) - star_counter), (len(x) - star_counter) / len(x) *100)



    if args.image is not None:
        subprocess.call(('rm ' + workdir + 'Focusrun/' + today_utc_date + '/Results/FocusResults.txt').format(workdir), shell=True)
        workdir = args.workdir

        # Creates new directories if non-existing
        if not os.path.exists(workdir + 'Focusrun'):
            subprocess.call(('mkdir ' + workdir + 'Focusrun').format(workdir), shell=True)
        if not os.path.exists(workdir + 'Focusrun/' + today_utc_date):
            subprocess.call(('mkdir ' + workdir + 'Focusrun/' + today_utc_date ).format(workdir), shell=True)

        file = args.image
        offset = 0
        if args.offset_image is not None:
            offset = float(args.offset_image)
        print '\n'+ '='*25 + ' Focus Run Analysis Started ' + '='*25
        print 'File submitted:', file
        name = file.split('/')[-1].split('.')[0]
        window_size = 400
        x, x2, y, y2, flux = select_sources(file, workdir)
        subprocess.call(('rm ' + workdir + 'Focusrun/' + today_utc_date + '/Plots/' + str(name) + '/*').format(workdir), shell=True)
        star_counter = 0
        print 'Number of candidates found:', len(x)
        for p in xrange(0,len(x)):
            single_file = Bahtinov_Hough.Bahtinov(file, name, x[p], x2[p] ,y[p], y2[p], SNR[p], offset, 0, p, window_size, workdir)
            #star_counter = single_file.main(star_counter)
            star_counter = single_file.houghtransform(star_counter)
        print 'Number of sources used: %.f. Detection percentage: %.2f %%' %((len(x) - star_counter), (len(x) - star_counter) / len(x) *100)

    FocusRunResults()
    period = time.time() - start
    print '\nThe computation time was %.3f seconds\n' %(period)
    print '='*25 + ' Focus Run Analysis Ended ' +  '='*25
    #subprocess.call(('python CCD.py'), shell=True)
