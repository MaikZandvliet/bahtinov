#!/usr/bin/env python

# ====================================================================================
#               Bahtinov Mask Analysis Software to Determine (De)Focus
# ====================================================================================

# Import required modules
from __future__ import division
import os
import sys
import numpy as np
import peakutils
import pyfits
import cv2
import math
import sep
import scipy.optimize
import datetime
import time
import subprocess
import tiptilt
from scipy import stats
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from astropy.nddata import Cutout2D
from astropy.io import fits
from kapteyn import kmpfit
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from scipy.optimize import curve_fit
from shapely.geometry import LineString
import functions

today_utc_date = datetime.datetime.utcnow().strftime('%Y-%m-%d')
today_utc_time = time.strftime('%c', time.gmtime(time.time()))

'''
Class Bahtinov
    Has the end goal to calculate the focus on the CCD.
    Some parameters have to be set manually to ensure a fit.
'''
class Bahtinov:
    def __init__(image, image_path, name, X, Xerr, Y, Yerr, SNR,  offset, k, p, size_i, workdir):
        '''
        __init__ :  Defines an image that is used throughout the class
            image_path ; string
            name ; string of name of file
            X, Y ; coordinates of the center of the star in question
            Xerr, Yerr ; error on coordinates of the center of the star in question
            offset ; offset for M2
            k, p ; automatic integers by program
            size_i ; size of the cutout image
        '''
        image.workdir = workdir
        image.bias = fits.open('/media/data/Bahtinov/2017_03_15/Bias/Masterbias.fits')
        image.bias = image.bias[0].data
        image.image_path  = image_path                                  # full image path
        image.title = image.image_path.split('/')[-1]                   # name of the image with extension
        image.name = name                                               # name of the image without extension
        image.number = float(image.title.split('_')[-2])                # image number
        image.image = fits.open(image.image_path)                       # opening fits image
        image.data = image.image[1].data                                # image data
        image.data = np.asarray(image.data - image.bias, dtype = np.float)
        image.X = X ; image.Y = Y                                       # x and y coordinates of star
        image.SNR = SNR
        image.Xerr = Xerr ; image.Yerr = Yerr                           # xerr and yerr of star obtained from sep
        image.angle = math.radians(20)                                  # angle of the diagnoal gratings of the Bahtinov mask
        image.p = p ; image.k = k                                       # integers used for saving data
        image.offset = offset                                           # M2 offset
        image.correction_factor = 4.73                                  # 1 mm offset in M2 corresponds to 4.73 mm shift at CCD focal plane
        image.delta = 2590                                              # 'center of gravity' radius of the Bahtinov gratings

        # Creates new directories if non-existing
        if not os.path.exists(workdir + 'Focusrun/' + today_utc_date + '/Results/'):
            subprocess.call(('mkdir ' + workdir + 'Focusrun/' + today_utc_date + '/Results').format(workdir), shell=True)
        if not os.path.exists(workdir + 'Focusrun/' + today_utc_date + '/Plots/'):
            subprocess.call(('mkdir ' + workdir + 'Focusrun/' + today_utc_date + '/Plots').format(workdir), shell=True)
        if not os.path.exists(workdir + 'Focusrun/' + today_utc_date + '/Plots/' + str(image.name) + '/'):
            subprocess.call(('mkdir ' + workdir + 'Focusrun/' + today_utc_date + '/Plots/' + str(image.name)).format(workdir), shell=True)

        # Create a cutout image for better data handling and save image with corresponding coordinates from original image
        cutout = Cutout2D(image.data, (X, Y), (size_i, size_i))
        image.data_new = cutout.data
        image.data_new = image.rotate_image(image.data_new, 48, size_i)
        image.mean_new, image.median_new, image.std_new = sigma_clipped_stats(image.data_new, sigma=3, iters=5)

        image.data_new = np.asarray(image.data_new, dtype = np.float)
        image.data_new = image.data_new.copy(order = 'C')
        background = sep.Background(image.data_new)
        threshold = background.globalrms * 5
        image.data_new = image.data_new - background
        source = sep.extract(image.data_new, threshold)
        image.x, image.y = source['x'][np.where(source['flux'] == np.max(source['flux']))], source['y'][np.where(source['flux'] == np.max(source['flux']))]

    def rotate_image(image, data, angle, size):
        M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1.0)
        data = cv2.warpAffine(data, M, (size, size))
        data = Cutout2D(data, (size/2, size/2), (size/2+80, size/2+80)).data
        return data

    def create_scan(image, data, i):
        scan = data[:,i]                                                    # Create slice of cutout image
        scan[:50] = np.zeros(50)                                          # Section which is not relevant
        scan[-50:] = np.zeros(50)                                         # Section which is not relevant
        return scan

    def determine_peakindices(image, data, i):
        threshold = (np.max(data) * 0.7 - np.min(data)) / (np.max(data) - np.min(data))     # sets threshold for peak detection
        peakindex = peakutils.indexes(np.array(data), thres = 0.7, min_dist = 20 )        # find peaks in the slice/scan
        return peakindex


    def calculate_focus_error(image, a, sigma_a, b, sigma_b, c, sigma_c, d, sigma_d):
        sigma_ad = (a*d)**2 * ( (sigma_a/a)**2 + (sigma_d/d)**2 )
        sigma_bc = (b*c)**2 * ( (sigma_b/b)**2 + (sigma_c/c)**2 )
        sigma2_x = (((d-c) / (a-b))**2 * ( ((sigma_c**2 + sigma_d**2)/(d-c))**2 + ((sigma_a**2 + sigma_b**2)/(a-b))**2))
        sigma2_y = (((a*d-b*c)/(a-b))**2 * ( (((sigma_ad + sigma_bc) / (a*d - b*c)))**2 + ((sigma_a**2 + sigma_b**2)/(a-b))**2))
        return (sigma2_y + image.std1**2)**.5  / 2 * 9 * (33000 / image.delta)

    def calculate_focus(image, outerline0, centralline, outerline1):
        line1 = LineString([(outerline0[0][0], outerline0[0][1]), (outerline0[-1][0], outerline0[-1][1])])
        line2 = LineString([(outerline1[0][0], outerline1[0][1]), (outerline1[-1][0], outerline1[-1][1])])
        diagonal_line_intersection = line1.intersection(line2)
        if abs(np.array(diagonal_line_intersection)[0] - image.x) > 2:
            a = 1/0
        line_center = LineString([(centralline[0][0], centralline[0][1]), (centralline[-1][0], centralline[-1][1])])
        if np.array(diagonal_line_intersection)[1] > image.intercept1:
            focus = - diagonal_line_intersection.distance(line_center) / 2 * 9 * (33000 / image.delta)
        else:
            focus = diagonal_line_intersection.distance(line_center) / 2 * 9 * (33000 / image.delta)
        return focus

    def main(image, star_counter):
        '''
        BahtinovSpikes :    Fits the lines of the Bahtinov spikes by first finding exclusion zones which are unfit to be used for the fit,
                            this is achieved by finding lower and upper thresholds. These are set by the user and can be modified.
        '''
        image.star_counter = star_counter
        today_utc_date = datetime.datetime.utcnow().strftime('%Y-%m-%d')

        x = [] ; x1 = [] ; x2 = [] ; y = [] ; y1= [] ; y2 = [] ; yerr = [] ; yerr1 = [] ; yerr2 = []            # Create empty arrays for positions of diffraction spikes
        xdata = np.linspace(0,len(image.data_new),len(image.data_new))

        outerthreshold = image.x - 90
        innerthreshold = image.x - 10
        innerthreshold1 = image.x + 10
        outerthreshold1 = image.x + 90

        '''
        Loop again through all the slices to determine the points to be used for the straight line fit.
        '''
        for i in xrange(len(image.data_new)):
            scan = image.create_scan(image.data_new, i)
            # Only slices/scan between the thresholds are relevant
            if (outerthreshold < i < innerthreshold) or (innerthreshold1 < i < outerthreshold1):
                peakindex = image.determine_peakindices(scan, i)
                values = [] ; Y = []
                if len(peakindex) == 0:
                    break
                for index_ in peakindex:
                    values.append(scan[index_])
                    Y.append(xdata[index_])
                    index = sorted(np.array(values).argsort()[-3:])
                # Only if at least three peaks are found continue to fit three lorentzian functions
                if len(index) >= 3:
                    parguess = (values[index[0]], Y[index[0]], 2, values[index[1]], Y[index[1]], 2, values[index[2]], Y[index[2]], 2)
                    fitobj = kmpfit.Fitter(residuals=functions.lorentzianresiduals, data=(xdata, scan))
                    # Try to fit using the guesses obtained from peak detection and append relevant values to position arrays
                    try:
                        fitobj.fit(params0 = parguess)

                        # Distinction between the central and diagonal peaks for left and right half of image
                        if i <= image.x:                                    # Left half of image
                            if (image.y - 35 < fitobj.params[1] < image.y - 15) :
                                y.append(fitobj.params[1])
                                x.append(i)
                                yerr.append(fitobj.stderr[1])
                            if (image.y - 10 < fitobj.params[4] < image.y + 10) :
                                y1.append(fitobj.params[4])
                                x1.append(i)
                                yerr1.append(fitobj.stderr[4])
                            if (image.y + 10 < fitobj.params[7] < image.y + 35) :
                                y2.append(fitobj.params[7])
                                x2.append(i)
                                yerr2.append(fitobj.stderr[7])
                        if i > image.x:                                     # Right half of image
                            if (image.y + 10 < fitobj.params[7] < image.y + 35) :
                                y.append(fitobj.params[7])
                                x.append(i)
                                yerr.append(fitobj.stderr[7])
                            if (image.y - 10 < fitobj.params[4] < image.y + 10) :
                                y1.append(fitobj.params[4])
                                x1.append(i)
                                yerr1.append(fitobj.stderr[4])
                            if (image.y - 35 < fitobj.params[1] < image.y - 15) :
                                y2.append(fitobj.params[1])
                                x2.append(i)
                                yerr2.append(fitobj.stderr[1])

                    # Skip if something went wrong with fit
                    except Exception, mes:
                        pass



        # Fit the arrays (which are the lines of the spikes) by a linear line
        fitobj = kmpfit.Fitter(residuals=functions.residuals, data=(np.array(x),np.array(y), np.array(yerr)))
        fitobj1 = kmpfit.Fitter(residuals=functions.residuals1, data=(np.array(x1),np.array(y1), np.array(yerr1)))
        fitobj2 = kmpfit.Fitter(residuals=functions.residuals2, data=(np.array(x2),np.array(y2), np.array(yerr2)))
        try:
            fitobj.fit(params0=[90])
            image.std = float(fitobj.stderr)

            fitobj1.fit(params0=[140])
            image.std1 = float(fitobj1.stderr)

            fitobj2.fit(params0=[190])
            image.std2 = float(fitobj2.stderr)
            # Define stuff
            image.intercept = fitobj.params
            image.intercept1 = fitobj1.params
            image.intercept2 = fitobj2.params
            image.Y0 = image.angle * np.array(xdata) + image.intercept
            image.Y1 = 0 * np.array(xdata) + image.intercept1
            image.Y2 = -image.angle * np.array(xdata) + image.intercept2
            image.XY = zip(xdata, image.Y0)
            image.XY1 = zip(xdata, image.Y1)
            image.XY2 = zip(xdata, image.Y2)

            image.Focus = image.calculate_focus(image.XY, image.XY1, image.XY2)
            image.focuserr = image.calculate_focus_error(image.angle, 0, -image.angle, 0, image.intercept[0], image.std, image.intercept2[0], image.std2)#(sigma2_y + image.std1**2)**.5


            if len(x) > 1 and len(x1) > 1 and len(x2) > 1:
                if np.min(x) > image.x or np.min(x1) > image.x or np.min(x2) > image.x:
                    print 1/0
                if np.max(x) < image.x or np.max(x1) < image.x or np.max(x2) < image.x:
                    print 1/0

            '''
            Saving stuff
            '''
            if image.focuserr < 82.5 and image.focuserr < abs(image.Focus):
                if os.path.exists(image.workdir +'Focusrun/' + today_utc_date + '/Results/FocusResults.txt'):
                    Results = np.loadtxt(image.workdir +'Focusrun/' + today_utc_date + '/Results/FocusResults.txt')
                    values = np.array([image.number, image.offset, image.Focus, image.focuserr, image.X, image.Y, image.SNR]).flatten()
                    Results = np.vstack((Results, values))
                    np.savetxt(image.workdir +'Focusrun/' + today_utc_date + '/Results/FocusResults.txt', Results, fmt = '%10.1f %10.1f %10.5f %10.5f %10.3f %10.3f %10.3f')

                else:
                    Results = np.zeros((1,7))
                    Results[0,0] = image.number ; Results[0,1] = image.offset
                    Results[0,2] = image.Focus ; Results[0,3] = image.focuserr
                    Results[0,4] = image.X ; Results[0,5] = image.Y
                    Results[0,6] = image.SNR
                    np.savetxt(image.workdir +'Focusrun/' + today_utc_date + '/Results/FocusResults.txt', Results, fmt = '%10.1f %10.1f %10.5f %10.5f %10.3f %10.3f %10.3f')

                # Plot star with fitted diffraction spikes
                image.fig, image.axis = plt.subplots(figsize = (10,10))
                image.axis.imshow(image.data_new, cmap='Greys' , origin='lower', norm = LogNorm(vmin=0.01, vmax = np.max(image.data_new)))
                image.axis.scatter(x,y, s = 20, color = 'r')
                image.axis.scatter(x1,y1, s = 20, color = 'g')
                image.axis.scatter(x2,y2, s = 20, color = 'c')
                image.axis.scatter(image.x, image.y, color = 'r')
                image.axis.set_xlim(0,len(image.data_new)) ; image.axis.set_ylim(0,len(image.data_new))
                image.axis.set_xlabel('x') ; image.axis.set_ylabel('y')
                image.axis.set_title('Bahtinov Source: %s (%.2f, %.2f)' %(image.name, image.X, image.Y))
                image.axis.plot(zip(*image.XY)[0], zip(*image.XY)[1], color = 'r')
                image.axis.plot(zip(*image.XY1)[0], zip(*image.XY1)[1], color = 'g')
                image.axis.plot(zip(*image.XY2)[0], zip(*image.XY2)[1], color = 'r')
                image.axis.annotate('Axial distance = %.2f $\pm$ %.3f $\\mu m$' %(image.Focus, image.focuserr), xy=(0, 1), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='top')
                image.fig.savefig(image.workdir +'Focusrun/' + today_utc_date + '/Plots/' + str(image.name) + '/' + str(image.name) + '_' + str(image.X) + '_' + str(image.Y) + '.png')
                plt.close()
            else:
                image.star_counter += 1
        #If anything goes wrong with fit skip the star
        except Exception, mes:
            image.star_counter += 1
            pass
        return image.star_counter
