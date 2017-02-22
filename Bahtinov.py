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


directory_prefix_work = '/media/data/bahtinov_results/'                        # directory prefix set where to save data
today_utc_date = datetime.datetime.utcnow().strftime('%Y-%m-%d')
today_utc_time = time.strftime('%c', time.gmtime(time.time()))


'''
Class Bahtinov
    Has the end goal to calculate the focus on the CCD.
    Some parameters have to be set manually to ensure a fit.
'''
class Bahtinov:
    def __init__(image, image_path, X, Xerr, Y, Yerr, offset, k, p, size_i):
        '''
        __init__ :  Defines an image that is used throughout the class
            image_path ; string
            X, Y ; coordinates of the center of the star in question
            Xerr, Yerr ; error on coordinates of the center of the star in question
            offset ; offset for M2
            k, p ; automatic integers by program
            size_i ; size of the cutout image
        '''
        bias = fits.open('/media/data/scripts/temp_12000x10600_22_test.fits')
        bias = bias[1].data
        image.image_path  = image_path                                  # full image path
        image.title = image.image_path.split('/')[-1]                   # name of the image with extension
        image.name = image.title.split('.')[0]                          # name of the image without extension
        image.number = float(image.title.split('_')[-2])                # image number
        image.image = fits.open(image.image_path)                       # opening fits image
        image.data = image.image[1].data                                # image data
        image.X = X ; image.Y = Y                                       # x and y coordinates of star
        image.Xerr = Xerr ; image.Yerr = Yerr                           # xerr and yerr of star as from SExtractor
        image.angle = math.radians(20)                                  # angle of the diagnoal gratings from Bahtinov mask
        image.p = p ; image.k = k                                       # integers used for saving data
        image.offset = offset                                           # M2 offset

        # Creates new directories if non-existing
        if not os.path.exists(directory_prefix_work + 'Focusrun'):
            subprocess.call(('mkdir ' + directory_prefix_work + 'Focusrun').format(directory_prefix_work), shell=True)
        if not os.path.exists(directory_prefix_work + 'Focusrun/' + today_utc_date):
            subprocess.call(('mkdir ' + directory_prefix_work + 'Focusrun/' + today_utc_date ).format(directory_prefix_work), shell=True)
        if not os.path.exists(directory_prefix_work + 'Focusrun/' + today_utc_date + '/Results/'):
            subprocess.call(('mkdir ' + directory_prefix_work + 'Focusrun/' + today_utc_date + '/Results').format(directory_prefix_work), shell=True)
        if not os.path.exists(directory_prefix_work + 'Focusrun/' + today_utc_date + '/Plots/'):
            subprocess.call(('mkdir ' + directory_prefix_work + 'Focusrun/' + today_utc_date + '/Plots').format(directory_prefix_work), shell=True)
        if not os.path.exists(directory_prefix_work + 'Focusrun/' + today_utc_date + '/Plots/' + str(image.name) + '/'):
            subprocess.call(('mkdir ' + directory_prefix_work + 'Focusrun/' + today_utc_date + '/Plots/' + str(image.name)).format(directory_prefix_work), shell=True)
        if not os.path.exists(directory_prefix_work + 'Focusrun/' + today_utc_date + '/Results/'+ str(image.name) + '/'):
            subprocess.call(('mkdir ' + directory_prefix_work + 'Focusrun/' + today_utc_date + '/Results/' + str(image.name)).format(directory_prefix_work), shell=True)

        # Create a cutout image for better data handling and save image with corresponding coordinates from original image
        cutout = Cutout2D(image.data, (X, Y), (size_i, size_i))
        image.data_new = cutout.data
        image.data_new = image.rotate_image(image.data_new, -42, size_i)
        image.mean_new, image.median_new, image.std_new = sigma_clipped_stats(image.data_new, sigma=3, iters=5)

        image.data_new = np.asarray(image.data_new, dtype = np.float)
        background = sep.Background(image.data_new)
        threshold = background.globalrms * 10
        image.data_new = image.data_new - background
        source = sep.extract(image.data_new, threshold)
        image.x, image.y = source['x'][np.where(source['flux'] == np.max(source['flux']))], source['y'][np.where(source['flux'] == np.max(source['flux']))]

    def rotate_image(image, data, angle, size):
        M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1.0)
        data = cv2.warpAffine(data, M, (size, size))
        data = Cutout2D(data, (size/2, size/2), (size/2+80, size/2+80)).data
        return data

    def calculate_threshold(image, data):
        '''
        Find the thresholds between which the program should calculate the points which are used to fit the straight lines. Threshold is set by
        setting lower of upper limits on the difference between the max value and mean of a slice in the main image, limits found by trail and error.
        '''
        innerthreshold = 0 ; outerthreshold = 0 ; outerthreshold1 = 0 ; innerthreshold1 = 0     # Start with 'no' thresholds for calculations
        for i in xrange(len(data)):
            scan = data[:,i]
            # Determine threshold at left half of image
            if outerthreshold == 0 :
                if abs(max(scan) - np.mean(scan)) > 3000:
                    outerthreshold = i + 10
            if innerthreshold == 0 :
                if abs(max(scan) - np.mean(scan)) > 12000:
                    innerthreshold = i + 10
            # Determine new threshold at right half of image
            if i > image.x:
                if innerthreshold1 == 0 :
                    if abs(max(scan) - np.mean(scan)) < 12000:
                        innerthreshold1 = i - 10
                if outerthreshold1 == 0 :
                    if abs(max(scan) - np.mean(scan)) < 3000:
                        outerthreshold1 = i - 10
        #show()
        return outerthreshold, innerthreshold, innerthreshold1, outerthreshold1


    def create_scan(image, data, i):
        scan = data[:,i]                                                    # Create slice of cutout image
        scan[:100] = np.zeros(100)                                          # Section which is not relevant
        scan[-100:] = np.zeros(100)                                         # Section which is not relevant
        return scan

    def determine_peakindices(image, data, i):
        threshold = (np.max(data) * 0.7 - np.min(data)) / (np.max(data) - np.min(data))     # sets threshold for peak detection
        peakindex = peakutils.indexes(np.array(data), thres=threshold, min_dist=15 )        # find peaks in the slice/scan
        return peakindex

    def sort_peaks(image, params, paramstd, i):
        if i <= image.x:                                    # Left half of image
            if (image.x - 40 < params[1] < image.x - 15) :
                y.append(params[1])
                x.append(i)
                yerr.append(paramstd[1])
            if (image.x - 10 < params[4] < image.x + 10) :
                y1.append(params[4])
                x1.append(i)
                yerr1.append(paramstd[4])
            if (image.x + 10 < params[7] < image.x + 40) :
                y2.append(params[7])
                x2.append(i)
                yerr2.append(paramstd[7])
        if i > image.x:                                     # Right half of image
            if (image.x + 10 < params[7] < image.x + 40) :
                y.append(params[7])
                x.append(i)
                yerr.append(paramstd[7])
            if (image.x - 10 < params[4] < image.x + 10) :
                y1.append(params[4])
                x1.append(i)
                yerr1.append(paramstd[4])
            if (image.x - 40 < params[1] < image.x - 15) :
                y2.append(params[1])
                x2.append(i)
                yerr2.append(paramstd[1])
        return x, x1, x2, y, y1, y2, yerr, yerr1, yerr2

    def calculate_focus_error(image, a, sigma_a, b, sigma_b, c, sigma_c, d, sigma_d):
        sigma_ad = (a*d)**2 * ( (sigma_a/a)**2 + (sigma_d/d)**2 )
        sigma_bc = (b*c)**2 * ( (sigma_b/b)**2 + (sigma_c/c)**2 )
        sigma2_x = (((d-c) / (a-b))**2 * ( ((sigma_c**2 + sigma_d**2)/(d-c))**2 + ((sigma_a**2 + sigma_b**2)/(a-b))**2 ))
        sigma2_y = (((a*d-b*c)/(a-b))**2 * ( (((sigma_ad + sigma_bc) / (a*d - b*c)))**2 + ((sigma_a**2 + sigma_b**2)/(a-b))**2  ))
        return (sigma2_y + image.std1**2)**.5

    def calculate_focus(image, outerline0, centralline, outerline1):
        line1 = LineString([(outerline0[0][0], outerline0[0][1]), (outerline0[-1][0], outerline0[-1][1])])
        line2 = LineString([(outerline1[0][0], outerline1[0][1]), (outerline1[-1][0], outerline1[-1][1])])
        diagonal_line_intersection = line1.intersection(line2)
        if abs(np.array(diagonal_line_intersection)[0] - image.x) > 2:
            a = 1/0
        line_center = LineString([(centralline[0][0], centralline[0][1]), (centralline[-1][0], centralline[-1][1])])
        if np.array(diagonal_line_intersection)[1] > image.intercept1:
            focus = -diagonal_line_intersection.distance(line_center) * 9
        else:
            focus = diagonal_line_intersection.distance(line_center) * 9
        return focus

    def main(image):
        '''
        BahtinovSpikes :    Fits the lines of the Bahtinov spikes by first finding exclusion zones which are unfit to be used for the fit,
                            this is achieved by finding lower and upper thresholds. These are set by the user and can be modified.
        '''
        today_utc_date = datetime.datetime.utcnow().strftime('%Y-%m-%d')

        x = [] ; x1 = [] ; x2 = [] ; y = [] ; y1= [] ; y2 = [] ; yerr = [] ; yerr1 = [] ; yerr2 = []            # Create empty arrays for positions of diffraction spikes
        xdata = np.linspace(0,len(image.data_new),len(image.data_new))

        outerthreshold, innerthreshold, innerthreshold1, outerthreshold1 = image.calculate_threshold(image.data_new)
        outerthreshold = 50
        innerthreshold = 120
        innerthreshold1 = 170
        outerthreshold1 = 240

        '''
        Loop again through all the slices to determine the points to be used for the straight line fit.
        '''
        for i in xrange(len(image.data_new)):
            scan = image.create_scan(image.data_new, i)
            # Only if all thresholds are non-zero the fit can start
            if outerthreshold != 0 and innerthreshold != 0 and innerthreshold1 != 0 and outerthreshold1 != 0:
                # Only slices/scan between the thresholds are relevant
                if (outerthreshold < i < innerthreshold) or (innerthreshold1 < i < outerthreshold1):
                    peakindex = image.determine_peakindices(scan, i)
                    values = [] ; Y = []
                    if len(peakindex) == 0:
                        print '=================# Error #================='
                        print 'No clear peaks were found to initate the fit.'
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
                            #print fitobj.params
                            #fig = figure()
                            #axis = fig.add_subplot(111)
                            #axis.plot(scan, linewidth = .5)
                            #axis.plot(xdata, functions.ThreeLorentzian(xdata, *fitobj.params))
                            #plt.close()
                            #show()
                            # Distinction between the central and diagonal peaks for left and right half of image
                            if i <= image.x:                                    # Left half of image
                                if (image.x - 40 < fitobj.params[1] < image.x - 15) :
                                    y.append(fitobj.params[1])
                                    x.append(i)
                                    yerr.append(fitobj.stderr[1])
                                if (image.x - 10 < fitobj.params[4] < image.x + 10) :
                                    y1.append(fitobj.params[4])
                                    x1.append(i)
                                    yerr1.append(fitobj.stderr[4])
                                if (image.x + 10 < fitobj.params[7] < image.x + 40) :
                                    y2.append(fitobj.params[7])
                                    x2.append(i)
                                    yerr2.append(fitobj.stderr[7])
                            if i > image.x:                                     # Right half of image
                                if (image.x + 10 < fitobj.params[7] < image.x + 40) :
                                    y.append(fitobj.params[7])
                                    x.append(i)
                                    yerr.append(fitobj.stderr[7])
                                if (image.x - 10 < fitobj.params[4] < image.x + 10) :
                                    y1.append(fitobj.params[4])
                                    x1.append(i)
                                    yerr1.append(fitobj.stderr[4])
                                if (image.x - 40 < fitobj.params[1] < image.x - 15) :
                                    y2.append(fitobj.params[1])
                                    x2.append(i)
                                    yerr2.append(fitobj.stderr[1])

                        # Skip if something went wrong with fit
                        except Exception, mes:
                            #print '=================# Error #================='
                            #print 'Something wrong with curve fit:', mes
                            #print 'This fit is being skipped.'
                            pass

        # Fit the arrays (which are the lines of the spikes) by a linear line
        fitobj = kmpfit.Fitter(residuals=functions.residuals, data=(np.array(x),np.array(y), np.array(yerr)))
        fitobj1 = kmpfit.Fitter(residuals=functions.residuals1, data=(np.array(x1),np.array(y1), np.array(yerr1)))
        fitobj2 = kmpfit.Fitter(residuals=functions.residuals2, data=(np.array(x2),np.array(y2), np.array(yerr2)))
        try:
            fitobj.fit(params0=[90])
            image.std = fitobj.stderr

            fitobj1.fit(params0=[140])
            image.std1 = fitobj1.stderr

            fitobj2.fit(params0=[190])
            image.std2 = fitobj2.stderr

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
            image.focuserr = image.calculate_focus_error(image.angle, 0, -image.angle, 0, image.intercept[0], image.std[0], image.intercept2[0], image.std2[0])#(sigma2_y + image.std1**2)**.5

            '''
            Saving stuff
            '''
            if os.path.exists(directory_prefix_work +'Focusrun/' + today_utc_date + '/Results/' + str(image.name) + '/FocusResults.txt'):
                Results = np.loadtxt(directory_prefix_work +'Focusrun/' + today_utc_date + '/Results/' + str(image.name) + '/FocusResults.txt')
                values = np.array([image.number, image.offset, image.Focus, image.focuserr, image.X, image.Y]).flatten()
                Results = np.vstack((Results, values))
                np.savetxt(directory_prefix_work +'Focusrun/' + today_utc_date + '/Results/' + str(image.name) + '/FocusResults.txt', Results, fmt = '%10.1f %10.1f %10.5f %10.5f %10.3f %10.3f')

            else:
                Results = np.zeros((1,6))
                Results[0,0] = image.number ; Results[0,1] = image.offset
                Results[0,2] = image.Focus ; Results[0,3] = image.focuserr
                Results[0,4] = image.X ; Results[0,5] = image.Y
                np.savetxt(directory_prefix_work +'Focusrun/' + today_utc_date + '/Results/' + str(image.name) + '/FocusResults.txt', Results, fmt = '%10.1f %10.1f %10.5f %10.5f %10.3f %10.3f')

            if os.path.exists(directory_prefix_work +'Focusrun/' + today_utc_date + '/Results/' + str(image.name) + '/FocusCCDResults_' + str(image.name)+ '.txt'):
                Results = np.loadtxt(directory_prefix_work +'Focusrun/' + today_utc_date + '/Results/' + str(image.name) + '/FocusCCDResults_' + str(image.name)+ '.txt')
                values = np.array([image.number, image.offset, image.Focus, image.focuserr, image.X, image.Y]).flatten()
                Results = np.vstack((Results, values))
                np.savetxt(directory_prefix_work +'Focusrun/' + today_utc_date + '/Results/' + str(image.name) + '/FocusCCDResults_' + str(image.name)+ '.txt', Results, fmt = '%10.1f %10.1f %10.5f %10.5f %10.3f %10.3f')

            else:
                Results = np.zeros((1,6))
                Results[0,0] = image.number ; Results[0,1] = image.offset
                Results[0,2] = image.Focus ; Results[0,3] = image.focuserr
                Results[0,4] = image.X ; Results[0,5] = image.Y
                np.savetxt(directory_prefix_work +'Focusrun/' + today_utc_date + '/Results/' + str(image.name) + '/FocusCCDResults_' + str(image.name)+ '.txt', Results, fmt = '%10.1f %10.1f %10.5f %10.5f %10.3f %10.3f')

            # Plot star with fitted diffraction spikes
            image.fig, image.axis = plt.subplots(figsize = (10,10))
            image.axis.imshow(image.data_new*(10**7), cmap='Greys' , origin='lower')
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
            image.axis.annotate('Focus distance = %.2f $\pm$ %.3f $\\mu m$' %(image.Focus, image.focuserr), xy=(0, 1), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='top')
            image.fig.savefig(directory_prefix_work +'Focusrun/' + today_utc_date + '/Plots/' + str(image.name) + '/Source_' + str(image.title) + '_' + str(image.X) + '_' + str(image.Y) + '_' + '.png')
            plt.close()

        #If anything goes wrong with fit skip the star
        except Exception, mes:
            #print 'Something wrong with fit: ', mes
        #    print 'This star was not fitted correctly, going to next star, due to absence of star no data is saved.'
            pass
