#!/usr/bin/env python

# ====================================================================================
#               Bahtinov Mask Analysis Software to Determine (De)Focus
# ====================================================================================

# Import required modules
from __future__ import division
import os
import numpy as np
import peakutils
import pyfits
import cv2
import math
import scipy.optimize
import datetime
import time
import subprocess
from scipy import stats
from matplotlib.lines import Line2D
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from photutils import CircularAperture
from astropy.stats import sigma_clip
from astropy.nddata import Cutout2D
from astropy.io import fits
from kapteyn import kmpfit
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from shapely.geometry import LineString
import focusfunctions


directory_prefix = '/media/maik/Maik/MeerLICHT/'                        # directory prefix set where to save data
today_utc_date = datetime.datetime.utcnow().strftime('%Y-%m-%d')
today_utc_time = time.strftime('%c', time.gmtime(time.time()))

'''
Class Bahtinov
    Has the end goal to calculate the focus on the CCD.
    Some parameters have to be set manually to ensure a fit.
'''
class Bahtinov:
    def __init__(image, image_path, X, Xerr, Y, Yerr, defocus, k, p, size_i):
        '''
        __init__ :  Defines an image that is used throughout the class
            image_path ; string
            X, Y ; coordinates of the center of the star in question
            Xerr, Yerr ; error onfcoordinates of the center of the star in question
            defocus ; defocus for M2
            k, p ; automatic integers by program
            size_i ; size of the cutout image
        '''

        # Creates new directories if non-existing
        if not os.path.exists('Focusrun'):
            subprocess.call(('mkdir Focusrun').format(directory_prefix), shell=True)
        if not os.path.exists('Focusrun/' + today_utc_date):
            subprocess.call(('mkdir Focusrun/' + today_utc_date ).format(directory_prefix), shell=True)
        if not os.path.exists('Focusrun/' + today_utc_date + '/Results/'):
            subprocess.call(('mkdir Focusrun/' + today_utc_date + '/Results').format(directory_prefix), shell=True)
        if not os.path.exists('Focusrun/' + today_utc_date + '/Plots/'):
            subprocess.call(('mkdir Focusrun/' + today_utc_date + '/Plots').format(directory_prefix), shell=True)

        image.image_path  = image_path                                  # full image path
        image.title = image.image_path .split('/')[-1]                  # name of the image with extension
        image.name = image.title.split('.')[0]                          # name of the image without extension
        image.number = float(image.title.split('_')[-1].split('.')[0])  # image number
        image.image = fits.open(image.image_path)                       # opening fits image
        image.data = image.image[1].data                                # image data
        image.X = X ; image.Y = Y                                       # x and y coordinates of star
        image.Xerr = Xerr ; image.Yerr = Yerr                           # xerr and yerr of star as from SExtractor
        image.angle = math.radians(20)                                  # angle of the diagnoal gratings from Bahtinov mask
        image.p = p ; image.k = k                                       # integers used for saving data
        image.offset = defocus                                          # M2 offset
        image.x = size_i/4 + 40 ; image.y = size_i/4 + 40               # center x and y position of resized image

        # Create a cutout image for better data handling and save image with corresponding coordinates from original image
        cutout = Cutout2D(image.data, (X, Y), (size_i, size_i))
        image.data_new = cutout.data
        try:
            subprocess.call(('rm ' + directory_prefix + 'Data/Cutout/Cutout_' + str(image.X) + '_' + str(image.Y) + '_' + str(image.title)).format(directory_prefix), shell=True)
        except:
            None
        hdu = fits.PrimaryHDU()
        hdulist = fits.HDUList([hdu,fits.ImageHDU(image.data_new)])
        hdulist.writeto(directory_prefix + 'Data/Cutout/Cutout_'+ str(image.X) + '_' + str(image.Y) + '_' +  str(image.title))

        # Rotate the image such that central diffraction spike is horizontal
        M = cv2.getRotationMatrix2D((size_i/2, size_i/2), 48, 1.0)
        image.data_new = cv2.warpAffine(image.data_new, M, (size_i, size_i))
        image.data_new = Cutout2D(image.data_new, (size_i/2, size_i/2), (size_i/2+80, size_i/2+80)).data

        # Calculate mean, median, std of cutout image and normalize the data
        image.mean_new, image.median_new, image.std_new = sigma_clipped_stats(image.data_new, sigma=3, iters=5)
        image.data_normalized = (image.data_new - np.min(image.data_new)) / (np.max(image.data_new) - np.min(image.data_new) )

    def FindSources(image):
        '''
        FindSources :   Find the new centra of sources inside an image, uses the smaller image to determine the center of the 'Bahtinov' star. If no center is found
                        the center is set to the users input for the star set in __init__
            FWHM ; the FWHM of the star in question, manual input from user
            sig ; the number of standard deviations to set limit on source detection
        '''
        try:
            image.daofind = DAOStarFinder(fwhm=5.5, threshold=10*image.std_new)
            image.sources = image.daofind(image.data_new - image.median_new)
            image.positions = (image.sources['xcentroid'], image.sources['ycentroid'])
            image.apertures = CircularAperture(image.positions, r=4.)
            #image.x = image.sources['xcentroid'][np.where(image.sources['peak'] == max(image.sources['peak']))]
            #image.y = image.sources['ycentroid'][np.where(image.sources['peak'] == max(image.sources['peak']))]
            if 138 < image.sources['xcentroid'][np.where(image.sources['peak'] == max(image.sources['peak']))] < 142:
                image.x = image.sources['xcentroid'][np.where(image.sources['peak'] == max(image.sources['peak']))]
            if 138 < image.sources['ycentroid'][np.where(image.sources['peak'] == max(image.sources['peak']))] < 142:
                image.y = image.sources['ycentroid'][np.where(image.sources['peak'] == max(image.sources['peak']))]
            else:
                pass
        except Exception, mes:
            print '===================Error==================='
            print 'Could not find sources in image:', mes
            pass

    def main(image):
        '''
        BahtinovSpikes :    Fits the lines of the Bahtinov spikes by first finding exclusion zones which are unfit to be used for the fit,
                            this is achieved by finding lower and upper thresholds. These are set by the user and can be modified.
        '''
        today_utc_date = datetime.datetime.utcnow().strftime('%Y-%m-%d')

        innerthreshold = 0 ; outerthreshold = 0 ; outerthreshold1 = 0 ; innerthreshold1 = 0                     # Start with no thresholds for calculations
        x = [] ; x1 = [] ; x2 = [] ; y = [] ; y1= [] ; y2 = [] ; yerr = [] ; yerr1 = [] ; yerr2 = []            # Create empty arrays for positions of diffraction spikes
        xdata = np.linspace(0,len(image.data_new),len(image.data_new))                                          # Create x values for plotting of pixel values
        '''
        Find the thresholds between which the program should calculate the points which are used to fit the straight lines. Threshold is set by
        setting lower of upper limits on the difference between the max value and mean of a slice in the main image, limits found by trail and error.
        '''
        for i in xrange(len(image.data_new)):
            scan = image.data_new[:,i] - image.median_new                       # Create slice minus median of cutout image
            # Determine threshold at left half of image
            if outerthreshold == 0 :
                if abs(max(scan) - np.mean(scan)) > 35:
                    outerthreshold = i
            if innerthreshold == 0 :
                if abs(max(scan) - np.mean(scan)) > 450:
                    innerthreshold = i
            # Determine new threshold at right half of image
            if i > image.x:
                if innerthreshold1 == 0 :
                    if abs(max(scan) - np.mean(scan)) < 550:
                        innerthreshold1 = i
                if outerthreshold1 == 0 :
                    if abs(max(scan) - np.mean(scan)) < 100:
                        outerthreshold1 = i
        outerthreshold=60
        innerthreshold=100
        innerthreshold1=170
        outerthreshold1=230
        '''
        Loop again through all the slices to determine the points to be used for the straight line fit.
        '''
        image.fig, image.axis = plt.subplots(figsize = (10,10))
        for i in xrange(len(image.data_new)):
            scan = image.data_new[:,i] - image.median_new                       # Create slice minus median of cutout image
            scan[:100] = np.zeros(100)                                          # Section which is not relevant
            scan[-100:] = np.zeros(100)                                         # Section which is not relevant
            # Only if all thresholds are non-zero the fit can start
            if outerthreshold != 0 and innerthreshold != 0 and innerthreshold1 != 0 and outerthreshold1 != 0:
                # Only slices/scan between the thresholds are relevant
                if (outerthreshold < i < innerthreshold) or (innerthreshold1 < i < outerthreshold1):
                    threshold = (np.max(scan)*0.3 - np.min(scan)) / (np.max(scan) - np.min(scan))       # sets threshold for peak detection
                    peakindex = peakutils.indexes(np.array(scan), thres=threshold, min_dist=15 )        # find peaks in the slice/scan
                    values = [] ; Y = []
                    if len(peakindex) == 0:
                        print '=================# Error #================='
                        print 'No clear peaks were found to initate the fit.'
                        break
                    for w in peakindex:
                        values.append(scan[w])
                        Y.append(xdata[w])
                        index = sorted(np.array(values).argsort()[-3:])
                    # Only if at least three peaks are found continue
                    if len(index) >= 3:
                        parguess = (values[index[0]], Y[index[0]], 3, values[index[1]], Y[index[1]], 3, values[index[2]], Y[index[2]], 3)
                        fitobj = kmpfit.Fitter(residuals=focusfunctions.lorentzianresiduals, data=(xdata, scan))
                        # Try to fit using the guesses obtained from peak detection and append relevant values to position arrays
                        try:
                            fitobj.fit(params0 = parguess)
                            # Distinction between the central and diagonal peaks for left and right half of image
                            if i <= image.x:                                    # Left half of image
                                if (image.x - 40 < fitobj.params[1] < image.x - 15) :
                                    y.append(fitobj.params[1])    ; x.append(i)
                                    yerr.append(fitobj.stderr[1])
                                if (image.x - 10 < fitobj.params[4] < image.x + 10) :
                                    y1.append(fitobj.params[4])   ; x1.append(i)
                                    yerr1.append(fitobj.stderr[4])
                                if (image.x + 10 < fitobj.params[7] < image.x + 40) :
                                    y2.append(fitobj.params[7]) ; x2.append(i)
                                    yerr2.append(fitobj.stderr[7])
                            if i > image.x:                                     # Right half of image
                                if (image.x + 10 < fitobj.params[7] < image.x + 40) :
                                    y.append(fitobj.params[7])    ; x.append(i)
                                    yerr.append(fitobj.stderr[7])
                                if (image.x - 10 < fitobj.params[4] < image.x + 10) :
                                    y1.append(fitobj.params[4])   ; x1.append(i)
                                    yerr1.append(fitobj.stderr[4])
                                if (image.x - 40 < fitobj.params[1] < image.x - 15) :
                                    y2.append(fitobj.params[1])   ; x2.append(i)
                                    yerr2.append(fitobj.stderr[1])

                        # Skip if something went wrong with fit
                        except Exception, mes:
                            print '=================# Error #================='
                            print 'Something wrong with curve fit:', mes
                            print 'This fit is being skipped.'
                            pass

        # Fit the arrays (which are the lines of the spikes) by a linear line
        image.axis.scatter(x,y, s = 20, color = 'r')
        image.axis.scatter(x1,y1, s = 20, color = 'g')
        image.axis.scatter(x2,y2, s = 20, color = 'c')
        fitobj = kmpfit.Fitter(residuals=focusfunctions.residuals, data=(np.array(x),np.array(y), np.array(yerr)))
        fitobj1 = kmpfit.Fitter(residuals=focusfunctions.residuals1, data=(np.array(x1),np.array(y1), np.array(yerr1)))
        fitobj2 = kmpfit.Fitter(residuals=focusfunctions.residuals2, data=(np.array(x2),np.array(y2), np.array(yerr2)))
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
            image.X0 = xdata ; image.Y0 = image.angle * np.array(xdata) + image.intercept
            image.X1 = xdata ; image.Y1 = 0 * np.array(xdata) + image.intercept1
            image.X2 = xdata ; image.Y2 = -image.angle * np.array(xdata) + image.intercept2
            image.XY = zip(image.X0, image.Y0)
            image.XY1 = zip(image.X1, image.Y1)
            image.XY2 = zip(image.X2, image.Y2)

            '''
            Finds the shortest path between intersection of the X spikes and the | spike determined from the BahtinovSpikes function and calculate focus in microns.
            Differentiates between inside and outside focus; defined as inside focus being to the left of the | spike and outside focus on the right of the | spike.
            '''
            line1 = LineString([(image.XY[0][0], image.XY[0][1]), (image.XY[-1][0], image.XY[-1][1])])
            line2 = LineString([(image.XY2[0][0], image.XY2[0][1]), (image.XY2[-1][0], image.XY2[-1][1])])
            image.point = line1.intersection(line2)
            image.line = LineString([(image.XY1[0][0], image.XY1[0][1]), (image.XY1[-1][0], image.XY1[-1][1])])

            # Error in intersection position
            a = image.angle ; c = image.intercept[0]
            b = -image.angle ; d = image.intercept2[0]
            s_a = 0 ; s_c = image.std[0]
            s_b = 0 ; s_d = image.std2[0]
            s_ad = (a*d)**2 * ( (s_a/a)**2 + (s_d/d)**2 ) ; s_bc = (b*c)**2 * ( (s_b/b)**2 + (s_c/c)**2 )
            sigma2_x = (((d-c) / (a-b))**2 * ( ((s_c**2 + s_d**2)/(d-c))**2 + ((s_a**2 + s_b**2)/(a-b))**2 ))
            sigma2_y = (((a*d-b*c)/(a-b))**2 * ( (((s_ad + s_bc) / (a*d - b*c)))**2 + ((s_a**2 + s_b**2)/(a-b))**2  ))

            # Calculate defocus and define intra- and extra defocus (negative or positive)
            if np.array(image.point)[1] > image.intercept1:
                image.Focus =  -image.point.distance(image.line) * 9
            else:
                image.Focus =  image.point.distance(image.line) * 9
            image.focuserr = (sigma2_y + image.std1**2)**.5
            image.point1 = image.line.interpolate(image.line.project(image.point))

            '''
            Saves the focus values with errors to a .txt file to be used later.
            '''
            if os.path.exists('Focusrun/' + today_utc_date + '/Results/FocusResults.txt'):
                Results = np.loadtxt('Focusrun/' + today_utc_date + '/Results/FocusResults.txt')
                values = np.array([image.number, image.offset, image.Focus, image.focuserr, image.X, image.Y]).flatten()
                Results = np.vstack((Results, values))
                np.savetxt('Focusrun/' + today_utc_date + '/Results/FocusResults.txt', Results, fmt = '%10.1f %10.1f %10.5f %10.5f %10.3f %10.3f')

            else:
                Results = np.zeros((1,6))
                Results[0,0] = image.number ; Results[0,1] = image.offset
                Results[0,2] = image.Focus ; Results[0,3] = image.focuserr
                Results[0,4] = image.X ; Results[0,5] = image.Y
                np.savetxt('Focusrun/' + today_utc_date + '/Results/FocusResults.txt', Results, fmt = '%10.1f %10.1f %10.5f %10.5f %10.3f %10.3f')

            if os.path.exists('Focusrun/' + today_utc_date + '/Results/FocusCCDResults_' + str(image.name)+ '.txt'):
                Results = np.loadtxt('Focusrun/' + today_utc_date + '/Results/FocusCCDResults_' + str(image.name)+ '.txt')
                values = np.array([image.number, image.offset, image.Focus, image.focuserr, image.X, image.Y]).flatten()
                Results = np.vstack((Results, values))
                np.savetxt('Focusrun/' + today_utc_date + '/Results/FocusCCDResults_' + str(image.name)+ '.txt', Results, fmt = '%10.1f %10.1f %10.5f %10.5f %10.3f %10.3f')

            else:
                Results = np.zeros((1,6))
                Results[0,0] = image.number ; Results[0,1] = image.offset
                Results[0,2] = image.Focus ; Results[0,3] = image.focuserr
                Results[0,4] = image.X ; Results[0,5] = image.Y
                np.savetxt('Focusrun/' + today_utc_date + '/Results/FocusCCDResults_' + str(image.name)+ '.txt', Results, fmt = '%10.1f %10.1f %10.5f %10.5f %10.3f %10.3f')

            # Plot star with fitted diffraction spikes
            image.axis.imshow((image.data_new - np.min(image.data_new))*(10**7), cmap='Greys' , origin='lower')
            image.axis.scatter(image.x, image.y, color = 'r')
            image.axis.set_xlim(0,len(image.data_new)) ; image.axis.set_ylim(0,len(image.data_new))
            image.axis.set_xlabel('x') ; image.axis.set_ylabel('y')
            image.axis.set_title('Bahtinov Source: %s (%.2f, %.2f)' %(image.name, image.X, image.Y))
            image.axis.plot(image.X0, image.Y0, color = 'r')
            image.axis.plot(image.X1, image.Y1, color = 'g')
            image.axis.plot(image.X2, image.Y2, color = 'r')
            image.axis.add_line( Line2D((image.point.x,image.point1.x) , (image.point.y, image.point1.y), color = 'k') )
            image.axis.annotate('Focus distance = %.2f $\pm$ %.3f $\\mu m$' %(image.Focus, image.focuserr), xy=(0, 1), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='top')
            image.fig.savefig('Focusrun/' + today_utc_date + '/Plots/Source_' + str(image.title) + '_' + str(image.X) + '_' + str(image.Y) + '_' + '.png')
            # Create zoom in image
            #image.axis.set_xlim(image.x-25,image.x+25) ; image.axis.set_ylim(image.y-25,image.y+25)
            #image.fig.savefig('Focusrun/' + today_utc_date + '/Plots/Source_Zoomin_' + str(image.name) + '_' + str(image.X) + '_' + str(image.Y) + '_'  + '.png')
            plt.close()

        #If anything goes wrong with fit skip the star
        except Exception, mes:
            print "Something wrong with fit: ", mes
            print 'This star was not fitted correctly, going to next star.'
            print 'Due to absence of star, no data is saved.'
            pass
