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
from skimage.feature import canny
import time
import subprocess
import tiptilt
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib import cm
from astropy.stats import sigma_clip
from astropy.nddata import Cutout2D
from astropy.io import fits
from kapteyn import kmpfit
import matplotlib
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from matplotlib.pyplot import figure, show
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
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
        image.bias = fits.open('/media/data/Bahtinov/2017_03_15/Bias/Masterbias.fits')
        image.bias = image.bias[0].data
        image.workdir = workdir
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

    def calculate_focus(image, outerline0, centralline, outerline1):
        line1 = LineString([(outerline0[0][0], outerline0[0][1]), (outerline0[-1][0], outerline0[-1][1])])
        line2 = LineString([(outerline1[0][0], outerline1[0][1]), (outerline1[-1][0], outerline1[-1][1])])
        # Calculate intersection
        diagonal_line_intersection = line1.intersection(line2)
        # Only if intersection location is close to center image
        if abs(np.array(diagonal_line_intersection)[0] - image.x) < 3:
            line_center = LineString([(centralline[0][0], centralline[0][1]), (centralline[-1][0], centralline[-1][1])])
            if np.array(diagonal_line_intersection)[1] > image.intercept1:
                focus = -diagonal_line_intersection.distance(line_center) / 2 * 9 * (33000 / image.delta)
            else:
                focus = diagonal_line_intersection.distance(line_center) / 2 * 9 * (33000 / image.delta)
        else:
            focus = None
        return focus

    def houghtransform(image, star_counter):
        image.star_counter = star_counter

        edges = canny(image.data_new, 3, .25, 25)
        lines = probabilistic_hough_line(edges, theta = np.array([-image.angle + np.pi/2, np.pi/2, image.angle + np.pi/2]), threshold = 10, line_length = 20, line_gap = 50)
        xdata = np.linspace(0,len(image.data_new),len(image.data_new))

        line0_intercept = [] ; line1_intercept = [] ; line2_intercept = []
        for line in lines:
            p0, p1 = line
            x = p0[0] ; x1 = p1[0]
            y = p0[1] ; y1 = p1[1]
            a = (y1 - y) / (x1 - x)
            b = y - a * x
            Y = a * xdata + b
            if (130 < b < 150) and a == 0:
                line0_intercept.append(b)
            if (180 < b < 195):
                line1_intercept.append(b)
            if (80 < b < 95):
                line2_intercept.append(b)

        if len(line0_intercept) >= 2 and len(line1_intercept) >= 2 and len(line2_intercept) >= 2:
            center_line_intercept = np.mean(line0_intercept)
            upper_line_intercept = np.mean(line2_intercept)
            lower_line_intercept = np.mean(line1_intercept)
            Y_center = 0 * xdata + center_line_intercept
            Y_upper = image.angle*xdata + upper_line_intercept
            Y_lower = -image.angle*xdata + lower_line_intercept
            image.intercept = lower_line_intercept
            image.intercept1 = center_line_intercept
            image.intercept2 = upper_line_intercept
            image.XY = zip(xdata, Y_lower)
            image.XY1 = zip(xdata, Y_center)
            image.XY2 = zip(xdata, Y_upper)
            image.Focus = image.calculate_focus(image.XY, image.XY1, image.XY2)
            if image.Focus != None:

                if os.path.exists(image.workdir +'Focusrun/' + today_utc_date + '/Results/FocusResults.txt'):
                    Results = np.loadtxt(image.workdir +'Focusrun/' + today_utc_date + '/Results/FocusResults.txt')
                    values = np.array([image.number, image.offset, image.Focus, image.X, image.Y, image.SNR]).flatten()
                    Results = np.vstack((Results, values))
                    np.savetxt(image.workdir +'Focusrun/' + today_utc_date + '/Results/FocusResults.txt', Results, fmt = '%10.1f %10.1f %10.5f %10.3f %10.3f %10.3f')

                else:
                    Results = np.zeros((1,6))
                    Results[0,0] = image.number ; Results[0,1] = image.offset
                    Results[0,2] = image.Focus ; Results[0,3] = image.X
                    Results[0,4] = image.Y ; Results[0,5] = image.SNR
                    np.savetxt(image.workdir +'Focusrun/' + today_utc_date + '/Results/FocusResults.txt', Results, fmt = '%10.1f %10.1f %10.5f %10.3f %10.3f %10.3f')

                image.fig, image.axis = plt.subplots(figsize = (10,10))
                image.axis.imshow(image.data_new, cmap=cm.gray, norm = matplotlib.colors.LogNorm(vmin=0.01, vmax = np.max(image.data_new)), origin = 'lower')
                image.axis.scatter(image.x, image.y, color = 'r')
                image.axis.set_xlim(0,len(image.data_new)) ; image.axis.set_ylim(0,len(image.data_new))
                image.axis.set_xlabel('x') ; image.axis.set_ylabel('y')
                image.axis.set_title('Bahtinov Source: %s (%.2f, %.2f)' %(image.name, image.X, image.Y))
                image.axis.plot(zip(*image.XY)[0], zip(*image.XY)[1], color = 'r')
                image.axis.plot(zip(*image.XY1)[0], zip(*image.XY1)[1], color = 'g')
                image.axis.plot(zip(*image.XY2)[0], zip(*image.XY2)[1], color = 'r')
                image.axis.annotate('Focus distance = %.2f $\\mu m$' %(image.Focus), xy=(1, -.06), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
                image.fig.savefig(image.workdir +'Focusrun/' + today_utc_date + '/Plots/' + str(image.name) + '/' + str(image.name) + '_' + str(image.X) + '_' + str(image.Y) + '.png')
                plt.close()
            else:
                image.star_counter += 1
        else:
            image.star_counter += 1
        return image.star_counter
