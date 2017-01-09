#!/usr/bin/env python

# ====================================================================================
#            Bahtinov Mask Analysis Software to Determine Best M2 Offset
# ====================================================================================

# Import required modules
from __future__ import division
import time
import datetime
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
from matplotlib.pyplot import figure, show, rc
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from astropy.io import fits
from matplotlib.ticker import AutoMinorLocator
from kapteyn import kmpfit
import functools
import focusfunctions
import Bahtinov
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
rc('font', size=12)
rc('legend', fontsize=12)
start = time.time()
today_utc_date = datetime.datetime.utcnow().strftime('%Y-%m-%d')
today_utc_time = time.strftime('%c', time.gmtime(time.time()))

print '=============== Focus Run Analysis Started ==============='
print today_utc_time

def FocusRunResults():
    data = np.loadtxt('Focusrun/' + today_utc_date + '/Results/FocusResults.txt')
    #data = np.loadtxt('Focusrun/2016-12-22/Results/FocusResults.txt')
    image = data[:,0] ; defocus = data[:,1] ; focus = data[:,2] ; focuserr = data[:,3]
    xp = np.linspace(min(defocus), max(defocus), 100)
    z = np.polyfit(defocus, focus, 1)
    fitobj = kmpfit.Fitter(residuals=focusfunctions.linfitresiduals, data=(defocus, focus, focuserr),
                           xtol=1e-12, gtol=1e-12)
    fitobj.fit(params0=z)
    print "\n=================== Results kmpfit ==================="
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
    ydummy, upperband, lowerband = focusfunctions.confidence_band(xp, dfdp, confprob, fitobj, focusfunctions.linfit)
    verts = zip(xp, lowerband) + zip(xp[::-1], upperband[::-1])
    bestfocus = -fitobj.params[0]/fitobj.params[1]
    bestfocusvar = bestfocus**2 * ( (fitobj.stderr[0] / fitobj.params[0])**2 + (fitobj.stderr[1] / fitobj.params[1])**2 - 2 * (fitobj.covar[0,1]/(fitobj.params[0]*fitobj.params[1])))
    fig, axis = plt.subplots()
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.errorbar(defocus,focus, yerr = focuserr, fmt = 'ko')
    axis.annotate('Best offset M2 = %.2f $\pm$ %.3f $\\mu m$' %(bestfocus, bestfocusvar**.5), xy=(0, 1), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='top')
    axis.plot(xp, focusfunctions.linfit(fitobj.params,xp), color = 'r', ls='--', lw=2,
        label = 'linear fit \n $\chi ^2_{reduced}$ = %.3f' %(fitobj.rchi2_min))
    axis.set_title('Focus run M2 vs calculated defocus', fontsize = 14)
    axis.set_xlabel('Offset M2 [$\mu m$]')
    axis.set_ylabel('Calculated defocus [$\mu m$]')
    axis.set_xlim(90,210)
    axis.grid(True)
    axis.legend(loc=4,fancybox=True, shadow=True, ncol=4, borderpad=1.01)
    fig.tight_layout()
    fig.savefig('Focusrun/' + today_utc_date + '/Results/Focusrun_defocus_results.png')
    #show()

def Exclusionzones():
    thresholds = np.loadtxt('Focusrun/Results/Focusthresholds.txt')
    inner = thresholds[:,1]
    outer = thresholds[:,0]
    #inner = np.concatenate([thresholds[:,1],thresholds[:,2]])
    #outer = np.concatenate([thresholds[:,0],thresholds[:,3]])
    fig, axis = plt.subplots()
    axis.scatter(inner,outer, color = 'b', marker = 'o')
    axis.axvline(40)
    triangle = Polygon([(0, 0), (140, 0), (140, 140)], closed=True, fc='c', ec='c', alpha=0.3)
    axis.add_patch(triangle)
    axis.set_ylim(0,140) ; axis.set_xlim(0,40)
    axis.set_xlabel('Inner zone') ; axis.set_ylabel('Outer zone')
    fig.tight_layout()
    #show()

# ====================================================================================
#                                   Start script
# ====================================================================================
Offset = [100,110,120,130,140,150,160,170,180,190,200,200,190,180,170,160,150,140,130,120,110,100]  # Offsets of M2
directory_prefix = '/media/maik/Maik/MeerLICHT/'
directory = directory_prefix + 'Data/2016_09_26/Fits_from_raw/New_images/Focusrun/'                 # Directory containing images
files = glob.glob(directory + '*.fits')
filescutout = glob.glob(directory_prefix + 'Data/Cutout/*.fits')

S = 400
'''
subprocess.call(('rm Focusrun/' + today_utc_date + '/Results/FocusResults.txt').format(directory_prefix), shell=True)
for k in tqdm(xrange(0,len(files))):
    name = files[k].split('/')[-1].split('.')[0]
    X_pos, Xerr_pos, Y_pos, Yerr_pos = np.loadtxt('SExtractor/' + str(name) + '_reduced.txt', usecols = (0,1,2,3), unpack = True)
    subprocess.call(('rm Focusrun/' + today_utc_date + '/Results/FocusCCDResults_' + str(name)+ '.txt').format(directory_prefix), shell=True)
    for p in xrange(0,len(X_pos)):
        Image = Bahtinov.Bahtinov(files[k], X_pos[p], Xerr_pos[p] ,Y_pos[p], Yerr_pos[p], Offset[k], k, p, S)
        #Image.FindSources()
        Image.main()
        #Image.Show()
'''
# ====================================================================================

FocusRunResults()
#Exclusionzones()
subprocess.call(('python CCD.py'), shell=True)

'''
bias_nofilter = fits.open('/media/maik/Maik/MeerLICHT/Data/2016_09_21/Fits_from_raw/ML__12000x10600_113.fits')[0].data[400:800,400:800]
data = np.uint8(fits.open(filescutout[67])[1].data)-bias_nofilter
#print data.dtype
data = np.uint8(data - np.min(data)+1)
#print data.min()
print np.max(data)
#data = cv2.imread('/home/maik/Pictures/Bahtinov_mask_example.jpg')
#gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(data,5,5)
lines = cv2.HoughLinesP(image = data, rho = .8, theta = np.pi/500, threshold=70, minLineLength = 150, maxLineGap = 1000)

x, y, x1, y1, theta = [], [], [], [], []
for i in range(len(lines)):
    for x_, y_, x1_, y1_ in lines[i]:
        dy = y_ - y1_
        dx = x_ - x1_
        theta_ = np.arctan(dy/dx)
        theta_ *= 180/np.pi # rads to degs

        x.append(x_)
        y.append(y_)
        x1.append(x1_)
        y1.append(y1_)
        theta.append(theta_)
        #cv2.line(data,(x_,y_), (x1_,y1_),(0,0,255),1)

lines = np.column_stack((x,y,x1,y1, theta))
print lines
theta1, theta2, theta3 = [], [], []
central, diag, diag1 = np.zeros((1,4)), np.zeros((1,4)), np.zeros((1,4))
for index, (x_, y_, x1_, y1_, t_) in enumerate(lines):
    if abs(t_ - 47) < 1:
        central = np.vstack((central,(x_, y_, x1_, y1_)))
        theta1.append(t_)
    if abs(t_ - 25) < 1:
        diag = np.vstack((diag, (x_, y_, x1_, y1_)))
        theta2.append(t_)
    if abs(t_ - 65) < 1:
        diag1 = np.vstack((diag1, (x_, y_, x1_, y1_)))
        theta3.append(t_)

central = np.delete(central, (0), axis=0)
diag = np.delete(diag, (0), axis=0)
diag1 = np.delete(diag1, (0), axis=0)
central = central.mean(axis=0)
diag = diag.mean(axis=0)
diag1 = diag1.mean(axis=0)
Lines = np.vstack((diag, central, diag1))
print Lines
fig, axis = plt.subplots(figsize = (10,10))
for x_, y_, x1_, y1_ in Lines:
    #cv2.line(data,(x_,y_), (x1_,y1_),(255,0,0),1)
    axis.imshow(data, cmap='Greys' , origin='lower')
    axis.set_xlabel('x') ; axis.set_ylabel('y')
    axis.add_line(Line2D((x_,x1_), (y_,y1_), color = 'r') )

line1 = LineString([(diag[0],diag[1]), (diag[2],diag[3])])
line2 = LineString([(diag1[0],diag1[1]), (diag1[2],diag1[3])])
point = line1.intersection(line2)
line = LineString([(central[0],central[1]), (central[2],central[3])])
x_, y_ = line1.coords[0]
x_1, y_1 = line1.coords[1]

a = (y_1-y_)/(x_1-x_)
b = y_ - a*x_
#print a,b

Focus =  point.distance(line)
#focuserr = (std**2 + std1**2 + std2**2)**.5
print Focus
#cv2.imwrite('houghlines3.jpg',data)
show()
'''

period = time.time() - start
print '\nThe computation time was %.3f seconds\n' %(period)
print '=============== Focus Run Analysis Ended ==============='
