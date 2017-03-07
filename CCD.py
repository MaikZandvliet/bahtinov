#!/usr/bin/env python

# ====================================================================================
#                   Determine CCD Surface via Bahtinov Defocus
# ====================================================================================

# Import required modules
from __future__ import division
from tiptilt import image
import time
import datetime
import os
import subprocess
import glob
import numpy as np
from tqdm import *
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.interpolate
from matplotlib.pyplot import figure, show, rc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits
from scipy.interpolate import griddata
from pylab import rcParams
# Set figure parameters and date
rcParams['figure.figsize'] = 12, 10
rc('font', size=18)
rc('legend', fontsize=18)
start = time.time()
today_utc_date = datetime.datetime.utcnow().strftime('%Y-%m-%d')
today_utc_time = time.strftime('%c', time.gmtime(time.time()))

print '=============== CCD Fit Started ==============='
print 'UTC:', today_utc_time

class CCDSurface:
    def __init__(self, name):
        #print name
        self.pixelsize = 9e-3                                                  # 9 micron per pixel to mm per pixel
        #self.data = np.loadtxt(directory +'Focusrun/' + today_utc_date + '/Results/' + str(name) + '/FocusResults.txt')
        self.data = np.loadtxt('/media/data/bahtinov_results/' + name + '.txt')
        #self.data = np.loadtxt('Focusrun/2016-12-21/Results/FocusResults.txt')
        self.x = self.data[:,4]#*self.pixelsize
        self.y = self.data[:,5]#*self.pixelsize
        self.z = self.data[:,2]

    def fitplane(self):
        # best-fit linear plane
        data = np.c_[self.x, self.y, self.z]
        A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        C, res, rank, s = scipy.linalg.lstsq(A, data[:,2])
        # evaluate it on grid
        X, Y = np.meshgrid([0, 10560], [0, 10600])
        Z = C[0]*X + C[1]*Y + C[2]
        ZZ = 0 * X
        tip = C[0]
        tilt = C[1]
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        axis.set_title('CCD Surface Orientation', fontsize = 14)
        axis.annotate('Tip  = %.4f \nTilt = %.4f' %(tip,tilt), xy=(0, 1), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='top')
        axis.scatter(data[:,0], data[:,1], data[:,2], c='r', s=20)
        axis.plot_surface(X, Y, ZZ, alpha=0.05, color='r', label = 'Horizontal')
        axis.plot_surface(X, Y, Z, color = 'g', rstride=1, cstride=1, alpha=0.2, label = 'CCD surface')
        #show()
        axis.set_xlabel('x [mm]')
        axis.set_ylabel('y [mm]')
        axis.set_zlabel('Defocus [$\mu m$]')
        axis.set_xlim(0, 10560)
        axis.set_ylim(0, 10600)
        axis.view_init(elev=20, azim=225)
        fig.savefig(directory + 'Focusrun/' + today_utc_date + '/Results/CCD_surface/Planefit_'+ str(name) +'_angled.png')
        axis.view_init(elev=0, azim=90)
        fig.savefig(directory + 'Focusrun/' + today_utc_date + '/Results/CCD_surface/Planefit_'+ str(name) +'_front.png')
        axis.view_init(elev=0, azim=180)
        fig.savefig(directory + 'Focusrun/' + today_utc_date + '/Results/CCD_surface/Planefit_'+ str(name) +'_side.png')
        plt.close()
        #show()

    def ccd_sufrace_contour(self, name):
        #data = fits.open('/media/data/Bahtinov/temp_12000x10600_890_test.fits')[1].data
        #fig = figure()
        #axis = fig.add_subplot(111)
        #axis.imshow(data, cmap = 'gray', interpolation = 'nearest', origin = 'lower')
        #axis.scatter(self.x,self.y)
        xi, yi = np.linspace(0, 10560, 100), np.linspace(0, 10600, 100)
        xi, yi = np.meshgrid(xi, yi)
        rbf = scipy.interpolate.Rbf(self.x, self.y, self.z, function='linear')
        zi = rbf(xi, yi)
        fig = figure()
        axis = fig.add_subplot(111)
        #CS = axis.tricontourf(self.x, self.y, self.z, 100)
        axis.scatter(self.x, self.y)
        CS = axis.imshow(zi, vmin=self.z.min(), vmax=self.z.max(), origin='lower', extent=[0, 10560, 0, 10600])
        axis.set_xlim(0, 10560)
        axis.set_ylim(0, 10600)
        axis.set_title('Defocus CCD Surface', fontsize = 22)
        axis.set_xlabel('x [pixel]')
        axis.set_ylabel('y [pixel]')
        cbar = plt.colorbar(CS, shrink = 0.7, label='defocus [micron]')
        fig.tight_layout()
        fig.savefig('/media/data/bahtinov_results/CCD_surface' + str(name) + '.png')
        #show()

# ====================================================================================
#                                   Start
# ====================================================================================
if __name__ == '__main__':
    directory_prefix = '/media/data/'
    directory = directory_prefix + 'bahtinov_results/'                 # Directory containing images
    file = '/media/data/bahtinov_results/' + today_utc_date + '/Results/FocusResults.txt'
    files = sorted(glob.glob(directory + '*.txt'))
    #if not os.path.exists(directory + 'Focusrun/' + today_utc_date + '/Results/CCD_surface/'):
    #    subprocess.call(('mkdir ' + directory + 'Focusrun/' + today_utc_date + '/Results/CCD_surface/').format(directory), shell=True)
    #subprocess.call(('rm ' + directory + 'Focusrun/' + today_utc_date + '/Results/CCD_surface/' + '/*.png').format(directory_prefix), shell=True)

    # ====================================================================================

    #Name = file.split('/')[-1].split('.')[0]
    #CCDSurface().fitplane()
    #Surface = CCDSurface()
    #Surface.ccd_sufrace_contour(Name)

    for k in xrange(0,len(files)):
        name = files[k].split('/')[-1].split('.')[0]
        surface = CCDSurface(name)
        surface.ccd_sufrace_contour(name)



    period = time.time() - start
    print '\nThe computation time was %.3f seconds\n' %(period)
    print '=============== CCD Fit Ended ==============='
