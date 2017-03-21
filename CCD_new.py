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
import matplotlib
import itertools
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
        self.data = np.loadtxt(directory_prefix + 'Focusrun/' + today_utc_date + '/Results/' + name + '.txt')
        self.x = self.data[:,3]#*self.pixelsize
        self.y = self.data[:,4]#*self.pixelsize
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
        axis.set_xlabel('x [mm]')
        axis.set_ylabel('y [mm]')
        axis.set_zlabel('Defocus [$\mu m$]')
        axis.set_xlim(0, 10560)
        axis.set_ylim(0, 10600)
        axis.view_init(elev=20, azim=225)
        fig.savefig(directory_prefix + 'Focusrun/' + today_utc_date + '/Results/Planefit_'+ str(name) +'_angled.png')
        axis.view_init(elev=0, azim=90)
        fig.savefig(directory_prefix + 'Focusrun/' + today_utc_date + '/Results/Planefit_'+ str(name) +'_front.png')
        axis.view_init(elev=0, azim=180)
        fig.savefig(directory_prefix + 'Focusrun/' + today_utc_date + '/Results/Planefit_'+ str(name) +'_side.png')
        #plt.close()
        show()

    def polyfit2d(self, x, y, z, order=3):
        ncols = (order + 1)**2
        G = np.zeros((x.size, ncols))
        ij = itertools.product(range(order+1), range(order+1))
        for k, (i,j) in enumerate(ij):
            G[:,k] = x**i * y**j
        m, _, _, _ = np.linalg.lstsq(G, z)
        return m

    def polyval2d(self, x, y, m):
        order = int(np.sqrt(len(m))) - 1
        ij = itertools.product(range(order+1), range(order+1))
        z = np.zeros_like(x)
        for a, (i,j) in zip(m, ij):
            z += a * x**i * y**j
        return z

    def ccd_sufrace_contour(self, name):
        xi, yi = np.linspace(0, 10560, 100), np.linspace(0, 10600, 100)
        xi, yi = np.meshgrid(xi, yi)

        #z_ = self.x**2 + self.y**2 + 3*self.x**3 + self.y
        #print z_
        m = self.polyfit2d(self.x,self.y,self.z)
        zz = self.polyval2d(xi, yi, m)
        #rbf = scipy.interpolate.Rbf(self.x, self.y, self.z, function='gaussian,')
        #zi = rbf(xi, yi)
        fig = figure()
        axis = fig.add_subplot(111)
        #CS = axis.tricontourf(self.x, self.y, self.z, 100)
        axis.scatter(self.x, self.y)
        #CS = axis.imshow(zi, vmin=(self.z).min(), vmax=(self.z).max(), origin='lower', extent=[0, 10560, 0, 10600])
        CS = axis.imshow(zz, extent=(self.x.min(), self.y.max(), self.x.max(), self.y.min()))
        axis.set_xlim(0, 10560)
        axis.set_ylim(0, 10600)
        axis.set_title('Defocus CCD Surface %s' %name, fontsize = 22)
        axis.set_xlabel('x [pixel]')
        axis.set_ylabel('y [pixel]')
        cbar = plt.colorbar(CS, shrink = 0.7, label='defocus [micron]')
        fig.tight_layout()
        fig.savefig('/media/data/bahtinov_results/Focusrun/' + today_utc_date + '/CCD_surface' + str(name) + '.png')

        fig = figure()
        axis = fig.add_subplot(111, projection='3d')
        axis.plot_surface(xi, yi, zz, alpha=0.05, color='r', label = 'Horizontal')
        axis.scatter(self.x, self.y, self.z)
        axis.set_xlim(0, 10560)
        axis.set_ylim(0, 10600)
        axis.set_title('Defocus CCD Surface %s' %name, fontsize = 22)
        axis.set_xlabel('x [pixel]')
        axis.set_ylabel('y [pixel]')
        axis.set_zlabel('[$\mu m$]')
        fig.tight_layout()
        fig.savefig('/media/data/bahtinov_results/Focusrun/' + today_utc_date + '/CCD_surface3D' + str(name) + '.png')
        show()

# ====================================================================================
#                                   Start
# ====================================================================================
if __name__ == '__main__':
    directory_prefix = '/media/data/bahtinov_results/'
    directory = directory_prefix + 'Focusrun/' + today_utc_date + '/Results/'                 # Directory containing images
    file = '/media/data/bahtinov_results/' + today_utc_date + '/Results/FocusResults.txt'
    files = sorted(glob.glob(directory + 'M2_*.txt'))
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
        #surface.fitplane()
        surface.ccd_sufrace_contour(name)



    period = time.time() - start
    print '\nThe computation time was %.3f seconds\n' %(period)
    print '=============== CCD Fit Ended ==============='
