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
from numpy.polynomial import polynomial
from mayavi import mlab
from scipy.interpolate.rbf import *
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
        #self.data = np.loadtxt(directory +'Focusrun/' + today_utc_date + '/Results/' + str(name) + '/FocusResults.txt')
        self.data = np.loadtxt(directory_prefix + 'Focusrun/' + today_utc_date + '/Results/' + name + '.txt')
        self.x = self.data[:,4]#*self.pixelsize
        self.y = self.data[:,5]#*self.pixelsize
        self.z = self.data[:,2]
        self.zerr = self.data[:,3]
        self.snr = self.data[:,6]

    def fitplane(self):
        # best-fit linear plane
        data = np.c_[self.x, self.y, self.z]
        # regular grid covering the domain of the data
        X, Y = np.linspace(0, 10560, 50), np.linspace(0, 10600, 50)
        X, Y = np.meshgrid(X, Y)
        XX = X.flatten()
        YY = Y.flatten()

        order = 2    # 1: linear, 2: quadratic
        if order == 1:
            # best-fit linear plane
            A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
            C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients

            # evaluate it on grid
            Z = C[0]*X + C[1]*Y + C[2]

            # or expressed using matrix/vector product
            #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

        elif order == 2:
            # best-fit quadratic curve
            A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
            C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])

            # evaluate it on a grid
            Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
            Z = C[4]*X**2. + C[5]*Y**2. + C[3]*X*Y + C[1]*X + C[2]*Y + C[0]
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        axis.set_title('CCD Surface Orientation', fontsize = 14)
        #axis.annotate('Tip  = %.4f \nTilt = %.4f' %(tip,tilt), xy=(0, 1), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='top')
        axis.scatter(data[:,0], data[:,1], data[:,2], c='r', s=20)
        #axis.plot_surface(X, Y, ZZ, alpha=0.05, color='r', label = 'Horizontal')
        axis.plot_surface(X, Y, Z, color = 'g', rstride=1, cstride=1, alpha=0.2, label = 'CCD surface')
        axis.set_xlabel('x [mm]')
        axis.set_ylabel('y [mm]')
        axis.set_zlabel('Defocus [$\mu m$]')
        axis.set_xlim(0, 10560)
        axis.set_ylim(0, 10600)
        axis.view_init(elev=20, azim=225)
        #fig.savefig(directory_prefix + 'Focusrun/' + today_utc_date + '/Results/Planefit_'+ str(name) +'_angled.png')
        axis.view_init(elev=0, azim=90)
        #fig.savefig(directory_prefix + 'Focusrun/' + today_utc_date + '/Results/Planefit_'+ str(name) +'_front.png')
        axis.view_init(elev=0, azim=180)
        #fig.savefig(directory_prefix + 'Focusrun/' + today_utc_date + '/Results/Planefit_'+ str(name) +'_side.png')
        #plt.close()
        show()

    def polyfit2d(self, x, y, z, order=1):
        ncols = (order + 1)**2
        G = np.zeros((x.size, ncols))
        ij = itertools.product(range(order+1), range(order+1))
        for k, (i,j) in enumerate(ij):
            G[:,k] = x**i * y**j
        m, res, rank, s = np.linalg.lstsq(G, z)
        return G, m, res, rank, s

    def polyval2d(self, x, y, m):
        order = int(np.sqrt(len(m))) - 1
        ij = itertools.product(range(order+1), range(order+1))
        z = np.zeros_like(x)
        for a, (i,j) in zip(m, ij):
            z += a * x**i * y**j
        return z


    def ccd_sufrace_contour(self, name):
        xi, yi = np.linspace(0, 10560, len(self.x)), np.linspace(0, 10600, len(self.y))
        xi, yi = np.meshgrid(xi, yi)
        x_ , y_ = np.meshgrid(self.x, self.y)

        #self.polynom(self.x,self.y,self.z)
        G, m, res, rank, s = self.polyfit2d(self.x,self.y,self.z)
        zz  = self.polyval2d(xi, yi, m)
        z_ = self.polyval2d(x_, y_, m)
        #for i in xrange(0,len(zz)):



        #rbf = scipy.interpolate.Rbf(self.x, self.y, self.z, function='linear')
        #zz = rbf(xi, yi)
        #z_ = rbf(x_, y_)
        chi_red = 1/(len(self.x)-(int(np.sqrt(len(m))) - 1)) * np.sum((zz-z_)**2 / self.zerr**2)

        print chi_red
        #s = self.snr/10
        #mlab.figure(size = (1920/2,1080/2))
        #mlab.mesh(xi,yi,zz)
        #mlab.points3d(self.x, self.y, self.z, s)
        #mlab.colorbar(orientation='vertical', nb_labels=10, label_fmt = '%.1f')
        #mlab.show()

        '''
        fig = figure()
        axis = fig.add_subplot(111)
        #CS = axis.tricontourf(self.x, self.y, self.z, 100)
        axis.scatter(self.x, self.y)
        #CS = axis.imshow(zi, vmin=(self.z).min(), vmax=(self.z).max(), origin='lower', extent=[0, 10560, 0, 10600])
        CS = axis.imshow(zz, extent=(xi.min(), yi.max(), xi.max(), yi.min()))
        axis.set_xlim(0, 10560)
        axis.set_ylim(0, 10600)
        axis.set_title('Defocus CCD Surface %s' %name, fontsize = 22)
        axis.set_xlabel('x [pixel]')
        axis.set_ylabel('y [pixel]')
        cbar = plt.colorbar(CS, shrink = 0.7, label='defocus [micron]')
        fig.tight_layout()
        fig.savefig('/media/data/bahtinov_results/Focusrun/' + today_utc_date + '/Results/CCD_surface' + str(name) + '.png')
        show()
        '''
        fig = figure()
        axis = fig.add_subplot(111, projection='3d')
        axis.plot_surface(xi, yi, zz, alpha=0.05, color='r', label = 'Horizontal')
        axis.scatter(self.x, self.y, self.z)
        for i in np.arange(0, len(self.x)):
            axis.plot([self.x[i], self.x[i]], [self.y[i], self.y[i]], [self.z[i], self.z[i]], marker="_")
            axis.plot([self.x[i], self.x[i]], [self.y[i], self.y[i]], [self.z[i], self.z[i]], marker="_")
            axis.plot([self.x[i], self.x[i]], [self.y[i], self.y[i]], [self.z[i]+self.zerr[i], self.z[i]-self.zerr[i]], marker="_")
        axis.set_xlim(0, 10560)
        axis.set_ylim(0, 10600)
        #axis.set_zlim(np.min(self.z) - 300, np.max(self.z) + 300)
        axis.set_title('Defocus CCD Surface %s' %name, fontsize = 22)
        axis.set_xlabel('x [pixel]')
        axis.set_ylabel('y [pixel]')
        axis.set_zlabel('[$\mu m$]')
        fig.tight_layout()
        fig.savefig('/media/data/bahtinov_results/Focusrun/' + today_utc_date + '/Results/CCD_surface3D' + str(name) + '.png')
        #show()


# ====================================================================================
#                                   Start
# ====================================================================================
if __name__ == '__main__':
    directory_prefix = '/media/data/bahtinov_results/'
    directory = directory_prefix + 'Focusrun/' + today_utc_date + '/Results/'                 # Directory containing images
    file = '/media/data/bahtinov_results/' + today_utc_date + '/Results/FocusResults.txt'
    files = sorted(glob.glob(directory + 'M2_*0.txt'))

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
