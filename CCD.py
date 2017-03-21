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
import matplotlib.animation as animation
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import pybrain.datasets as pd
from matplotlib import cm
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
rc('font', size=12)
rc('legend', fontsize=12)
start = time.time()
today_utc_date = datetime.datetime.utcnow().strftime('%Y-%m-%d')
today_utc_time = time.strftime('%c', time.gmtime(time.time()))

print '=============== CCD Fit Started ==============='
print 'UTC:', today_utc_time


def get_plane(xl, xu, yl, yu, i):
    xx = np.arange(xl, xu, i)
    yy = np.arange(yl, yu, i)
    extrapolation_spots = np.zeros((len(xx) * len(yy), 2))
    count = 0
    for i in xx:
        for j in yy:
            extrapolation_spots[count, 0] = i
            extrapolation_spots[count, 1] = j
            count += 1
    return extrapolation_spots


def nearest_analysis(extrapolation_spots, data, name):
    data = (data - np.min(data)) #/ (np.max(data) - np.min(data))
    top_extra = extrapolation(data, extrapolation_spots, method = 'nearest')
    gridx_data, gridy_data, gridz_data = interpolation(top_extra)
    #plot(data, gridx_data, gridy_data, gridz_data, method = 'contour', title = name)
    plot(data, gridx_data, gridy_data, gridz_data, method = 'snaps', title = name)


def extrapolation(data, extrapolation_spots, method='nearest'):
    if method == 'kriging':
        xx, yy, zz, ss = kriging(data, extrapolation_spots)

        new_points = np.zeros((len(yy) * len(zz), 3))
        count = 0
        for i in range(len(xx)):
            for j in range(len(yy)):
                new_points[count, 0] = xx[i]
                new_points[count, 1] = yy[j]
                new_points[count, 2] = zz[i, j]
                count += 1
        combined = np.concatenate((data, new_points))
        return combined

    if method == 'nearest':
        new_points = np.zeros((len(extrapolation_spots), 3))
        new_points[:, 0] = extrapolation_spots[:, 0]
        new_points[:, 1] = extrapolation_spots[:, 1]
        for i in range(len(extrapolation_spots)):
            new_points[i, 2] = nearest_neighbor_interpolation(data,
                                    extrapolation_spots[i, 0],
                                    extrapolation_spots[i, 1])
        combined = np.concatenate((data, new_points))
        return combined


def interpolation(data):
    gridx, gridy = np.mgrid[0:10560:100j, 0:10600:100j]
    gridz = griddata(data[:, :2],data[:, 2], (gridx, gridy),
                method='nearest')
    return gridx, gridy, gridz


def nearest_neighbor_interpolation(data, x, y, p = 10):
    """
    Nearest Neighbor Weighted Interpolation
    http://paulbourke.net/miscellaneous/interpolation/
    http://en.wikipedia.org/wiki/Inverse_distance_weighting

    :param data: numpy.ndarray
        [[float, float, float], ...]
    :param p: float=0.5
        importance of distant samples
    :return: interpolated data
    """
    n = len(data)
    vals = np.zeros((n, 2), dtype = np.float64)
    distance = lambda x1, x2, y1, y2: (x2 - x1)**2 + (y2 - y1)**2
    for i in range(n):
        vals[i, 0] = data[i, 2] / (distance(data[i, 0], x, data[i, 1], y))**p
        vals[i, 1] = 1          / (distance(data[i, 0], x, data[i, 1], y))**p
    z = np.sum(vals[:, 0]) / np.sum(vals[:, 1])
    return z


def plot(data, gridx, gridy, gridz, method='rotate', title='nearest', both=False):
    def update(i):
        axis.view_init(azim = i)
        return axis,

    if method == 'rotate':
        fig = plt.figure()
        axis = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')

        axis.plot_wireframe(gridx, gridy, gridz, alpha=0.5)
        axis.scatter(data[:, 0], data[:, 1], data[:, 2], c='red')
        axis.set_title('Defocus CCD Surface %s' %title, fontsize = 18)
        axis.set_xlabel('x [mm]')
        axis.set_ylabel('y [mm]')
        axis.set_zlabel('[$\mu m$]')
        axis.set_xlim(0, 10560)
        axis.set_ylim(0, 10600)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps = 15, metadata=dict(artist= 'me'), bitrate = 1800)
        im_ani = animation.FuncAnimation(fig, update, np.arange(360 * 2), interval=1)
        im_ani.save('/media/data/bahtinov_results/Focusrun/' + today_utc_date + '/{}.mp4'.format(title), writer = writer)
        plt.close()

    elif method== 'snaps':
        fig = plt.figure(figsize=(10, 10))
        axis = fig.add_subplot(1, 1, 1, projection='3d')
        axis.plot_wireframe(gridx, gridy, gridz, alpha=0.5)
        axis.set_title('Defocus CCD Surface %s' %title, fontsize = 18)
        axis.set_xlabel('x [mm]')
        axis.set_ylabel('y [mm]')
        axis.set_zlabel('[$\mu m$]')
        axis.set_xlim(0, 10560)
        axis.set_ylim(0, 10600)
        axis.scatter(data[:, 0], data[:, 1], data[:, 2], c='red')
        plt.gca().invert_xaxis()
        plt.savefig('/media/data/bahtinov_results/Focusrun/' + today_utc_date + '/CCD_snap' + str(name) + '.png')
        plt.close()

    elif method == 'contour':
        fig = figure()
        axis = fig.add_subplot(111)
        CS = axis.contourf(gridx, gridy, gridz, cmap=cm.coolwarm, origin = 'lower')
        axis.scatter(data[:, 0], data[:, 1], s = 15, color = 'k')
        axis.set_xlim(0, 10560)
        axis.set_ylim(0, 10600)
        axis.set_title('Defocus CCD Surface %s' %name, fontsize = 22)
        axis.set_xlabel('x [pixel]')
        axis.set_ylabel('y [pixel]')
        cbar = plt.colorbar(CS, shrink = 0.8, label='[micron]')
        fig.tight_layout()
        fig.savefig('/media/data/bahtinov_results/Focusrun/' + today_utc_date + '/CCD_contour' + str(name) + '.png')
        plt.close()


def main(name):
    data = np.loadtxt(directory + name + '.txt')
    datax = data[:,3]
    datay = data[:,4]
    dataz = data[:,2]
    data = np.column_stack([datax, datay, dataz])
    extrapolation_spots = get_plane(0, 10560, 0, 10600, 30)
    nearest_analysis(extrapolation_spots, data, name)


# ====================================================================================
#                                   Start
# ====================================================================================
if __name__ == '__main__':
    directory_prefix = '/media/data/'
    directory = directory_prefix + 'bahtinov_results/Focusrun/' + today_utc_date + '/Results/'                 # Directory containing images
    file = '/media/data/bahtinov_results/' + today_utc_date + '/Results/FocusResults.txt'
    files = sorted(glob.glob(directory + 'M2_*.txt'))
    #if not os.path.exists(directory + 'Focusrun/' + today_utc_date + '/Results/CCD_surface/'):
    #    subprocess.call(('mkdir ' + directory + 'Focusrun/' + today_utc_date + '/Results/CCD_surface/').format(directory), shell=True)
    #subprocess.call(('rm ' + directory + 'Focusrun/' + today_utc_date + '/Results/CCD_surface/' + '/*.png').format(directory_prefix), shell=True)

    # ====================================================================================


    for k in xrange(0,len(files)):
        name = files[k].split('/')[-1].split('.')[0]
        main(name)



    period = time.time() - start
    print '\nThe computation time was %.3f seconds\n' %(period)
    print '=============== CCD Fit Ended ==============='
