#!/usr/bin/env python

# ====================================================================================
#                           Bahtinov Mask Analysis Software
# ====================================================================================

# Import required modules
from __future__ import division
from pathos.multiprocessing import ProcessingPool
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
import os
import subprocess
from pylab import rcParams
import sys
import cv2
import numpy as np
from pylab import *
from math import *
import sep
import math
import scipy
from scipy import signal
from astropy.io import fits
from astropy.convolution import convolve
import time
from scipy import ndimage
from scipy import stats
from astropy.stats import sigma_clip
from astropy.nddata import Cutout2D
from skimage.feature import canny
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.pyplot import figure, show, rc, cm
from mpl_toolkits.mplot3d import Axes3D
from astropy.stats import sigma_clipped_stats
rcParams['figure.figsize'] = 14, 10

today_utc_date = datetime.datetime.utcnow().strftime('%Y-%m-%d')
today_utc_time = time.strftime('%c', time.gmtime(time.time()))


start = time.time()
Pixel_scale = []
class Bahtinov_model:
    def __init__(model):
        model.lambnm = list(np.arange(350,1000, 10))                            # wavelength (nm)
        model.width = 7.9 * 10**(-3)                                            # slit width (m)
        model.spacing = 2 * model.width                                         # slit spacing (m)
        model.spacingnm = model.spacing*1e9
        model.quadrants = [1, 23, 4]                                            # mask quadrants
        model.defocus = list(np.arange(-200, 210, 10))                          # axial defocus (um)
        model.filters_name = ['u', 'g', 'q', 'r', 'i', 'z']
        model.filters_central = [380, 480, 580, 626.5, 765, 900]
        #model.filters_central = [350, 410, 440, 562, 690, 840]
        #model.Pixel_scale = []
        model.workdir = '/media/maik/Aurelia/MeerLICHT/Model/Factor1' # os.getcwd()
        model.nproc = 3
        model.pool = ProcessingPool()
        #model.mask = cv2.imread(os.getcwd() + '/Mask/bahtinov_mask1_crop.png', 0)
        model.mask = cv2.imread('/media/maik/Veritas/MeerLICHT/Data/bahtinov_mask1_crop.png', 0)
        model.dot = cv2.imread('/media/maik/Veritas/MeerLICHT/Data/dot.png', 0)
        model.mask_data = np.array(model.mask)/255
        model.dot_data = np.array(model.dot)/255
        if not os.path.exists(model.workdir + '/Model'):
            subprocess.call(('mkdir ' + model.workdir + '/Model').format(model.workdir), shell=True)
        if not os.path.exists(model.workdir + '/Model/Data'):
            subprocess.call(('mkdir ' + model.workdir + '/Model/Data' ).format(model.workdir), shell=True)
        if not os.path.exists(model.workdir + '/Model/Data/Convolved'):
            subprocess.call(('mkdir ' + model.workdir + '/Model/Data/Convolved' ).format(model.workdir), shell=True)
        if not os.path.exists(model.workdir + '/Model/Data/Filters'):
            subprocess.call(('mkdir ' + model.workdir + '/Model/Data/Filters' ).format(model.workdir), shell=True)
        if not os.path.exists(model.workdir + '/Model/Data/Wavelengths'):
            subprocess.call(('mkdir ' + model.workdir + '/Model/Data/Wavelengths' ).format(model.workdir), shell=True)
        if not os.path.exists(model.workdir + '/Model/Data/Pattern'):
            subprocess.call(('mkdir ' + model.workdir + '/Model/Data/Pattern' ).format(model.workdir), shell=True)
        if not os.path.exists(model.workdir + '/Model/Images'):
            subprocess.call(('mkdir ' + model.workdir + '/Model/Images' ).format(model.workdir), shell=True)
        if not os.path.exists(model.workdir + '/Model/Images/Convolved'):
            subprocess.call(('mkdir ' + model.workdir + '/Model/Images/Convolved' ).format(model.workdir), shell=True)
        if not os.path.exists(model.workdir + '/Model/Images/Filters'):
            subprocess.call(('mkdir ' + model.workdir + '/Model/Images/Filters' ).format(model.workdir), shell=True)
        if not os.path.exists(model.workdir + '/Model/Images/Wavelengths'):
            subprocess.call(('mkdir ' + model.workdir + '/Model/Images/Wavelengths' ).format(model.workdir), shell=True)
        if not os.path.exists(model.workdir + '/Model/Images/Pattern'):
            subprocess.call(('mkdir ' + model.workdir + '/Model/Images/Pattern' ).format(model.workdir), shell=True)
        print '='*30 + ' Script Information ' + '=' * 30
        print 'Script started on (UTC):', today_utc_time
        start = time.time()
        print 'Current working directory is:', model.workdir
        print 'Running on %g threads.' % int(np.ceil(model.nproc))
        print '='*80



    def binArray(model, data, axis, binstep, binsize, func=np.nanmean):
        data = np.array(data)
        dims = np.array(data.shape)
        argdims = np.arange(data.ndim)
        argdims[0], argdims[axis]= argdims[axis], argdims[0]
        data = data.transpose(argdims)
        data = [func(np.take(data, np.arange(int(i*binstep), int(i*binstep+binsize)), 0), 0) for i in np.arange(dims[axis]//binstep)]
        data = np.array(data).transpose(argdims)
        return data

    def pad_data(model, data, bp_factor, factor = 8):
        sizex = data.shape[0] * factor
        sizey = sizex
        data_resized = cv2.resize(data, (int(np.array(data).shape[0]*bp_factor),int(np.array(data).shape[1]*bp_factor)))
        shapex = int(((sizex - data_resized.shape[0])/2))
        shapey = int(((sizey - data_resized.shape[1])/2))
        data_pad = np.lib.pad(data_resized, ((shapex, shapex),(shapey, shapey)), 'constant', constant_values = 0)
        if data_pad.shape[0] != sizex and data_pad.shape[1] != sizey:
            diffx = abs(data_pad.shape[0] - sizex)
            diffy = abs(data_pad.shape[1] - sizey)
            data_pad = np.lib.pad(data_pad, ((0, 0),(diffx, 0)), 'constant', constant_values = 0)
            data_pad = np.lib.pad(data_pad, ((diffy, 0),(0, 0)), 'constant', constant_values = 0)
        else:
            pass

        return data_pad

    def Fourier_transform(model, data):
        fft = np.fft.fft2(data)
        shift = np.fft.fftshift(fft)
        powerspectrum = abs(shift)**2
        return powerspectrum

    def create_mask(model, l):
        lamb = l
        lamb0 = 340    # wavelength (m)  340 nm

        bandpass_factor = 1 / (lamb/lamb0)
        mask_data = model.pad_data(model.mask, bandpass_factor)
        size = int(len(model.mask_data)/2)
        return mask_data, size

    def create_filters(model, filter, l):
        mask_data, size = model.create_mask(l)

        filter1_size_x = mask_data[:size,size:].shape[0]
        filter1_size_y = mask_data[:size,size:].shape[1]
        filter23_size_x = mask_data[:,:size].shape[0]
        filter23_size_y = mask_data[:,:size].shape[1]
        filter4_size_x = mask_data[size:,size:].shape[0]
        filter4_size_y = mask_data[size:,size:].shape[1]

        quadrant1 = np.zeros((mask_data.shape))
        quadrant1[:size,size:] = mask_data[:size,size:]
        quadrant23 = np.zeros((mask_data.shape))
        quadrant23[:,:size] = mask_data[:,:size]
        quadrant4 = np.zeros((mask_data.shape))
        quadrant4[size:,size:] = mask_data[size:,size:]

        #fig = figure()
        #ax = fig.add_subplot(131)
        #ax.imshow(quadrant1, cmap = 'Greys', interpolation = 'none')
        #ax = fig.add_subplot(132)
        #ax.imshow(quadrant23, cmap = 'Greys', interpolation = 'none')
        #ax = fig.add_subplot(133)
        #ax.imshow(quadrant4, cmap = 'Greys', interpolation = 'none')
        #show()
        q_1 = Fourier_transform(quadrant1)
        q_23 = Fourier_transform(quadrant23)
        q_4 = Fourier_transform(quadrant4)

        q_1_sliced = Cutout2D(q_1, (q_1.shape[0]/2, q_1.shape[1]/2), (1000, 1000)).data
        q_23_sliced = Cutout2D(q_23, (q_23.shape[1]/2, q_23.shape[0]/2), (1000, 1000)).data
        q_4_sliced = Cutout2D(q_4, (q_1.shape[0]/2, q_4.shape[1]/2), (1000, 1000)).data

        np.savetxt(model.workdir + '/Model/Data/Wavelengths/Filter_%s_quadrant1_%s.txt' %(filter, l), q_1_sliced)
        np.savetxt(model.workdir + '/Model/Data/Wavelengths/Filter_%s_quadrant23_%s.txt' %(filter, l), q_23_sliced)
        np.savetxt(model.workdir + '/Model/Data/Wavelengths/Filter_%s_quadrant4_%s.txt' %(filter, l), q_4_sliced)


    def create_spectrum_values(model, l):
        print 'Current wavelength [nm]:', l
        if 350 <= l <= 410:
            model.create_filters('u', l)

        if 410 <= l <= 550:
            model.create_filters('g', l)

        if 440 <= l <= 720:
            model.create_filters('q', l)

        if 562 <= l <= 690:
            model.create_filters('r', l)

        if 690 <= l <= 840:
            model.create_filters('i', l)

        if 840 <= l <= 990:
            model.create_filters('z', l)


    def create_spectrum_bandpass(model, Q):
        u, g, q, r, i, z = [], [], [], [], [], []
        filters = [u, g, q, r, i, z]
        path = model.workdir + '/Model/Data/Wavelengths/'
        for file in os.listdir(path):
            for n, f in enumerate(model.filters_name):
                if os.path.isfile(os.path.join(path,file)) and file.startswith('Filter_%s_quadrant%s' %(f, Q)):
                    bandpass = np.loadtxt(model.workdir + '/Model/Data/Wavelengths/' + file)
                    filters[n].append(bandpass)
        for n, f in enumerate(filters):
            filter = np.array(f).sum(axis=0)

            sources = sep.extract(filter, np.max(filter)/1e2,  gain = 1.0, minarea = 30)
            x,y = sources['x'], sources['y']
            orders = zip(x,y)
            x,y = zip(*orders)
            for x,y in orders:
                if abs(x - filter.shape[0]/2) < 20 and abs(y - filter.shape[1]/2) < 20:
                    x_cen = x
                    y_cen = y
                else:
                    x_1st = x
                    y_1st = y
            d = (abs(x_cen - x_1st)**2 + abs(y_cen - y_1st)**2)**.5
            pixel_scale = model.filters_central[n] / model.spacingnm  * 206265  / d
            #print model.filters_name[n], Q
            #print x_cen, y_cen, x_1st, y_1st, d, pixel_scale
            Pixel_scale.append(pixel_scale)
            #print model.Pixel_scale

            x,y = zip(*orders)
            np.savetxt(model.workdir + '/Model/Data/Filters/Filter_%s_quadrant%s.txt' %(model.filters_name[n], Q), filter)
            #np.savetxt(workdir + '/Model/Data/Filters/Orders_Filter_%s_quadrant%s.txt' %(filters_name[n], Q), orders)
            fig = figure()
            ax = fig.add_subplot(111)
            ax.imshow(filter, cmap = 'Greys', interpolation = 'none')
            ax.scatter(sources['x'], sources['y'])
            #ax.grid(which='major', linestyle='-')
            #ax.grid(which='minor', linestyle='--')
            ax.grid()
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            #ax.set_xlim(350, 650)
            #ax.set_ylim(200, 800)
            fig.savefig(model.workdir + '/Model/Images/Filters/Filter_%s_quadrant%s.png' %(model.filters_name[n], Q))
            #show()
    	plt.close()
        return Pixel_scale


    def create_bandpass_defocus(model, delta, Pixel_scale):
        print 'Current defocus value is %g nm' %delta

        mirror = 6.50e5                          # diameter mirror micron
        if delta < 0:
            scaling_factor = abs(delta / 3.3e6)
            resize_mask = int(scaling_factor * mirror)
        if delta > 0:
            scaling_factor = delta / 3.3e6
            resize_mask = int(scaling_factor * mirror)
        if delta == 0 :
            resize_mask = 1

        #print Pixel_scale, np.mean(Pixel_scale)
        pixel_rescale_parameter = 0.56/np.mean(Pixel_scale) / 9

        size = int(len(model.dot_data)/2)
        quadrant1 = model.dot_data[size:,size:]
        quadrant23 = model.dot_data[:,:size]
        quadrant4 = model.dot_data[:size,size:]

        mask_resized = cv2.resize(model.dot_data, (int(np.ceil(resize_mask * pixel_rescale_parameter)), int(np.ceil(resize_mask * pixel_rescale_parameter))))
        #mask_resized_Q23 = cv2.resize(quadrant23, (int(scaling_factor * mirror), int(scaling_factor * mirror)))
        #mask_resized_Q4 = cv2.resize(quadrant4, (int(scaling_factor * mirror), int(scaling_factor * mirror)))
        size = int(len(mask_resized)/2)

        quadrant1 = np.zeros((mask_resized.shape))
        quadrant1[:size,size:] = mask_resized[:size,size:]
        quadrant23 = np.zeros((mask_resized.shape))
        quadrant23[:,:size] = mask_resized[:,:size]
        quadrant4 = np.zeros((mask_resized.shape))
        quadrant4[size:,size:] = mask_resized[size:,size:]

        if delta < 0:
            quadrant1 = np.fliplr(quadrant1)
            quadrant23 = np.fliplr(quadrant23)
            quadrant4 = np.fliplr(quadrant4)
        else:
            pass
        fig = figure()
        ax = fig.add_subplot(131)
        ax.imshow(quadrant1, cmap = 'Greys', interpolation = 'none')
        ax = fig.add_subplot(132)
        ax.imshow(quadrant23, cmap = 'Greys', interpolation = 'none')
        ax = fig.add_subplot(133)
        ax.imshow(quadrant4, cmap = 'Greys', interpolation = 'none')
        show()


        mask_quadrants = [quadrant1, quadrant23, quadrant4]
        for Q in xrange(len(model.quadrants)):
            for f in model.filters_name:
                bandpass = np.loadtxt(model.workdir + '/Model/Data/Filters/Filter_%s_quadrant%s.txt' %(f, model.quadrants[Q]))
                defocus_data = ndimage.convolve(bandpass, mask_quadrants[Q])
                #defocus_data = np.log(defocus_data)
                np.savetxt(model.workdir + '/Model/Data/Convolved/Filter_%s_quadrant%s_axial%snm.txt' %(f, model.quadrants[Q], delta), defocus_data)
                fig = figure()
                ax = fig.add_subplot(111)
                ax.set_title('Filter: %s' %(f))
                ax.imshow(defocus_data, cmap = 'Greys', interpolation = 'none')
                xlim = defocus_data.shape[1] / 2
                ylim = defocus_data.shape[0] / 2
                ax.scatter(xlim, ylim, s = 1)
                #ax.set_xlim(xlim - 230, xlim + 230)
                #ax.set_ylim(ylim - 1000, ylim + 1000)
                ax.invert_yaxis()
                ax.set_xlabel('Pixel')
                ax.set_ylabel('Pixel')
                ax.grid()
                ax.annotate('Axial defocus = %.f$\\mu m$' %(delta),
                    xy=(0, 1), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='top')
                fig.savefig(model.workdir + '/Model/Images/Convolved/Convolved_Filter_%s_quadrant%s_axial%snm.png' %(f, model.quadrants[Q], delta))
                #show()
                plt.close()

    def combine_quadrants(model, delta):
        print 'Current axial defocus value is %g nm' %delta
        for f in model.filters_name:
            quadrant1 = np.loadtxt( model.workdir + '/Model/Data/Convolved/Filter_%s_quadrant1_axial%snm.txt' %(f, delta))
            quadrant23 = np.loadtxt( model.workdir + '/Model/Data/Convolved/Filter_%s_quadrant23_axial%snm.txt' %(f, delta))
            quadrant4 = np.loadtxt( model.workdir + '/Model/Data/Convolved/Filter_%s_quadrant4_axial%snm.txt' %(f, delta))
            diffraction_pattern = quadrant1 + quadrant23 + quadrant4
            #diffraction_pattern = np.log(diffraction_pattern)
            np.savetxt( model.workdir + '/Model/Data/Pattern/Diffraction_pattern_filter_%s_axial%sUm' %(f, delta), diffraction_pattern )
            fig = figure()
            ax = fig.add_subplot(111)
            ax.set_title('Filter: %s' %(f))
            cax = ax.imshow(diffraction_pattern, cmap = 'Greys', interpolation = 'none')
            cbar = fig.colorbar(cax, ticks=[np.min(diffraction_pattern), np.max(diffraction_pattern)/2, np.max(diffraction_pattern)])
            #cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar
            ax.invert_yaxis()
            ax.set_xlabel('Pixel')
            ax.set_ylabel('Pixel')
            ax.grid()
            #ax.set_xlim(150,850)
            #ax.set_ylim(150,850)
            ax.annotate('Axial defocus = %.f$\\mu m$' %(delta),
                xy=(0, 1), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='top')
            fig.savefig( model.workdir + '/Model/Images/Pattern/Diffraction_pattern_Filter_%s_axial%snm.png' %(f, delta))
            #show()
            plt.close()

    def main(model):
        #model.map(model.create_spectrum_values, model.lambnm)
        for Q in model.quadrants:
            Pixel_scale = model.create_spectrum_bandpass(Q)
        print Pixel_scale
        for d in model.defocus:
            model.create_bandpass_defocus(d, Pixel_scale)
        for d in model.defocus:
            model.combine_quadrants(d)


if __name__ == '__main__':
    Model = Bahtinov_model()
    #Pixel_scale = pool.map(Model.create_spectrum_bandpass, Model.quadrants)
    #pool.map(Model.create_bandpass_defocus, Model.defocus, Pixel_scale)
    Model.main()
    #pool.map(create_spectrum_values, lambnm)
    #pool.map(create_spectrum_bandpass, quadrants)
    #pool.map(create_bandpass_defocus, defocus)
    #pool.map(combine_quadrants, defocus)


'''
Class Bahtinov
    Has the end goal to calculate the axial defocus on the CCD.
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
        #image.flat = fits.open('/media/data/Bahtinov/2017_03_15/Bias/Masterflat.fits')
        #image.flat = image.flat[0].data
        image.workdir  = workdir
        image.image_path  = image_path                                  # full image path
        image.title = image.image_path.split('/')[-1]                   # name of the image with extension
        image.name = name                                               # name of the image without extension
        image.number = float(image.title.split('_')[-2])                # image number
        image.image = fits.open(image.image_path)                       # opening fits image
        image.data = image.image[1].data                                # image data
        image.data = np.asarray(image.data - image.bias, dtype = np.float)
        image.X = X ; image.Y = Y                                       # x and y coordinates of star
        image.SNR = SNR                                                 # Calcalated SNR
        image.Xerr = Xerr ; image.Yerr = Yerr                           # xerr and yerr of star obtained from sep
        image.angle = math.radians(21)                                  # angle of the diagnoal gratings of the Bahtinov mask
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
        image.data_new = image.rotate_image(image.data_new, 45, size_i)
        image.mean_new, image.median_new, image.std_new = sigma_clipped_stats(image.data_new, sigma=3, iters=5)

        image.data_new = np.asarray(image.data_new, dtype = np.float)
        image.data_new = image.data_new.copy(order = 'C')
        background = sep.Background(image.data_new)
        threshold = background.globalrms * 5
        image.data_new = image.data_new - background
        source = sep.extract(image.data_new, threshold)
        if len(source) != 0:
            image.x, image.y = source['x'][np.where(source['flux'] == np.max(source['flux']))], source['y'][np.where(source['flux'] == np.max(source['flux']))]
        else:
            image.x, image.y = 0, 0

    def rotate_image(image, data, angle, size):
        M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1.0)
        data = cv2.warpAffine(data, M, (size, size))
        data = Cutout2D(data, (size/2, size/2), (size/2+80, size/2+80)).data
        return data

    def calculate_focus_error(image, a, sigma_a, b, sigma_b, c, sigma_c, d, sigma_d, sigma_center):
        A_y = ( (b*(c-d)) / (a-b)**2 )**2 * sigma_a**2
        B_y = ( (a*(d-c)) / (a-b)**2 )**2 * sigma_b**2
        C_y = (- b / (a-b))**2 * sigma_c**2
        D_y = ( a / (a-b))**2 * sigma_d**2
        sigma2_y = A_y + B_y + C_y + D_y

        A_x = ( (c-d)/(a-b)**2 )**2 * sigma_a**2
        B_x = ( (d-c)/(a-b)**2 )**2 * sigma_b**2
        C_x = ( 1 / (b-a))**2 * sigma_c**2
        D_x = ( 1 / (a-b))**2 * sigma_d**2
        sigma2_x = A_x + B_x + C_x + D_x

        sigma2_focus = (sigma2_y + sigma_center**2)
        sigma2_focus = ((9/2) * (33000/2590))**2 * sigma2_focus
        return sigma2_x, sigma2_y, sigma2_focus**.5

    def calculate_focus(image, outerline0, centralline, outerline1):
        line1 = LineString([(outerline0[0][0], outerline0[0][1]), (outerline0[-1][0], outerline0[-1][1])])
        line2 = LineString([(outerline1[0][0], outerline1[0][1]), (outerline1[-1][0], outerline1[-1][1])])
        # Calculate intersection
        diagonal_line_intersection = line1.intersection(line2)
        # Only if intersection location is close to center image
        if abs(np.array(diagonal_line_intersection)[0] - image.x) < 10:
            line_center = LineString([(centralline[0][0], centralline[0][1]), (centralline[-1][0], centralline[-1][1])])
            if np.array(diagonal_line_intersection)[1] > image.intercept1:
                focus = -diagonal_line_intersection.distance(line_center) / 2 * 9 * (33000 / image.delta)
            else:
                focus = diagonal_line_intersection.distance(line_center) / 2 * 9 * (33000 / image.delta)
        else:
            focus = None
        return focus, diagonal_line_intersection

    def main(image):
        #if image.Focus != None:
        pos1 = len(magnitude_spectrum_quadrant1_binned_norm_rotated)
        pos23 = len(magnitude_spectrum_quadrant23_binned_norm_rotated)
        pos4 = len(magnitude_spectrum_quadrant4_binned_norm_rotated)
        size = len(image.data_new)
        datacut1 =  Cutout2D(magnitude_spectrum_quadrant1_binned_norm_rotated*np.max(image.data_new), (pos1/2, pos1/2), (size, size)).data
        datacut23 =  Cutout2D(magnitude_spectrum_quadrant23_binned_norm_rotated*np.max(image.data_new), (pos23, pos23/2), (size, size)).data
        datacut4 =  Cutout2D(magnitude_spectrum_quadrant4_binned_norm_rotated*np.max(image.data_new), (pos4/2, pos4/2), (size, size)).data
        datacut = datacut1 + datacut23 + datacut4
        datacut = datacut/3
        fig = figure()
        ax = fig.add_subplot(121)
        cax = ax.imshow(image.data_new, norm = matplotlib.colors.Normalize(vmin=0, vmax = np.max(image.data_new)))
        scale_min = np.min(image.data_new)
        scale_half = np.max(image.data_new) / 2
        scale_max = np.max(image.data_new)
        cbar = fig.colorbar(cax, ticks = [scale_min, scale_half, scale_max], shrink = 0.8)
        ax = fig.add_subplot(122)
        cax = ax.imshow(datacut, norm = matplotlib.colors.Normalize(vmin=0, vmax = np.max(datacut)))
        scale_min = np.min(datacut)
        scale_half = np.max(datacut) / 2
        scale_max = np.max(datacut)
        cbar = fig.colorbar(cax, ticks = [scale_min, scale_half, scale_max], shrink = 0.8)
        show()
        '''
        if image.Focus != None:
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

        image.fig, image.axis = plt.subplots(figsize = (10,10))
        image.axis.errorbar(point.x,point.y, yerr=sigma2_y**.5, xerr=sigma2_x**.5, color = 'r')
        image.axis.imshow(image.data_new, cmap=cm.gray, norm = matplotlib.colors.LogNorm(vmin = 0.01, vmax = np.max(image.data_new)), origin = 'lower')
        image.axis.scatter(image.x, image.y, color = 'r')
        image.axis.set_xlim(0,len(image.data_new)) ; image.axis.set_ylim(0,len(image.data_new))
        image.axis.set_xlabel('x') ; image.axis.set_ylabel('y')
        image.axis.set_title('Bahtinov Source: %s (%.2f, %.2f)' %(image.name, image.X, image.Y))
        image.axis.plot(zip(*image.XY)[0], zip(*image.XY)[1], color = 'b')
        image.axis.plot(zip(*image.XY1)[0], zip(*image.XY1)[1], color = 'g')
        image.axis.plot(zip(*image.XY2)[0], zip(*image.XY2)[1], color = 'b')
        image.axis.annotate('Axial distance = %.2f $\pm$ %.3f $\\mu m$' %(image.Focus, image.focuserr), xy=(1, -.06), xycoords='axes fraction', fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        image.fig.savefig(/media/data/bahtinov_results/Model/ +'Focusrun/' + today_utc_date + '/Plots/' + str(image.name) + '/' + str(image.name) + '_' + str(image.X) + '_' + str(image.Y) + '.png')
        plt.close()
        '''

today_utc_time = time.strftime('%c', time.gmtime(time.time()))
elapsed_sec = time.time() - start
elapsed_min = elapsed_sec // 60
elapsed_sec_remain = elapsed_sec - elapsed_min
print '+\n' + 'Script ended on', today_utc_time
print 'Total computation time = %g minutes, %.3g seconds.' %(elapsed_min, elapsed_sec_remain)
