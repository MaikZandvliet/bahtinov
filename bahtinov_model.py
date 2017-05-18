#!/usr/bin/env python

# ====================================================================================
#                           Bahtinov Mask Analysis Software
# ====================================================================================

# Import required modules
from __future__ import division
import multiprocessing
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.tools import make_default_context
from pyfft.cuda import Plan
from joblib import Parallel, delayed
import os
from pylab import rcParams
from tqdm import *
import sys
import cv2
import numpy as np
from pylab import *
from math import *
import scipy
from scipy import signal
from astropy.io import fits
from astropy.convolution import convolve
import time
from scipy import stats
from astropy.stats import sigma_clip
from astropy.nddata import Cutout2D
from mayavi import mlab
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.pyplot import figure, show, rc, cm
from mpl_toolkits.mplot3d import Axes3D
from astropy.stats import sigma_clipped_stats
rcParams['figure.figsize'] = 14, 10

today_utc_date = datetime.datetime.utcnow().strftime('%Y-%m-%d')
today_utc_time = time.strftime('%c', time.gmtime(time.time()))

def main():
    # generates simple matrix, (e.g. image with a signal at the center)
    size = 4096
    center = size/2
    in_matrix = np.zeros((size, size), dtype='complex64')
    in_matrix[center:center+2, center:center+2] = 10.

    pool_size = 4  # integer up to multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=pool_size)
    func = FuncWrapper(in_matrix, size)
    nffts = 16  # total number of ffts to be computed
    par = np.arange(nffts)

    results = pool.map(func, par)
    pool.close()
    pool.join()

    print results

class FuncWrapper(object):
    def __init__(self, matrix, size):
        self.in_matrix = matrix
        self.size = size
        print("Func initialized with matrix size=%i" % size)

    def __call__(self, par):
        proc_id = multiprocessing.current_process().name

        # take control over the GPU
        cuda.init()
        context = make_default_context()
        device = context.get_device()
        proc_stream = cuda.Stream()

        # move data to GPU
        # multiplication self.in_matrix*par is just to have each process computing
        # different matrices
        in_map_gpu = gpuarray.to_gpu(self.in_matrix*par)

        # create Plan, execute FFT and get back the result from GPU
        plan = Plan((self.size, self.size), dtype=np.complex64,
                    fast_math=False, normalize=False, wait_for_finish=True,
                    stream=proc_stream)
        plan.execute(in_map_gpu, wait_for_finish=True)
        result = in_map_gpu.get()

        # free memory on GPU
        del in_map_gpu

        mem = np.array(cuda.mem_get_info())/1.e6
        print("%s free=%f\ttot=%f" % (proc_id, mem[0], mem[1]))

        # release context
        context.pop()

main()

def binArray(data, axis, binstep, binsize, func=np.nanmean):
    data = np.array(data)
    dims = np.array(data.shape)
    argdims = np.arange(data.ndim)
    argdims[0], argdims[axis]= argdims[axis], argdims[0]
    data = data.transpose(argdims)
    data = [func(np.take(data, np.arange(int(i*binstep), int(i*binstep+binsize)), 0), 0) for i in np.arange(dims[axis]//binstep)]
    data = np.array(data).transpose(argdims)
    return data

def pad_data(data, bp_factor, scaling_factor, factor = 2):
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
    #data_pad_resized = cv2.resize(data_pad, (int(np.array(data_pad).shape[0]*scaling_factor), int(np.array(data_pad).shape[1]*scaling_factor)))
    #print 'final shape', data_pad_resized.shape
    return data_pad

def Fourier_transform(data):
    fft = np.fft.fft2(data)
    shift = np.fft.fftshift(fft)
    powerspectrum = abs(shift)**2
    #powerspectrum = (powerspectrum - np.min(powerspectrum)) / (np.max(powerspectrum - np.min(powerspectrum)))
    return powerspectrum


def create_spectrum_values(l):
    print 'Current wavelength [nm]:', l
    lamb = l * 10e-10
    lamb0 = 0.55*10**(-6)    # wavelength (m)  550 nm

    bandpass_factor = 1 / (lamb/spacing * spacing/lamb0)
    scaling_factor = 100 / 3.3e6

    mask = cv2.imread('/media/data/Bahtinov/bahtinov_mask4_crop.png', 0)
    #image = fits.open('/media/data/Bahtinov/2017_03_15/temp_12000x10600_105_cut.fits')
    mask_data = np.array(mask)/255
    mask_data = pad_data(mask, bandpass_factor, scaling_factor)
    size = int(len(mask_data)/2)

    quadrant1 = mask_data[size:,size:]
    quadrant23 = mask_data[:,:size]
    quadrant4 = mask_data[:size,size:]

    # obtain magnitude spectrum per quadrant linear
    magnitude_spectrum_quadrant1 = Fourier_transform(quadrant1)
    magnitude_spectrum_quadrant23 = Fourier_transform(quadrant23)
    magnitude_spectrum_quadrant4 =  Fourier_transform(quadrant4)

    '''
    # rotate the image 90 deg
    magnitude_spectrum_quadrant1_binned_norm_rotated = np.rot90(magnitude_spectrum_quadrant1_binned_norm)
    magnitude_spectrum_quadrant23_binned_norm_rotated = np.rot90(magnitude_spectrum_quadrant23_binned_norm)
    magnitude_spectrum_quadrant4_binned_norm_rotated = np.rot90(magnitude_spectrum_quadrant4_binned_norm)
    '''

    np.savetxt('/media/data/bahtinov_results/Model/Data/Wavelengths/Quadrant1_%s.txt' %(l), magnitude_spectrum_quadrant1)
    np.savetxt('/media/data/bahtinov_results/Model/Data/Wavelengths/Quadrant23_%s.txt' %(l), magnitude_spectrum_quadrant23)
    np.savetxt('/media/data/bahtinov_results/Model/Data/Wavelengths/Quadrant4_%s.txt' %(l), magnitude_spectrum_quadrant4)
    '''
    fig = figure()
    ax = fig.add_subplot(221)
    ax.set_title("Bahtinov mask input image")
    ax.imshow(mask_data, cmap = 'gray', interpolation = 'none')
    ax = fig.add_subplot(222)
    ax.set_title("FFT Quadrant 1")
    cax = ax.imshow(magnitude_spectrum_quadrant1, cmap = 'Blues', interpolation = 'none')
    scale_min = np.min(magnitude_spectrum_quadrant1)
    scale_half = np.max(magnitude_spectrum_quadrant1)/2
    scale_max = np.max(magnitude_spectrum_quadrant1)
    cbar = fig.colorbar(cax, ticks = [scale_min, scale_half, scale_max])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    xlim = magnitude_spectrum_quadrant1.shape[1]/2
    ylim = magnitude_spectrum_quadrant1.shape[0]/2
    ax.set_xlim(xlim - 50, xlim + 50)
    ax.set_ylim(ylim - 125, ylim + 125)

    ax = fig.add_subplot(223)
    ax.set_title("FFT Quadrant 2&3")
    cax = ax.imshow(magnitude_spectrum_quadrant23, cmap = 'Blues', interpolation = 'none')
    scale_min = np.min(magnitude_spectrum_quadrant23)
    scale_half = np.max(magnitude_spectrum_quadrant23)/2
    scale_max = np.max(magnitude_spectrum_quadrant23)
    cbar = fig.colorbar(cax, ticks = [scale_min, scale_half, scale_max])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    xlim = magnitude_spectrum_quadrant23.shape[1]/2
    ylim = magnitude_spectrum_quadrant23.shape[0]/2
    ax.set_xlim(xlim - 50, xlim + 50)
    ax.set_ylim(ylim - 210, ylim + 210)

    ax = fig.add_subplot(224)
    ax.set_title("FFT Quadrant 4")
    cax = ax.imshow(magnitude_spectrum_quadrant4, cmap = 'Blues', interpolation = 'none')
    scale_min = np.min(magnitude_spectrum_quadrant4)
    scale_half = np.max(magnitude_spectrum_quadrant4) / 2
    scale_max = np.max(magnitude_spectrum_quadrant4)
    bar = fig.colorbar(cax, ticks = [scale_min, scale_half, scale_max])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    xlim = magnitude_spectrum_quadrant4.shape[1]/2
    ylim = magnitude_spectrum_quadrant4.shape[0]/2
    ax.set_xlim(xlim - 50, xlim + 50)
    ax.set_ylim(ylim - 125, ylim + 125)

    plt.suptitle('Wavelength = %s [nm]' %l, fontsize = 14)
    plt.tight_layout()
    fig.subplots_adjust(wspace = .1, hspace = 0.12 )
    fig.savefig('/media/data/bahtinov_results/Model/Images/Wavelengths/FFT_bahtinovmask_%snm.png' %(l))
    #show()
    plt.close()
    '''

width = 7.9 * 10**(-3)      # slit width (m)
spacing = 2 * width

lambnm = list(np.arange(350,1000, 10)) # wavelength (nm)


quadrants = [1, 4, 23]
def create_spectrum_bandpass(Q):

    shapex = np.loadtxt('/media/data/bahtinov_results/Model/Data/Wavelengths/Quadrant%s_350.txt' %(Q)).shape[0]
    shapey = np.loadtxt('/media/data/bahtinov_results/Model/Data/Wavelengths/Quadrant%s_350.txt' %(Q)).shape[1]
    u = np.zeros((shapex,shapey)) ; g = np.zeros((shapex,shapey)) ; q = np.zeros((shapex,shapey))
    r = np.zeros((shapex,shapey)) ; i = np.zeros((shapex,shapey)) ; z = np.zeros((shapex,shapey))
    for l in tqdm(lambnm):
        wave = np.loadtxt('/media/data/bahtinov_results/Model/Data/Wavelengths/Quadrant%s_%s.txt' %(Q, l))
        if 350 <= l <= 410:
            u += wave
        if 410 <= l <= 550:
            g += wave
        if 440 <= l <= 720:
            q += wave
        if 562 <= l <= 690:
            r += wave
        if 690 <= l <= 840:
            i += wave
        if 840 <= l <= 990:
            z += wave

    filters = [u, g, q, r, i, z]
    filters_name = ['u', 'g', 'q', 'r', 'i', 'z']
    if Q == 23:
        #fig, axes = plt.subplots(2,3, sharex=True, sharey=True)
        #ax = axes.ravel()
        #plt.title('Filters')
        for m in range(0,6):
            np.savetxt('/media/data/bahtinov_results/Model/Data/Filters/Filter_%s_quadrant%s.txt' %(filters_name[m], Q), (filters[m]))
            #ax[m].set_title('Filter: %s'%(filters_name[m]))
            #ax[m].imshow(filters[m], cmap = 'Blues', interpolation = 'none')
            #ax[m].xaxis.set_minor_locator(AutoMinorLocator())
            #ax[m].yaxis.set_minor_locator(AutoMinorLocator())
            #xlim = filters[m].shape[1]/2
            #ylim = filters[m].shape[0]/2
            #ax[m].set_xlim(xlim - 30, xlim + 30)
            #ax[m].set_ylim(ylim - 200, ylim + 200)
        #show()
        #fig.savefig('/media/data/bahtinov_results/Model/Images/FFT_filters_quadrant%s.png' %(Q))
        #plt.close()
    else:
        #fig, axes = plt.subplots(2,3, sharex=True, sharey=True)
        #ax = axes.ravel()
        #plt.title('Filters')
        for m in range(0,6):
            np.savetxt('/media/data/bahtinov_results/Model/Data/Filters/Filter_%s_quadrant%s.txt' %(filters_name[m], Q), (filters[m]))
            #ax[m].set_title('Filter: %s'%(filters_name[m]))
            #ax[m].imshow(filters[m], cmap = 'Blues', interpolation = 'none')
            #ax[m].xaxis.set_minor_locator(AutoMinorLocator())
            #ax[m].yaxis.set_minor_locator(AutoMinorLocator())
            #xlim = filters[m].shape[1] / 2
            #ylim = filters[m].shape[0] / 2
            #ax[m].set_xlim(xlim - 30, xlim + 30)
            #ax[m].set_ylim(ylim - 110, ylim + 110)
        #show()
        #fig.savefig('/media/data/bahtinov_results/Model/Images/FFT_filters_quadrant%s.png' %(Q))
        #plt.close()

def slice_bandpass():
    quadrants = [1, 4, 23]
    filters_name = ['u', 'g', 'q', 'r', 'i', 'z']
    for Q in quadrants:
        for f in tqdm(filters_name):
            bandpass = np.loadtxt('/media/data/bahtinov_results/Model/Data/Filters/Filter_%s_quadrant%s.txt' %(f, Q))
            bandpass_sliced = Cutout2D(bandpass, (bandpass.shape[0]/2, bandpass.shape[1]/2), (400, 400)).data
            np.savetxt('/media/data/bahtinov_results/Model/Data/Filters/Filter_%s_quadrant%s_sliced.txt'  %(f, Q), bandpass_sliced)
            #fig = figure()
            #ax = fig.add_subplot(111)
            #ax.imshow(bandpass_sliced, cmap = 'Blues', interpolation = 'none')
        #show()



def create_bandpass_defocus(delta):
    mask = cv2.imread('/media/data/Bahtinov/bahtinov_mask4_crop.png', 0)
    quadrants = [1, 23, 4]
    mask_data = np.array(mask) / 255
    mirror = 650e3                          # diameter mirror micron
    if delta < 0:
        scaling_factor = abs(delta / 3.3e6)
        resize_mask = int(scaling_factor * mirror)
    if delta > 0:
        scaling_factor = delta / 3.3e6
        resize_mask = int(scaling_factor * mirror)
    if delta == 0 :
        resize_mask = int(1)


    filters_name = ['u', 'g', 'q', 'r', 'i', 'z']

    size = int(len(mask_data)/2)
    quadrant1 = mask_data[size:,size:]
    quadrant23 = mask_data[:,:size]
    quadrant4 = mask_data[:size,size:]

    mask_resized = cv2.resize(mask_data, (resize_mask, resize_mask))
    #mask_resized_Q23 = cv2.resize(quadrant23, (int(scaling_factor * mirror), int(scaling_factor * mirror)))
    #mask_resized_Q4 = cv2.resize(quadrant4, (int(scaling_factor * mirror), int(scaling_factor * mirror)))
    size = int(len(mask_resized)/2)
    quadrant1 = mask_resized[size:,size:]
    quadrant23 = mask_resized[:,:size]
    quadrant4 = mask_resized[:size,size:]
    fig = figure()
    ax = fig.add_subplot(131)
    ax.imshow(quadrant1, cmap = 'Greys', interpolation = 'none')
    ax = fig.add_subplot(132)
    ax.imshow(quadrant23, cmap = 'Greys', interpolation = 'none')
    ax = fig.add_subplot(133)
    ax.imshow(quadrant4, cmap = 'Greys', interpolation = 'none')
    #show()
    print quadrant1.shape
    print quadrant23.shape
    print quadrant4.shape

    mask_quadrants = [quadrant1, quadrant23, quadrant4]
    for Q in quadrants:
        for f in tqdm(filters_name):
            bandpass = np.loadtxt('/media/data/bahtinov_results/Model/Data/Filters/Filter_%s_quadrant%s.txt' %(f, Q))
            print bandpass.shape
            defocus_data = convolve(bandpass,mask_quadrants[Q])
            fig = figure()
            ax = fig.add_subplot(111)
            ax.imshow(defocus_data, cmap = 'Greys', interpolation = 'none')
            show()


#Parallel(n_jobs = 7)(delayed(create_spectrum_values)(i) for i in lambnm)
#Parallel(n_jobs = 3)(delayed(create_spectrum_bandpass)(q) for q in quadrants)
#slice_bandpass()
#create_bandpass_defocus(100)



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
