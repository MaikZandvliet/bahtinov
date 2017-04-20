#!/usr/bin/env python

# ====================================================================================
#               Bahtinov Mask Analysis Software to Determine (De)Focus
# ====================================================================================

# Import required modules
from __future__ import division
import os
from pylab import rcParams
import sys
import cv2
import numpy as np
from pylab import *
from math import *
import scipy
import time
from scipy import stats
from astropy.stats import sigma_clip
from astropy.nddata import Cutout2D
from mayavi import mlab
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show, rc, cm
from mpl_toolkits.mplot3d import Axes3D
from astropy.stats import sigma_clipped_stats
rcParams['figure.figsize'] = 16, 12

today_utc_date = datetime.datetime.utcnow().strftime('%Y-%m-%d')
today_utc_time = time.strftime('%c', time.gmtime(time.time()))

mask = cv2.imread('/media/data/Bahtinov/bahtinov_mask1_crop.png',0)
mask_data = np.array(mask)/255

factor = 600/95.04
# Apply fft in 2d to mask divided into quadrants
mask_fft_quadrant1 = np.fft.fft2(mask_data[:int(len(mask_data)/2),int(len(mask_data)/2):])
mask_fft_quadrant23 = np.fft.fft2(mask_data[:,:int(len(mask_data)/2)])
mask_fft_quadrant4 = np.fft.fft2(mask_data[int(len(mask_data)/2):,int(len(mask_data)/2):])


# Shift the zero-frequency component to the center of the spectrum.
mask_fft_quadrant1_shift = np.fft.fftshift(mask_fft_quadrant1)
mask_fft_quadrant23_shift = np.fft.fftshift(mask_fft_quadrant23)
mask_fft_quadrant4_shift = np.fft.fftshift(mask_fft_quadrant4)

# obtain magnitude spectrum per quadrant linear
magnitude_spectrum_quadrant1 = (abs(mask_fft_quadrant1_shift))
magnitude_spectrum_quadrant23 =  (abs(mask_fft_quadrant23_shift))
magnitude_spectrum_quadrant4 =  (abs(mask_fft_quadrant4_shift))



fig = figure()
ax = fig.add_subplot(221)
ax.set_title("Bahtinov mask input image")
ax.imshow(mask_data, cmap = 'gray', interpolation = 'none')
ax = fig.add_subplot(222)
ax.set_title("FFT Quadrant 1")
ax.imshow(magnitude_spectrum_quadrant1, interpolation = 'none')
ax = fig.add_subplot(223)
ax.set_title("FFT Quadrant 2 & 3")
ax.imshow(magnitude_spectrum_quadrant23, interpolation = 'none')
ax = fig.add_subplot(224)
ax.set_title("FFT Quadrant 4")
ax.imshow(magnitude_spectrum_quadrant4, interpolation = 'none')
plt.tight_layout()
fig.savefig('/media/data/bahtinov_results/FFT_bahtinovmask.png')
#show()
'''
diffraction = (abs(np.fft.fftshift(np.fft.fft2(mask_data))))
mlab.figure(size = (1920/1.5,1080/1.5))
surf = mlab.surf(diffraction, warp_scale = 'auto')
#surf = mlab.surf(magnitude_spectrum_quadrant23, warp_scale = 'auto')
#surf = mlab.surf(magnitude_spectrum_quadrant4, warp_scale = 'auto')
mlab.colorbar(orientation='vertical', nb_labels=10, label_fmt = '%.2e')
ax = mlab.axes(line_width = .5, nb_labels = 5, xlabel = 'x [pixel]', ylabel = 'y [pixel]', zlabel = 'Intensity', x_axis_visibility = True, y_axis_visibility = True,
    ranges = [-10560/2, 10560/2, -10600/2, 10600/2, 0, np.max(diffraction)])
ax.axes.font_factor = 1
mlab.show()
'''
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
        image.workdir = workdir
        image.image_path  = image_path                                  # full image path
        image.title = image.image_path.split('/')[-1]                   # name of the image with extension
        image.name = name                                               # name of the image without extension
        image.number = float(image.title.split('_')[-2])                # image number
        image.image = fits.open(image.image_path)                       # opening fits image
        image.data = image.image[1].data                                # image data
        image.data = np.asarray(image.data - image.bias , dtype = np.float)
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
            image.fig.savefig(image.workdir +'Focusrun/' + today_utc_date + '/Plots/' + str(image.name) + '/' + str(image.name) + '_' + str(image.X) + '_' + str(image.Y) + '.png')
            plt.close()












'''

def grating_equation(m, lmbda, a, offset_angle = 0):
    return np.arcsin(m * lmbda / a)

angle_Bahtinov = math.radians(20)
lamb = 0.55 * 10**(-6)      # wavelength (m)  500 nm
k = 2.0 * np.pi/lamb        # wave number
d = 7.9 * 10**(-3)          # slit width  (m)
a = d                       # slit spacing  (m)
l_x = d                     # slit width (m)
l_y = 0.3                   # slit length (m)
dmm = d * 1000.0            # slit width  (mm)
N = 26                      # number of slits
lambnm = lamb * 10**9       # wavelength  (nm)
I_0 = 1e6                   # Intesity at 0
num = 300                   # number of points

M = np.linspace(0,49,50)
angles = grating_equation(M, lamb, a)
inten = np.zeros(num)
thetx = np.zeros(num)
#theta = np.linspace(-np.pi/10000,np.pi/10000,500)
#beta = k * d * np.sin(theta)/2.0
#alpha = k * a * np.sin(theta)/2.0
#inten = (np.sin(beta)/beta)**2 * (np.sin(N*alpha)/np.sin(alpha))**2      # Intensity
angle = np.linspace(-num,num,num)

theta =  np.linspace(-2*np.pi/num,2*np.pi/num, num)
inten = np.zeros((len(theta),len(theta)))
thetx = np.zeros(len(theta))
y_, z_ = np.linspace(-10560/2, 10560/2, len(theta)), np.linspace(-10600/2, 10600/2, len(theta))
y, z = np.meshgrid(y_,z_)
beta = ((k * d * z) / 2) * np.sin(theta)
alpha = ((k * a * y) / 2) * np.sin(theta)
inten = I_0/N**2 * (np.sin(beta)/beta)**2 * (np.sin(N*alpha)/np.sin(alpha))**2

mlab.figure(size = (1920/1.5,1080/1.5))
surf = mlab.surf(inten, warp_scale = 'auto')
mlab.colorbar(orientation='vertical', nb_labels=10, label_fmt = '%.2e')
M_20 = cv2.getRotationMatrix2D((len(inten)/2, len(inten)/2), angle_Bahtinov, 1)
inten_20 = cv2.warpAffine(inten, M_20, (len(inten), len(inten)))
M_m20 = cv2.getRotationMatrix2D((len(inten)/2, len(inten)/2), -angle_Bahtinov, 1)
inten_m20 = cv2.warpAffine(inten, M_m20, (len(inten), len(inten)))
#surf = mlab.surf(inten_20, warp_scale = 'auto')
#surf = mlab.surf(inten_m20, warp_scale = 'auto')
ax = mlab.axes(line_width = .5, nb_labels = 5, xlabel = 'x [pixel]', ylabel = 'y [pixel]', zlabel = 'Intensity', x_axis_visibility = True, y_axis_visibility = True,
    ranges = [-10560/2, 10560/2, -10600/2, 10600/2, 0, np.max(inten)])
ax.axes.font_factor = 1
#mlab.show()
'''
'''
fig = figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(y, z, inten, cmap = cm.coolwarm, linewidth = 0)
ax.set_xlabel('x [pixel]')
ax.set_ylabel('y [pixel]')
ax.set_zlabel('Intensity')
show()
'''
