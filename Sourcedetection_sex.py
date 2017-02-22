#!/usr/bin/env python

# ==============================================================================
# Source position detection using SExtractor
# ==============================================================================

# Modules to include
# sewpy obtained via http://sewpy.readthedocs.io/en/latest/index.html

from __future__ import division
import time
import datetime
import glob
import sys, os
import numpy as np
import sewpy
import logging
from astropy.io import fits
import argparse
start = time.time()

sys.path.insert(0, os.path.abspath('../'))

#logging.basicConfig(format='%(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)

# Get command line arguments
params = argparse.ArgumentParser(description='User parameters.')
params.add_argument('--image', default=None, help='Single image to run script on.')
params.add_argument('--path', default=None, help='Path to images to run script on.')
args = params.parse_args()

if len(sys.argv) < 3:
    print 'Usage:', sys.argv[0], '--path directory path'
    print 'Usage:', sys.argv[0], '--image image.fits'
    sys.exit(1)

print '=============== Source Detection Started ==============='
# Setup sewpy parameters for SExtractor
# Use default files from local directory
sew = sewpy.SEW(
        params = ['NUMBER', 'X_IMAGE', 'X2_IMAGE', 'Y_IMAGE', 'Y2_IMAGE', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'CLASS_STAR' ],
        configfilepath = '/media/data/scripts/bahtinov/default_sextractor/configuration.sex',
        config = {'FILTER_NAME' : '/media/data/scripts/bahtinov/default_sextractor/default.conv',
            'STARNNW_NAME' : '/media/data/scripts/bahtinov/default_sextractor/default.nnw'},
        sexpath= '/usr/bin/sextractor',
        workdir = '/media/data/scripts/bahtinov/SExtractor'
        )

# Determine positions of sources for one image
if args.image is not None:
    file = args.image
    data_reduced = {}
    candidates = {}
    name = (file.split('/')[-1]).split('.')[0]
    candidates = sew(file)
    # Create dictionary containing the data using the parameters set in SExtractor
    parameters = open('SExtractor/params.txt', 'r').read().split('\n')
    for n in range(len(parameters)-1):
        candidates['%s' %parameters[n]] = np.loadtxt('SExtractor/' + str(name) + '.cat.txt', unpack = True)[n]
    # Filter sources having a Kron-like aperture ranging between -15 and -20
    for n in range(len(parameters)-1):
        data_reduced['%s' %parameters[n]] = candidates['%s' %parameters[n]][(candidates['FLUX_AUTO'] > 1e5) & (candidates['CLASS_STAR'] != 0) & (candidates['X_IMAGE'] != 1162 )]
    #print data_reduced['X_IMAGE'], data_reduced['Y_IMAGE'], data_reduced['FLUX_AUTO'], data_reduced['CLASS_STAR']

    # Saving the filtered data using original image name
    data = np.column_stack([data_reduced['X_IMAGE'], data_reduced['X2_IMAGE'], data_reduced['Y_IMAGE'], data_reduced['Y2_IMAGE'], data_reduced['FLUX_AUTO'], data_reduced['FLUXERR_AUTO']])
    np.savetxt('SExtractor/' + str(name) + '_reduced.txt', data)

# Determine positions of sources for all images in path having .fits extension
if args.path is not None:
    files = glob.glob(args.path+'*test.fits')
    for i in xrange(len(files)):
        file = files[i]
        data_reduced = {}
        candidates = {}
        name = (file.split('/')[-1]).split('.')[0]
        candidates = sew(file)
        # Create dictionary containing the data using the parameters set in SExtractor
        parameters = open('SExtractor/params.txt', 'r').read().split('\n')
        for n in range(len(parameters)-1):
            candidates['%s' %parameters[n]] = np.loadtxt('SExtractor/' + str(name) + '.cat.txt', unpack = True)[n]
        # Filter sources having a Kron-like aperture ranging between -15 and -20
        for n in range(len(parameters)-1):
            data_reduced['%s' %parameters[n]] = candidates['%s' %parameters[n]][(candidates['FLUX_AUTO'] > 1e5) & (candidates['CLASS_STAR'] != 0) & (candidates['X_IMAGE'] != 1162 )]
        # Saving the filtered data using original image name per image in path
        data = np.column_stack([data_reduced['X_IMAGE'], data_reduced['X2_IMAGE'], data_reduced['Y_IMAGE'], data_reduced['Y2_IMAGE'], data_reduced['FLUX_AUTO'], data_reduced['FLUXERR_AUTO']])
        np.savetxt('SExtractor/' + str(name) + '_reduced.txt', data)

# Delete unnessary files generate by sew
os.system('rm SExtractor/conv.txt')
os.system('rm SExtractor/default.psf')


period = time.time() - start
print '\nThe computation time was %.3f seconds\n' %(period)
print '=============== Source Detection Ended ==============='

# python Sourcedetection_sex.py --path /media/maik/Maik/MeerLICHT/Data/2016_09_26/Fits_from_raw/New_images/Focusrun/
