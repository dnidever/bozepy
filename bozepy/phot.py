#!/usr/bin/env python

"""PHOT.PY - Generic CCD image calibration/reduction

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20211021'  # yyyymmdd    

import os
import numpy as np
from glob import glob
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from astropy.stats import sigma_clipped_stats, SigmaClip, mad_std
from photutils import aperture_photometry, CircularAperture, CircularAnnulus
from photutils import Background2D, MedianBackground
from photutils import DAOStarFinder

def background(im,clipsigma=3.0,boxsize=(200,200),filtersize=(3,3)):
    sigma_clip = SigmaClip(sigma=clipsigma)
    bkg_estimator = MedianBackground()
    bkg = Background2D(im, boxsize, filter_size=filtersize,
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    return bkg.background

def detection(im,fwhm=5,nsig=5):
    # Smooth the image
    smim = ndimage.gaussian_filter(im, sigma=(fwhm/2.35, fwhm/2.35), order=0)
    # Calculate the median and scatter
    mean, median, std = sigma_clipped_stats(smim, sigma=3.0)
    # Shift the images 4 ways
    smim1 = np.roll(smim,1,axis=0)
    smim2 = np.roll(smim,-1,axis=0)
    smim3 = np.roll(smim,1,axis=1)
    smim4 = np.roll(smim,-1,axis=1)
    # Set the threshold
    thresh = median + nsig*std
    # Do the detection
    det = (smim > thresh) & (smim>smim1) & (smim>smim2) & (smim>smim3) & (smim>smim4)
    ind1, ind2 = np.where(det == True)
    # Make a table
    dtype = np.dtype([('xpos',float),('ypos',float)])
    cat = np.zeros(len(ind1),dtype=dtype)
    cat['xpos'] = ind2
    cat['ypos'] = ind1
    return cat

def daodetect(im,fwhm=5.0,nsig=5.0):
    mean, median, std = sigma_clipped_stats(im, sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=nsig*std)  
    sources = daofind(sim) 
    return sources
    
def aperphot(im,positions,r=None,rin=None,rout=None):
    # Define the aperture right around our star
    aperture = CircularAperture(positions, r=r)
    # Define the sky background circular annulus aperture
    annulus_aperture = CircularAnnulus(positions, r_in=rin, r_out=rout)
    # This turns our sky background aperture into a pixel mask that we can use to calculate the median value
    annulus_masks = annulus_aperture.to_mask(method='center')
    # Measure the median background value for each star
    bkg_median = []
    for mask in annulus_masks:  # loop over the stars
        # Get the data in the annulus
        annulus_data = mask.multiply(im)
        annulus_data_1d = annulus_data[mask.data > 0]                # Only want positive values
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)  # calculate median
        bkg_median.append(median_sigclip)                            # add to our median list
    bkg_median = np.array(bkg_median)                                # turn into numpy array
    # Calculate the aperture photometry
    phot = aperture_photometry(im, aperture)
    # Stuff it in a table
    phot['annulus_median'] = bkg_median
    phot['aper_bkg'] = bkg_median * aperture.area                      # the number of bkg counts in our aperture
    phot['aper_flux'] = phot['aperture_sum'] - phot['aper_bkg']  # subtract bkg contribution
    phot['mag'] = -2.5*np.log10(phot['aper_flux'].data)+25
    return phot

def morphology(data,threshold=0):
    """ Measure centroid and morphology."""
    # Masking
    if threshold is not None:
        data[data<threshold] = 0
    # Create array of x-values for the image
    ny,nx = data.shape
    xx,yy = np.meshgrid(np.arange(nx),np.arange(ny))
    # First moments
    mnx = np.sum(data*xx) / np.sum(data)
    mny = np.sum(data*yy) / np.sum(data)
    # Second moments
    sigx2 = np.sum(data*(xx-mnx)**2) / np.sum(data)
    sigx = np.sqrt(sigx2)
    sigy2 = np.sum(data*(yy-mny)**2) / np.sum(data)
    sigy = np.sqrt(sigy2)
    sigxy = np.sum(data*(xx-mnx)*(yy-mny)) / np.sum(data)
    # Ellipse parameters
    asemi = np.sqrt( 0.5*(sigx2+sigy2) + np.sqrt(((sigx2-sigy2)*0.5)**2 + sigxy**2 ) )
    bsemi = np.sqrt( 0.5*(sigx2+sigy2) - np.sqrt(((sigx2-sigy2)*0.5)**2 + sigxy**2 ) )
    theta = np.rad2deg(0.5*np.arctan2(2*sigxy,sigx2-sigy2))
    dtype = np.dtype([('xcentroid',float),('ycentroid',float),('sigmax',float),('sigmay',float),('sigmaxy',float),
                      ('asemi',float),('bsemi',float),('theta',float)])
    cat = np.zeros(1,dtype=dtype)
    cat['xcentroid'] = mnx
    cat['ycentroid'] = mny
    cat['sigmax'] = sigx
    cat['sigmay'] = sigy
    cat['sigmaxy'] = sigxy
    cat['asemi'] = asemi
    cat['bsemi'] = bsemi
    cat['theta'] = theta

    return cat
