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
from astropy.table import Table
from astropy.stats import sigma_clipped_stats, SigmaClip, mad_std
from photutils import aperture_photometry, CircularAperture, CircularAnnulus, EllipticalAperture, EllipticalAnnulus
from photutils import Background2D, MedianBackground
from photutils import DAOStarFinder
import scipy.ndimage as ndimage

def background(im,clipsigma=3.0,boxsize=None,filtersize=(3,3)):
    """
    Estimate a smooth background in an image.

    Parameters
    ----------
    im : 2D numpy array
       The image to estimate the background for.
    clipsigma : float, optional
       Value to use for sigma clipping.  Default is 3.0.
    boxsize : tuple/list, optional
       Box size to use for the background estimation.  Default is (ny//10,nx//10).
    filtersize : tuple/list, optional
       Filter size to use.  Default is (3,3).

    Returns
    -------
    background : 2D numpy array
       The estimate background image.  Will be same shape as "im".

    Example
    -------
    
    back = background(im)

    """
    
    ny,nx = im.shape
    if boxsize is None:
        boxsize = (ny//109,nx//10)
    sigma_clip = SigmaClip(sigma=clipsigma)
    bkg_estimator = MedianBackground()
    bkg = Background2D(im, boxsize, filter_size=filtersize,
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    return bkg.background


def detection(im,fwhm=5,nsig=5):
    """
    Detect sources in an image.

    Parameters
    ----------
    im : 2D numpy array
       The image to estimate the background for.
    fwhm : float, optional
      The full width at half maximum of the PSF in the image.  Default is 5.
    nsig : float, optional
      The number of sigma above the background to set for the threshold.
         Default is 5.0 sigma.

    Returns
    -------
    cat : astropy table
       Catalog of detected sources and their central positions.

    Example
    -------
    
    cat = detection(im)

    """
    
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
    dtype = np.dtype([('id',int),('xpos',float),('ypos',float)])
    cat = np.zeros(len(ind1),dtype=dtype)
    cat['id'] = np.arange(len(ind1))+1
    cat['xpos'] = ind2
    cat['ypos'] = ind1
    return Table(cat)

def daodetect(im,fwhm=5.0,nsig=5.0):
    """
    Detect sources in an image using DAO technique.

    Parameters
    ----------
    im : 2D numpy array
       The image to estimate the background for.
    fwhm : float, optional
      The full width at half maximum of the PSF in the image.  Default is 5.
    nsig : float, optional
      The number of sigma above the background to set for the threshold.
         Default is 5.0 sigma.

    Returns
    -------
    cat : astropy table
       Catalog of detected sources and their central positions.

    Example
    -------
    
    cat = daodetect(im)

    """
    
    mean, median, std = sigma_clipped_stats(im, sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=nsig*std)  
    sources = daofind(im) 
    return sources
    
def aperphot(im,positions,rap=5.0,rin=10.0,rout=20.0,asemi=None,bsemi=None,theta=None):
    """
    Calculate aperture photometry for a list of sources.
    
    Use rap/rin/rout for a circules aperture.  Use asemi/bsemi/theta and rin/rout
    for an elliptical aperture (rin/rout scale asemi/bsemi for the background
    annulus).

    Parameters
    ----------
    im : 2D numpy array
       The image to estimate the background for.
    positions : list
       List of two-element positions or catalog.
    rap : float, optional
       Radius of the aperture.  Default is 5.0 pixels.
    rin : float, optional
       Radius of the inner background aperture.  Default is 10.0 pixels.
    rout : float, optional
       Radius of the outer background aperture.  Default is 20.0 pixels.
    asemi : float, optional
       Semi-major axis length (in pixels) for elliptical aperture.   If elliptical
         apertures are used, then rin/rout will be used to scale asemi/bsemi
         for the ellptical background annulus.
    bsemi : float, optional
       Semi-minor axis length (in pixels) for elliptical aperture.
    theta : float, optional
       Orientation angle (in radians) for elliptical aperture.

    Returns
    -------
    phot : astropy table
       Catalog of measured aperture photometry.

    Example
    -------
    
    phot = aperphot(im)

    """

    # Positions is a catalog
    if type(positions) is not list and type(positions) is not tuple:    
        pcat = positions
        if 'xpos' in pcat.colnames:
            positions = list(zip(np.array(pcat['xpos']),np.array(pcat['ypos'])))
        elif 'x' in pcat.colnames:
            positions = list(zip(np.array(pcat['x']),np.array(pcat['y'])))
        elif 'xcenter' in pcat.colnames:
            positions = list(zip(np.array(pcat['xcenter']),np.array(pcat['ycenter'])))
        elif 'xcentroid' in pcat.colnames:
            positions = list(zip(np.array(pcat['xcentroid']),np.array(pcat['ycentroid'])))
        else:
            raise ValueError('No X/Y positions found')

    # Elliptical aperture
    if asemi is not None and bsemi is not None and theta is not None:
        # Define the aperture right around our star
        aperture = EllipticalAperture(positions, a=asemi, b=bsemi, theta=theta)
        # Define the sky background circular annulus aperture
        annulus_aperture = EllipticalAnnulus(positions, a_in=rin*asemi, a_out=rout*asemi,
                                             b_in=rin*bsemi, b_out=rout*bsemi, theta=theta)
    # Circular aperture
    else:
        # Define the aperture right around our star
        aperture = CircularAperture(positions, r=rap)
        # Define the sky background circular annulus aperture
        annulus_aperture = CircularAnnulus(positions, r_in=rin, r_out=rout)
        
    # This turns our sky background aperture into a pixel mask that we can use to calculate the median value
    annulus_masks = annulus_aperture.to_mask(method='center')
    # Measure the median background value for each star
    bkg_median = []
    area = []
    ones = np.ones(im.shape,int)
    for i in range(len(positions)):
        # Get area in the circular aperture, account for edges        
        mask = aperture[i].to_mask()
        data = mask.multiply(ones)
        area.append(np.sum(data))
        # Get the data in the annulus
        amask = annulus_masks[i]        
        annulus_data = amask.multiply(im,fill_value=np.nan)
        annulus_data_1d = annulus_data[(amask.data > 0) & np.isfinite(annulus_data)]  # Only want positive values
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)  # calculate median
        bkg_median.append(median_sigclip)                            # add to our median list
    bkg_median = np.array(bkg_median)                                # turn into numpy array
    # Calculate the aperture photometry
    phot = aperture_photometry(im, aperture)
    # Stuff it in a table
    phot['aper_area'] = np.array(area)
    phot['annulus_median'] = bkg_median
    phot['aper_bkg'] = bkg_median * phot['aper_area']
    phot['aper_flux'] = phot['aperture_sum'] - phot['aper_bkg']  # subtract bkg contribution
    phot['mag'] = -2.5*np.log10(phot['aper_flux'].data)+25
    return phot

def morphology(im,positions=None,threshold=0):

    """
    Calculate centroid and morphology for sources.

    Parameters
    ----------
    im : 2D numpy array
       The image to estimate the background for.
    positions : list or table, optional
       List of positions or table.  Default is to fit a single source in the center.

    Returns
    -------
    cat : astropy table
       Catalog of calculated values.

    Example
    -------
    
    cat = morphology(im,pos)

    """

    ny,nx = im.shape

    # No position input
    if positions is None:
        positions = [ny//2,nx//2]
    
    # Positions is a catalog
    if type(positions) is not list and type(positions) is not tuple:    
        pcat = positions
        if 'xpos' in pcat.colnames:
            positions = list(zip(np.array(pcat['xpos']),np.array(pcat['ypos'])))
        elif 'x' in pcat.colnames:
            positions = list(zip(np.array(pcat['x']),np.array(pcat['y'])))
        elif 'xcenter' in pcat.colnames:
            positions = list(zip(np.array(pcat['xcenter']),np.array(pcat['ycenter'])))
        elif 'xcentroid' in pcat.colnames:
            positions = list(zip(np.array(pcat['xcentroid']),np.array(pcat['ycentroid'])))
        else:
            raise ValueError('No X/Y positions found')
    
    # Masking negative pixels
    im[im<threshold] = 0

    # Background values
    back = np.median(im)
    bsig = mad_std(im)
        
    # Positions loop
    npos = len(positions)
    dtype = np.dtype([('id',int),('xcentroid',float),('ycentroid',float),('npix',int),('sigmax',float),
                      ('sigmay',float),('sigmaxy',float),('asemi',float),('bsemi',float),('theta',float)])
    cat = np.zeros(npos,dtype=dtype)
    cat['id'] = np.arange(npos)+1
    for i in range(npos):
        # Iterate to get the right size
        flag = 0
        count = 0
        mnx,mny = positions[i]
        sigx = sigy = 5
        maxiter = 1 # 3
        while (flag==0):
            # sub image
            xlo = int(np.maximum(mnx-2.5*sigx,0))
            xhi = int(np.minimum(mnx+2.5*sigx,nx))
            ylo = int(np.maximum(mny-2.5*sigy,0))
            yhi = int(np.minimum(mny+2.5*sigy,ny))
            im2 = im[ylo:yhi,xlo:xhi]
            mask = np.ones(im2.shape,bool)
            mn = np.mean(im2[im2>back+3*bsig])
            thresh = np.maximum(mn*0.1+back,1)
            ngood = np.sum(im2>=thresh)
            if ngood<5:
                thresh *= 0.5
                ngood = np.sum(im2>=thresh)
            mask[im2<thresh] = 0
            # Create array of x-values for the image
            ny2,nx2 = im2.shape
            xx,yy = np.meshgrid(np.arange(nx2)+xlo,np.arange(ny2)+ylo)
            totflux = np.sum(im2*mask)
            # First moments
            old_mnx = mnx
            old_mny = mny
            mnx = np.sum(im2*mask*xx) / totflux
            mny = np.sum(im2*mask*yy) / totflux
            # Second moments
            old_sigx = sigx
            old_sigy = sigy
            sigx2 = np.sum(im2*mask*(xx-mnx)**2) / totflux
            sigx = np.sqrt(sigx2)
            sigy2 = np.sum(im2*mask*(yy-mny)**2) / totflux
            sigy = np.sqrt(sigy2)
            
            # Stopping criteria
            posdiff = np.sqrt((mnx-old_mnx)**2+(mny-old_mny)**2)
            sigdiff = np.sqrt((sigx-old_sigx)**2+(sigy-old_sigy)**2)            
            if (posdiff<1 and sigdiff<2) or sigx==0 or sigy==0 or count>maxiter:
                flag = 1
            count += 1
            
        # Final values
        sigxy = np.sum(im2*mask*(xx-mnx)*(yy-mny)) / totflux
        # Ellipse parameters
        asemi = np.sqrt( 0.5*(sigx2+sigy2) + np.sqrt(((sigx2-sigy2)*0.5)**2 + sigxy**2 ) )
        bsemi = np.sqrt( 0.5*(sigx2+sigy2) - np.sqrt(((sigx2-sigy2)*0.5)**2 + sigxy**2 ) )
        theta = np.rad2deg(0.5*np.arctan2(2*sigxy,sigx2-sigy2))
        cat['xcentroid'][i] = mnx
        cat['ycentroid'][i] = mny
        cat['npix'][i] = np.sum(mask)
        cat['sigmax'][i] = sigx
        cat['sigmay'][i] = sigy
        cat['sigmaxy'][i] = sigxy
        cat['asemi'][i] = asemi
        cat['bsemi'][i] = bsemi
        cat['theta'][i] = theta

    return Table(cat)
