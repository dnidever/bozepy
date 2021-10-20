#!/usr/bin/env python

# CCDPROC code

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import os
from glob import glob
import time
import re
import subprocess
#import matplotlib
#matplotlib.use('nbagg')

def datadir():
    """ Get package data directory."""
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = codedir+'/data/'
    return datadir

def ccdlist(input=None):
    if input is None: input='*.fits'
    files = glob(input)
    nfiles = len(files)
    dt = np.dtype([('file',np.str,100),('object',np.str,100),('naxis1',int),('naxis2',int),
                      ('imagetyp',np.str,100),('exptime',float),('filter',np.str,100)])
    cat = np.zeros(nfiles,dtype=dt)
    for i,f in enumerate(files):
        base = os.path.basename(f)
        base = base.split('.')[0]
        h = fits.getheader(f)
        cat['file'][i] = f
        cat['object'][i] = h.get('object')
        cat['naxis1'][i] = h.get('naxis1')
        cat['naxis2'][i] = h.get('naxis2')
        cat['imagetyp'][i] = h.get('imagetyp')
        cat['exptime'][i] = h.get('exptime')
        cat['filter'][i] = h.get('filter')
        print(base+'  '+str(cat['naxis1'][i])+'  '+str(cat['naxis2'][i])+'  '+cat['imagetyp'][i]+'  '+str(cat['exptime'][i])+'  '+cat['filter'][i])
    return cat

def library():
    """ Get calibration library file info."""
    ddir = datadir()
    files = glob(ddir+'*.fit')
    nfiles = len(files)
    if nfiles==0:
        return None
    cat = np.zeros(nfiles,dtype=np.dtype([('name',np.str,100),('type',np.str,50),()]))
    for i in range(nfiles):
        head = fits.getheader(files[i])
        cat['name'][i] = files[i]
        cat['type'][i] = head['exptype']
    return cat

def fixheader(head):
    """ Update the headers."""
    head2 = head.copy()
    head2['BIASSEC1'] = '[1:41,1:2728]'
    head2['BIASSEC2'] = '[3430:3465,1:2728]'
    head2['TRIMSEC'] = '[42:3429,15:2726]'
    head2['DATASEC'] = '[42:3429,15:2726]'
    head2['RDNOISE'] = (4.5, 'readnoise in e-')
    head2['GAIN'] = (0.15759900212287903, 'Electronic gain in e-/ADU')
    head2['BUNIT'] = 'ADU'
    return head2

    
def overscan(im,head):
    """ This calculate the overscan and subtracts it from the data and then trims off the overscan region"""
    # y = [0:40] and [3429:3464]
    # x = [0:13] and [2726:2727]
    # DATA = [14:2725,41:3428]
    # 2712 x 3388
    nx,ny = im.shape

    # Use trimsec
    trimsec = head.get('TRIMSEC')
    if trimsec is None:
        raise ValueError('No TRIMSEC found in header')
    trim = [int(s) for s in re.findall(r'\d+',trimsec)]

    # biassec
    biassec = head.get('BIASSEC')
    biassec2 = None
    if biassec is None:
        biassec = head.get('BIASSEC1')
        biassec2 = head.get('BIASSEC2')        
    if biassec is None:
        raise ValueError('No BIASSEC found in header')
    bias = [int(s) for s in re.findall(r'\d+',biassec)]    

    # Y first, then X
    o = im[bias[2]-1:bias[3],bias[0]-1:bias[1]]
    # check for second biassec
    if biassec2 is not None:
        bias2 = [int(s) for s in re.findall(r'\d+',biassec2)]
        o2 = im[bias2[2]-1:bias2[3],bias2[0]-1:bias2[1]]
        o = np.hstack((o,o2))
        
    # Subtract overscan
    oshape = o.shape
    if oshape[0] > oshape[1]:
        # Take the mean        
        mno = np.mean(o,axis=1)
        # Fit line to it
        coef = np.polyfit(np.arange(nx),mno,1)
        fit = np.poly1d(coef)(np.arange(nx))
        # Subtract from entire image
        oim = np.repeat(fit,ny).reshape(nx,ny)
        out = im.astype(float)-oim
    else:
        # Take the mean        
        mno = np.mean(o,axis=0)
        # Fit line to it
        coef = np.polyfit(np.arange(ny),mno,1)
        fit = np.poly1d(coef)(np.arange(ny))
        # Subtract from entire image
        oim = np.repeat(fit,nx).reshape(nx,ny)
        out = im.astype(float)-oim        
        
    # Trim the overscan
    out = out[trim[2]-1:trim[3],trim[0]-1:trim[1]]
    #out = out[14:2726,41:3429]
    # Update header
    nx1, ny1 = out.shape
    head2 = head.copy()
    head2['NAXIS1'] = ny1
    head2['NAXIS2'] = nx1
    head2['BITPIX'] = -32
    if biassec2 is not None:
        head2['BIASSEC1'] = biassec
        head2['BIASSEC2'] = biassec2
    else:
        head2['BIASSEC'] = biassec        
    head2['TRIMSEC'] = trimsec
    head2['OVSNMEAN'] = np.mean(oim)
    head2['TRIM'] = time.ctime()+' Trim is '+trimsec
    if biassec2 is not None:
        head2['OVERSCAN'] = time.ctime()+' Overscan is '+biassec+' and '+biassec2+', mean '+str(np.mean(oim))
    else:
        head2['OVERSCAN'] = time.ctime()+' Overscan is '+biassec+', mean '+str(np.mean(oim))    
    return out, head2
    
def masterbias(files,outfile=None,clobber=True):
    """
    Load the bias images.  Overscan correct and trim them.  Then average them.

    Parameters
    ----------
    files : list
        List of bias FITS files.
    outfile : string, optional
        Filename to write the master bias image to.
    clobber : bool, optional
        If the output file already exists, then overwrite it.

    Returns
    -------
    aim : numpy image
        The 2D master bias image.
    ahead : header dictionary
        The master bias header.

    Example
    -------

    bias, bhead = masterbias(bias_files)

    """

    nfiles = len(files)
    head0 = fits.headfits(files[0])
    nx = head['NAXIS1']
    ny = head['NAXIS2']    
    imarr = np.zeros((ny, nx, nfiles),float)
    for i in range(nfiles):
        print(str(i+1)+' '+files[i])
        im,head = fits.getdata(files[i],0,header=True)
        im2,head2 = ccdproc(im,head)
        imarr[:,:,i] = im2
        if i==0: ahead=head2.copy()
        ahead['CMB'+str(i+1)] = files[i]
    aim = np.mean(imarr,axis=2)
    ahead['NCOMBINE'] = nfiles
    # Output file
    if outfile is not None:
        if os.path.exists(outfile):
            if clobber is False:
                raise ValueError(outfile+' already exists and clobber=False')
            else:
                os.remove(outfile)
        print('Writing master bias to '+outfile)
        hdu = fits.PrimaryHDU(aim,ahead).writeto(outfile)
    
    return aim, ahead


def masterdark(files,zero,outfile=None,clobber=True):
    """
    Load the dark images.  Overscan correct and trim them.  zero subtract.  Then average them.

    Parameters
    ----------
    files : list
        List of dark FITS files.
    outfile : string, optional
        Filename to write the master dark image to.
    clobber : bool, optional
        If the output file already exists, then overwrite it.

    Returns
    -------
    aim : numpy image
        The 2D master dark image.
    ahead : header dictionary
        The master dark header.

    Example
    -------

    dark, dhead = masterdark(dark_files)

    """

    nfiles = len(files)
    imarr = np.zeros((2712, 3388, nfiles),float)
    for i in range(nfiles):
        print(str(i+1)+' '+files[i])
        im,head = fits.getdata(files[i],0,header=True)
        im2,head2 = ccdproc(im,head,zero)
        imarr[:,:,i] = im2 / np.median(im2)
        if i==0: ahead=head2.copy()
        ahead['CMB'+str(i+1)] = files[i]
    # Take average
    aim = np.mean(imarr,axis=2)
    # Divide by exposure time
    aim /= head['exptime']
    ahead['NCOMBINE'] = nfiles
    # Output file
    if outfile is not None:
        if os.path.exists(outfile):
            if clobber is False:
                raise ValueError(outfile+' already exists and clobber=False')
            else:
                os.remove(outfile)
        print('Writing master dark to '+outfile)
        hdu = fits.PrimaryHDU(aim,ahead).writeto(outfile)
    
    return aim, ahead

def masterflat(files,zero,dark,outfile=None,clobber=True):
    """
    Load the flat images.  Overscan correct and trim them.  Bias and dark subtract. Then divide by median and average them.

    Parameters
    ----------
    files : list
        List of flat FITS files.
    outfile : string, optional
        Filename to write the master flat image to.
    clobber : bool, optional
        If the output file already exists, then overwrite it.

    Returns
    -------
    aim : numpy image
        The 2D master flat image.
    ahead : header dictionary
        The master flat header.

    Example
    -------

    flat, fhead = masterflat(flat_files)

    """

    nfiles = len(files)
    imarr = np.zeros((2712, 3388, nfiles),float)
    for i in range(nfiles):
        im,head = fits.getdata(files[i],0,header=True)
        print(str(i+1)+' '+files[i]+' '+str(head.get('FILTER')))
        im2,head2 = ccdproc(im,head,zero,dark)
        imarr[:,:,i] = im2 / np.median(im2)
        if i==0: ahead=head2.copy()
        ahead['CMB'+str(i+1)] = files[i]
    aim = np.mean(imarr,axis=2)
    ahead['NCOMBINE'] = nfiles

    # Output file
    if outfile is not None:
        if os.path.exists(outfile):
            if clobber is False:
                raise ValueError(outfile+' already exists and clobber=False')
            else:
                os.remove(outfile)
        print('Writing master flat to '+outfile)
        hdu = fits.PrimaryHDU(aim,ahead).writeto(outfile)

    return aim, ahead

def ccdproc(data,head=None,bpm=None,zero=None,dark=None,flat=None,outfile=None,verbose=False,
            clobber=True,compress=False):
    """
    Overscan subtract, trim, subtract master zero, subtract master dark, flat field.

    Parameters
    ----------
    data : list or numpy 2D array
        This can either be a list of image filenames or a 2D image.
    head : header dictionary, optional
        The header if a single image is input.
    bpm : filename or numpy 2D array, optional
        The master bad pixel mask.  Either the 2D image or the filename.
    zero : filename or numpy 2D array, optional
        The master bias.  Either the 2D image or the filename.
    dark : filename or numpy 2D array, optional
        The master dark.  Either the 2D image or the filename.
    flat : filename or numpy 2D array, optional
        The master flat.  Either the 2D image or the filename.
    outfile : string, optional
        Filename to write the processed image to.
    verbose : boolean, optional
        Verbose output to the screen.
    clobber : boolean, optional
        If the output file already exists, then overwrite it.
    compress : boolean, optional
        Gzip compress output file.  Default is False.

    Returns
    -------
    fim : numpy image
        The 2D processed image.
    fhead : header dictionary
        The header for the processed image.

    Example
    -------

    flat, fhead = ccdproc(files,bpm,zero,dark,flat)

    """

    # Filename input
    if type(data) is str:
        if os.path.exists(data):
            if verbose:
                print('Loading '+data) 
            im,head = fits.getdata(data,0,header=True)
        else:
            raise ValueError(data+' NOT FOUND')
    # Image input
    else:
        im = data
        if head is None:
            raise ValueError('Header not input')


        
    # Fix header, if necessary
    if (head.get('TRIMSEC') is None) | (head.get('BIASSEC') is None):
        head = fixheader(head)
        
    # Overscan subtract and trim
    #---------------------------
    if head.get('OVERSCAN') is None:
        fim,fhead = overscan(im,head)
    else:
        print('Already OVERSCAN corrected')
        fim = im.copy()
        fhead = head.copy()

    # Initialize error and mask image
    error = np.zeros(fim.shape,float)
    mask = np.zeros(fim.shape,np.uint)

    # Bad pixel mask
    #---------------
    if (bpm is not None):
        # Not corrected yet
        if head.get('BPMCOR') is None:
            # Filename input
            if type(bpm) is str:
                if os.path.exists(bpm):
                    bpmim,bpmhead = fits.getdata(bpm,0,header=True)
                else:
                    raise ValueError(bpm+' NOT FOUND')
            # Image input
            else:
                bpmim = bpm
            # Do the correction
            nbadbpm = np.sum(bpm>0)
            if nbadbpm>0:
                fim[bpm>0] = 0.0
                mask[bpm>0] = 1
                error[bpm>0] = 1e30
            fhead['BPMCOR'] = time.ctime()+' masked '+str(nbadbpm)+' bad pixels'
        # Corrected already
        else:
            print('Already ZERO subtracted')
            
    # Set mask and error for saturated pixels
    #----------------------------------------
    saturation = head.get('saturate')
    if saturation is None:
        saturation = 64000
        sat = (fim>saturation) & (mask==0)
        mask[sat] = 2
        error[sat] = 1e30
    
    # Subtract master bias
    #---------------------
    if (zero is not None):
        # Not corrected yet
        if head.get('ZEROCOR') is None:
            # Filename input
            if type(zero) is str:
                if os.path.exists(zero):
                    zeroim,zerohead = fits.getdata(zero,0,header=True)
                else:
                    raise ValueError(zero+' NOT FOUND')
            # Image input
            else:
                zeroim = zero
            # Do the correction
            fim[mask==0] -= zeroim[mask==0]
            fhead['ZEROCOR'] = time.ctime()+' mean %6.2f, stdev %6.2f' % (np.mean(zeroim),np.std(zeroim))
        # Corrected already
        else:
            print('Already ZERO subtracted')

    # Calculate error array
    #------------------------
    gain = head.get('gain')
    if gain is None:
        gain = 1.0
    rdnoise = head.get('rdnoise')
    if rdnoise is None:
        rdnoise = 0.0
    # Add Poisson noise and readnoise in quadrature
    error[mask==0] = np.sqrt(np.maximum(fim[mask==0]/gain,0)+rdnoise**2)
            
    # Subtract master dark scaled to this exposure time
    #--------------------------------------------------
    if (dark is not None):
        # Not corrected yet
        if head.get('DARKCOR') is None:
            # Filename input
            if type(dark) is str:
                if os.path.exists(dark):
                    darkim,darkhead = fits.getdata(dark,0,header=True)
                else:
                    raise ValueError(dark+' NOT FOUND')
            # Image input
            else:
                darkim = dark
            # Do the correction
            fim[mask==0] -= darkim[mask==0]*head['exptime']
            fhead['DARKCOR'] = time.ctime()+' mean %6.2f, stdev %6.2f' % \
                               (np.mean(darkim*head['exptime']),np.std(darkim*head['exptime']))        
        # Corrected already
        else:
            print('Already DARK corrected')
            
    # Flat field
    #-----------
    if (flat is not None):
        # Not corrected yet
        if head.get('FLATCOR') is None:
            # Filename input
            if type(flat) is str:
                if os.path.exists(flat):
                    flatim,flathead = fits.getdata(flat,0,header=True)
                else:
                    raise ValueError(flat+' NOT FOUND')
            # Image input
            else:
                flatim = flat
            # Do the correction
            fim[mask==0] /= flatim[mask==0]
            error[mask==0] /= flatim[mask==0]  # need to divide error as well
            fhead['FLATCOR'] = time.ctime()+' mean %6.2f, stdev %6.2f' % (np.mean(flatim),np.std(flatim))
        # Already corrected
        else:
            print('Already FLAT corrected')

    fhead['CCDPROC'] = time.ctime()+' CCD processing done'

    # Write to output file
    if outfile is not None:
        if os.path.exists(outfile):
            if clobber is False:
                raise ValueError(outfile+' already exists and clobber=False')
            else:
                os.remove(outfile)
        print('Writing processed file to '+outfile)
        hdulist = fits.HDUList()
        hdu = fits.PrimaryHDU(fim,fhead).writeto(outfile)
        hdulist.append(hdu)
        # Add error image
        hdulist.append(fits.ImageHDU(error))
        hdulist[1].header['BUNIT'] = 'error'
        # Add mask image
        hdulist.append(fits.ImageHDU(mask))
        hdulist[2].header['BUNIT'] = 'mask'
        hdulist[2].header['HISTORY'] = ' Mask values'
        hdulist[2].header['HISTORY'] = ' 0: good'        
        hdulist[2].header['HISTORY'] = ' 1: bad pixel'
        hdulist[2].header['HISTORY'] = ' 2: saturated'   
        hdulist.writeto(outfile,overwrite=clobber)
        hdulist.close()
        # Gzip compress
        if compress:
            out = subprocess.run(['gzip',outfile])
        
    return fim, fhead

def redrun(files):
    """
    Automatically reduce an entire run of data.
    """

    tab = ccdlist(files)
    # Step 1) Reduce bias/zeros
    #--------------------------
    
    # Step 2) Make master zero
    #--------------------------
    
    # Step 3) Reduce darks
    #---------------------
    
    # Step 4) Make master dark
    #-------------------------
    
    # Step 5) Reduce flats
    #---------------------
    
    # Step 6) Make master flat
    #-------------------------
    
    # Step 7) Reduce science/object exposures
    #----------------------------------------
    
    pass

def autored(datadir='.'):
    """ Automatically pick up FITS files from a directory and reduce."""

    # While loop
    count = 0
    wait = 10
    flag = 0
    lastfiles = []
    while (flag==0):
        # Check directory for new files
        files = glob(datadir+'/*.fit*')
        nfiles = len(files)
        
        # If there are new files then reduce them
        if files!=lastfiles:
            newfiles = [f for f in files if f not in lastfiles]
            nnewfiles = len(newfiles)
            print(time.ctime()+' Found '+str(nnewfiles)+' new files: '+','.join(newfiles))
            for i in range(nnewfiles):
                base = os.path.basename(newfiles[i])
                outfile = datadir+'/red/'+base
                out = ccdproc(newfiles[i],outfile=outfile,compress=True)
        
        # Sleep for a while
        time.sleep(wait)

        # Last list of files
        lastfiles = files
