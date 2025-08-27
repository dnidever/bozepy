import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table,vstack
from dlnpyutils import utils as dln,coords,plotting as pl
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from . import utils

def loadgaia():
    """ Load the Gaia DR3 catalogs """
    datadir = utils.datadir()
    gaiafiles = glob(datadir+'/gaiadr3*fits.gz')
    gaiafiles.sort()
    gaia = []
    for i in range(len(gaiafiles)):
        gaia1 = Table.read(gaiafiles[i])
        gaia.append(gaia1)
    gaia = vstack(gaia)
    return gaia


def sbig_align(filename,ra=None,dec=None):
    """ Figure out the wcs alignment of an SBIG file using Gaia bright stars """

    # Load the gaia data
    gaia = loadgaia()
    gcoo = SkyCoord(gaia['ra'],gaia['dec'],unit='degree',frame='icrs')
    # Load the example wcs
    wcsfile = os.path.join(utils.datadir(),'sbig_wcs.fits')
    whead = fits.getheader(wcsfile)
    wcs = WCS(whead)

    # Find gaia sources nearby
    coo = SkyCoord(ra,dec,unit='degree',frame='icrs')
    dist = coo.separation(gcoo).degree
    gd, = np.where(dist < 1.0)
    gaia2 = gaia[gd]
    gcoo2 = gcoo[gd]
    
    # Convert to rough X/Y coordinates
    wcs.wcs.crval[:] = [ra,dec]
    x,y = wcs.world_to_pixel(gcoo2)
    

    # Make the plot
    pl.scatter(x,y,gaia2['gmag'],size=5,xtitle='X',
               ytitle='Y',colorlabel='G (mag)')

    # Detect sources in the image
    hdu = fits.open(filename)
    im = hdu[0].data
    backim = phot.background(im,boxsize=200)
    sim = im-backim
    tab = phot.detection(sim)
    phtab = phot.aperphot(im,tab)

    pl.scatter(ptab['xcenter'],ptab['ycenter'],ptab['mag'])
    
    import pdb; pdb.set_trace()
