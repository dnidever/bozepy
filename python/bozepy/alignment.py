import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table,vstack
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
    # Load the example wcs
    wcsfile = os.path.join(utils.datadir(),'sbig_wcs.fits')
    whead = fits.getheader(wcsfile)
    wcs = WCS(whead)

    import pdb; pdb.set_trace()
