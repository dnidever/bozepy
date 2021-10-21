***************
Getting Started
***************



ccdproc
=======

:mod:`~bozepy.ccdproc` is a module that allows for CCD image calibration and reduction.  The main functions are:

 - :func:`~bozepy.ccdproc.ccdproc`: CCD calibration.
 - :func:`~bozepy.ccdproc.ccdlist`: Get information on FITS images.
 - :func:`~bozepy.ccdproc.masterbias`: Make a master bias frame.
 - :func:`~bozepy.ccdproc.masterdark`: Make a master dark frame.
 - :func:`~bozepy.ccdproc.masterflat`: Make a master flat frame.
 - :func:`~bozepy.ccdproc.makebpm`: Make a bad pixel mask image.
 - :func:`~bozepy.ccdproc.autored`: Automatically pick up new images and reduce them.
 - :func:`~bozepy.ccdproc.redrun`: Automatically reduce a set of images. (in development)


phot
====

:mod:`~bozepy.phot` is a module that has a number of photometry routines.

 - :func:`~bozepy.phot.background`: Estimate the smooth background in an image.
 - :func:`~bozepy.phot.detection`: Detect sources in an image.
 - :func:`~bozepy.phot.daodetect`: DAO source detection.
 - :func:`~bozepy.phot.aperphot`: Perform circular aperture photometry on image.
 - :func:`~bozepy.phot.morphology`: Calculate centroids and morphology of sources.
