# ;+
# ; NAME:
# ;   uptiles
# ;
# ; PURPOSE:
# ;   Update the DECam tile list to mark them as observed
# ;
# ; CALLING SEQUENCE:
# ;   update( expr, topdir=, tfile=, wtime=, /noloop, /debug )
# ;
# ; INPUTS:
# ;
# ; OPTIONAL INPUTS:
# ;   expr       - Match expression for files; default to 'DECam_????????.fits.fz'
# ;   topdir     - Directory with raw data files; default to $DECAM_DATA
# ;   tfile      - Tile file name; default to
# ;                $HOME/observing/obstatus/decam-tiles_obstatus.fits
# ;   wtime      - Wait time before looking for new files; default to 10 sec
# ;   noloop     - Set to only read the files in the specified directory once,
# ;                then exit without waiting for new files to be written
# ;   debug      - If set, then do not update the FITS file on disk,
# ;                only print to the terminal
# ;
# ; OUTPUTS:
# ;
# ; OPTIONAL OUTPUTS:
# ;
# ; COMMENTS:
# ;   This code can be run as an infinite loop while observing at the telescope
# ;   to continuously update the tile file while observing.  Exposures must
# ;   be OBTYPE='object' and EXPTIME > 25 sec.  An attempt is made to read
# ;   the TILEID from the OBJECT keyword in the header, where the assumed
# ;   format is "DECaLS_XXXXX_f", XXXXX is the tile number and f is the filter
# ;   name.  Otherwise, a positional match is made to the closest tile
# ;   if there is a tile within 1 arcmin.
# ;
# ;   This can also be run offsite to construct a file with all existing raw
# ;   or reduced exposures on disk.
# ;
# ;   The tile file (TFILE) is updated (overwritten) by setting G_DONE, R_DONE,
# ;   Z_DONE and G_DATE, R_DATE, Z_DATE and G_EXPNUM, R_EXPNUM, Z_EXPNUM
# ;   in that file for tiles that have been observed in the g, r, z filters
# ;   respectively.  The dates are reported as the local date for the start
# ;   of the night, which should agree with the NOAO Science Archive dates.
# ;
# ; EXAMPLES:
# ;
# ; BUGS:
# ;
# ; DATA FILES:
# ;   $TOPDIR/DECam_????????.fits.fz
# ;   $TFILE
# ;
# ; REVISION HISTORY:
# ;   26-Mar-2015  Written by D. Schlegel, LBL
# ;   07-Mar-2016  Initial port to python, EFS
# ;-
# ;------------------------------------------------------------------------------
# ; Get the sorted list of all raw data files; the last files are assumed
# ; to be the most recent.

import numpy
import time
import os
import pyfits
import fnmatch
import dateutil.parser
import datetime

# stolen from internet, Simon Brunning
def locate(pattern, root=None):
    '''Locate all files matching supplied filename pattern in and below
    supplied root directory.'''
    if root is None:
        root = os.curdir
    for path, dirs, files in os.walk(os.path.abspath(root)):
        files2 = [os.path.join(os.path.relpath(path, start=root), f)
                  for f in files]
        for filename in fnmatch.filter(files2, pattern):
            yield os.path.join(os.path.abspath(root), filename)

def search(expr, topdir=None):
    files = numpy.array(list(locate(expr, root=topdir)))
    s = numpy.argsort(files)
    files = files[s]
    return files

def gc_dist(lon1, lat1, lon2, lat2):
    from numpy import sin, cos, arcsin, sqrt

    lon1 = numpy.radians(lon1); lat1 = numpy.radians(lat1)
    lon2 = numpy.radians(lon2); lat2 = numpy.radians(lat2)

    return numpy.degrees(2*arcsin(sqrt( (sin((lat1-lat2)*0.5))**2 + cos(lat1)*cos(lat2)*(sin((lon1-lon2)*0.5))**2 )));

def str2dec(string):
    string = string.strip()
    sign = 1
    if string[0] == '-':
        sign = -1
        string = string[1:]
    elif string[0] == '+':
        sign = 1
        string = string[1:]
    h, m, s = [float(s) for s in string.split(':')]
    return sign * (h + (m / 60.) + (s / 60. / 60.))

def process(file, tdata, minexptime=25):
    print('Processing file %s' % file)
    qdone = False
    for i in xrange(5):
        try:
            hdr = pyfits.getheader(file)
        except:
            time.sleep(2)
        else:
            qdone = True
    if not qdone:
        print('Could not read file %s' % file)
        return
    obstype = hdr['OBSTYPE']
    exptime = hdr['EXPTIME']
    expnum = hdr['EXPNUM']
    ra = str2dec(hdr['ra'])*15.
    dec = str2dec(hdr['dec'])
    dateobstime = dateutil.parser.parse(hdr['DATE-OBS'])
    dateobstime = dateobstime + datetime.timedelta(hours=-18)
    dateobs = dateobstime.isoformat()[0:10]

    obj = hdr['OBJECT']
    filt = hdr['filter'][0:1].lower()
    obj3 = obj.split('_')
    if (obstype != 'object') or (exptime <= minexptime):
        print obstype, exptime
        tileid = 0
    else:
        if (obj3[0] == 'DECaLS') and (len(obj3) == 3):
            tileid = int(obj3[1])
        else:
            dd = gc_dist(tdata['ra'], tdata['dec'], ra, dec)
            imin = numpy.argmin(dd)
            dmin = dd[imin]
            if dmin < 1./60.:
                tileid = int(tdata[imin]['tileid'])
            else:
                tileid = 0

    if tileid <= 0:
        print('Invalid TILEID = %d' % tileid)
        return
    ind = numpy.flatnonzero(tdata['tileid'] == tileid)
    if len(ind) == 0:
        print('Invalid TILEID = %d' % tileid)
        return
    if filt not in 'grz':
        print('Invalid filter %s' % filt)
        return
    print('Adding TILEID=%d FILTER=%s' % (tileid, filt))
    tdata[filt+'_done'][ind] = 1
    tdata[filt+'_date'][ind] = dateobs
    tdata[filt+'_expnum'][ind] = expnum

def write(tdata, tfile):
    print('Writing file %s' % tfile)
    pyfits.writeto(tfile, tdata, clobber=True)

def update(expr='*/*.fits.fz', topdir=None, tfile=None, wtime=None, 
           noloop=False, debug=False):
    if topdir is None:
        topdir = os.environ.get('DECAM_DATA', '')
    if topdir == '':
        raise ValueError('topdir keyword or $DECAM_DATA env must be set!')
    if tfile is None:
        tfile = os.path.join(os.environ['HOME'], 'observing', 'obstatus', 
                             'decam-tiles_obstatus.fits')
    if wtime is None:
        wtime = 10
    if wtime < 1:
        wtime = 1

    tdata = pyfits.getdata(tfile, 1)

    nfile = 0
    while True:
        print('uptiles checking for new files...')
        files = search(expr, topdir=topdir)
        if len(files) != nfile:
# if I really know the order and what it means, better to just check if the
# last file is the same as it used to be?  Currently seems fragile to file 
# deletion, weird files, ...
            for i in xrange(nfile, len(files)):
                process(files[i], tdata, minexptime=25)
            if debug == 0:
                write(tdata, tfile)
            nfile = len(files)
        if noloop:
            break
        else:
            time.sleep(wtime)
