"""Simple tool to mark times when the weather was misbehaving.

We mark misbehaving weather via a text file, with rows:

563628, 563667
or
2016-08-10T07:21, 2016-08-10T09:41

The first format marks exposure IDs, inclusive, that were taken in bad
conditions, and the second format marks times in UT.
"""

import numpy
import pdb
from astropy.io import ascii
from astropy.time import Time

filt2ind = {'g': 0, 'r': 1, 'i': 2, 'z': 3, 'y': 4}


def check_badexp(tiles, badexpfn=None):
    if badexpfn is None:
        import os
        badexpfn = os.path.join(os.environ['DECAM_DIR'], 'data', 'badexp.txt')
    from astropy.io import ascii
    badexp = ascii.read(badexpfn, names=['expnum']).as_array()['expnum']
    res = numpy.zeros((len(tiles), 5), dtype='bool')
    for badexpnum in badexp:
        for i, f in enumerate('grizy'):
            m = numpy.flatnonzero((tiles[f+'_expnum'] == badexpnum) & 
                               (tiles[f+'_done'] == 1))
            if len(m) > 0:
                res[m, i] = 1
    return res


def check_bad(dat, badfile, fieldname=None, reject=('bad', 'marginal')):
    conditions = get_conditions(dat, badfile, fieldname=fieldname)
    bad = numpy.zeros(conditions.shape, dtype=numpy.bool)
    for cond in reject:
        bad |= (conditions == cond)
    return bad
    #return (conditions == 'bad') | (conditions == 'marginal')


def get_conditions(dat, badfile, fieldname=None):
    badlist = ascii.read(badfile, delimiter=',')
    if fieldname is None:
        conditions = numpy.zeros((len(dat), 5), dtype='a20')
    else:
        conditions = numpy.zeros(len(dat), dtype='a20')
    for row in badlist:
        field = None
        try:
            start = int(row['start'])
            end = int(row['end'])
            field = 'expnum'
        except ValueError:
            pass
	try:
            start = Time(row['start'], format='isot', scale='utc')
            end = Time(row['end'], format='isot', scale='utc')
            start = start.mjd
            end = end.mjd
            field = 'mjd_obs'
        except ValueError:
            pass

	if field is None:
            raise ValueError('row format not understood: %s' % row)
        if start > end:
            raise ValueError('file not understood, start > end')

        condition = row['type'].strip()
        if fieldname is not None:
            val = dat[fieldname]
            m = (val >= start) & (val <= end)
            conditions[m] = condition
        else:
            for f in 'grizy':
                fn = f+'_'+field
                try:
                    val = dat[fn]
                except:
                    val = dat[fn.upper()]
                m = (val >= start) & (val <= end)
                conditions[m, filt2ind[f]] = condition
    return conditions
