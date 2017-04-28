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


def check_bad(dat, badfile, fieldname=None):
    conditions = get_conditions(dat, badfile, fieldname=fieldname)
    return conditions == 'bad'


def get_conditions(dat, badfile, fieldname=None):
    badlist = ascii.read(badfile, delimiter=',')
    if fieldname is None:
        conditions = numpy.zeros((len(dat), 5), dtype='a200')
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
                fieldname = f+'_'+field
                try:
                    val = dat[fieldname]
                except:
                    val = dat[fieldname.upper()]
                m = (val >= start) & (val <= end)
                conditions[m, filt2ind[f]] = condition
    return conditions
