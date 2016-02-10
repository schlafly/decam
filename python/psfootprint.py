import pyfits
from lsd.builtins.misc import galequ, equgal
import pdb
from matplotlib.mlab import rec_append_fields
import numpy

def extend_footprint_to_matches(tiles, infootprint):
    # collect any pointings where not all three landed within the footprint
    npointings = len(tiles) / 3
    tileno = tiles['tileid'] % npointings
    tilenoinpointing = tileno[infootprint]
    for tileno0 in tilenoinpointing:
        infootprint = infootprint | (tileno == tileno0) 
        # O(N^2), easy to speed up if necessary
    return infootprint
    

def make_footprint(tilefile='/n/home13/schlafly/decals/observing/trunk/obstatus/decam-tiles_obstatus.fits'):
    lbound = [260, 320]
    bbound = [-4, 4]
    tiles = pyfits.getdata(tilefile)
    lt, bt = equgal(tiles['ra'], tiles['dec'])
    mpilot = ((bt > bbound[0]) & (bt < bbound[1]) & 
              (lt > lbound[0]) & (lt < lbound[1]))
    mpilot = extend_footprint_to_matches(tiles, mpilot)
    zeros = numpy.zeros(len(tiles), dtype='i4')
    tiles = rec_append_fields(tiles, ['i_done', 'y_done', 'in_decaps'], 
                              [zeros.copy() for i in xrange(3)])
    tiles['in_decaps'][mpilot] |= 2**0
    lbound = [240, 360]
    bbound = [-10, 10]
    mall = ((bt > bbound[0]) & (bt < bbound[1]) & 
            (lt > lbound[0]) & (lt < lbound[1]))
    tiles['in_decaps'][mall] |= 2**1
    tiles.dtype.names = [n.lower() for n in tiles.dtype.names]
    return tiles
    
    
