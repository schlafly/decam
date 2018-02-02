#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def get_tiling(lon0, lat0, dlon, dlat):
    rot = hp.rotator.Rotator(rot=(0., -lat0, -lon0),
                             eulertype='Y',
                             deg=True).mat
    x = hp.dir2vec(dlon, dlat, lonlat=True)
    if hasattr(dlon, '__len__') and (len(x.shape) == 1):
        x.shape = (3,1)
    y = np.einsum('ij,jn', rot, x)
    lonlat = hp.vec2dir(y, lonlat=True)
    return lonlat


def main():
    pix_scale = 0.27 / 3600.
    ccd_scale = (4096.*pix_scale, 2048.*pix_scale)

    dlon = np.array([0., 0.5*ccd_scale[0], -0.5*ccd_scale[0]])
    dlat = np.array([-0.5*ccd_scale[1], 0., 0.])

    dlon += 0.1 * ccd_scale[0] * np.random.random(3)
    dlat += 0.1 * ccd_scale[1] * np.random.random(3)

    print(dlon)
    print(dlat)

    lonlat = get_tiling(10., 85., dlon, dlat)

    print(lonlat)
    print(lonlat.shape)

    return 0


if __name__ == '__main__':
    main()
