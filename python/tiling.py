#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import textwrap
from argparse import ArgumentParser
from progressbar import ProgressBar


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
    parser = ArgumentParser(
        description='Modify the dithering scheme of a tile file.',
        add_help=True)
    parser.add_argument(
        'infname', metavar='input-tile-file.fits', type=str,
        help='Input tile file.')
    parser.add_argument(
        'outfname', metavar='output-tile-file.fits', type=str,
        help='Output tile file.')
    parser.add_argument(
        '--delta-RA', '-dRA', type=float, nargs=3, required=True,
        help='RA offset for each dither (in pixels).')
    parser.add_argument(
        '--delta-Dec', '-dDec', type=float, nargs=3, required=True,
        help='Declination offset for each dither (in pixels).')
    parser.add_argument(
        '--rotate', '-r', type=float,
        help='Rotate all dithers by given amount (in deg).')
    parser.add_argument(
        '--scale', '-s', type=float,
        help='Scale all dithers by given amount.')
    parser.add_argument(
        '--jitter', '-j', type=float,
        help='Add random offsets to each dither on this scale (in pix).')
    #parser.add_argument(
    #    '--alter-observed', action='store_true',
    #    help='Change the dithering of already-observed tiles.')
    args = parser.parse_args()
    
    dlon = np.array(args.delta_RA)
    dlat = np.array(args.delta_Dec)
    
    # Rotate dithers?
    if args.rotate is not None:
        theta = np.radians(args.rotate)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        x = np.vstack([dlon, dlat])
        dlon, dlat = np.einsum('ij,in', R, x)
        dlon -= np.mean(dlon)
        dlat -= np.mean(dlat)
    
    # Scale dithers?
    if args.scale is not None:
        dlon *= args.scale
        dlat *= args.scale
    
    print('')
    print('Dither (in pixels):')
    print('  dlon       dlat  ')
    print('-------------------')
    for dx,dy in zip(dlon,dlat):
        print('{: >8.1f}  {: >8.1f}'.format(dx,dy))
    print('')
    print('Jitter: {:.1f} px'.format(
        0 if args.jitter is None else args.jitter))
    print('')
    
    # Pixel scale, in deg
    pix_scale = 0.27 / 3600.
    
    dlon *= pix_scale
    dlat *= pix_scale
    
    if args.jitter is not None:
        noise = args.jitter * pix_scale
        np.random.seed(2)
    
    with fits.open(args.infname, mode='readonly') as f:
        data = f[1].data
        
        h = str(f[1].header)
        print('\n'.join([h[i:i+80] for i in range(0,len(h),80)]))
        
        # Each "group" consists of 3 passes in one location
        n_groups = len(data) // 3
        
        bar = ProgressBar(max_value=n_groups)
        
        # Redo dithering in each group
        for k in range(n_groups):
            idx = [k, k+n_groups, k+2*n_groups]
            
            base = data[idx[0]]
            
            # Add in random jitter?
            if args.jitter is not None:
                lon_noise = noise * np.random.uniform(-1., 1., size=dlon.shape)
                lat_noise = noise * np.random.uniform(-1., 1., size=dlat.shape)
                lon,lat = get_tiling(
                    base['ra'], base['dec'],
                    dlon + lon_noise,
                    dlat + lat_noise)
            else:
                lon,lat = get_tiling(base['ra'], base['dec'], dlon, dlat)
            
            data['ra'][idx] = np.mod(lon, 360.)
            data['dec'][idx] = lat
            
            bar.update(k)
        
        # Write to new output file
        hdulist = fits.HDUList([
            f[0],
            fits.BinTableHDU(data=data, header=f[1].header)
        ])
        
        hdulist.writeto(args.outfname, clobber=True)
    
    return 0


if __name__ == '__main__':
    main()
