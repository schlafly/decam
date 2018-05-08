#!/usr/bin/env python
# 
# plot_coverage.py
# Plots the survey footprint, showing which pointings have been observed and
# which pointings are planned to be observed.
#
# Gregory M. Green, 2017
#

from __future__ import print_function, division

from argparse import ArgumentParser
from glob import glob
import os
import textwrap
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import astropy.io.fits as fits
from astropy.coordinates import SkyCoord

import badweather


script_dir = os.path.dirname(os.path.realpath(__file__))


def plot_circles(ax, x, y, radius, **kwargs):
    if not hasattr(radius, '__len__'):
        radius = np.full_like(x, radius)

    for xx,yy,rr in zip(x,y,radius):
        circle = patches.Circle((xx,yy), radius=rr, **kwargs)
        ax.add_patch(circle)


def main():
    parser = ArgumentParser(
        description='Plot sky coverage.',
        add_help=True)
    parser.add_argument(
        'tilefile',
        metavar='TILE_FILENAME',
        type=str,
        help='Filename of the tile file.')
    parser.add_argument(
        '--output', '-o',
        metavar='PLOT_FILENAME',
        type=str,
        required=True,
        help='Output filename of plot.')
    parser.add_argument(
        '--plan', '-p',
        metavar='PLAN_FILENAME',
        type=str,
        nargs='+',
        help='Filenames of DECam plan JSON(s).')
    parser.add_argument(
        '--pass',
        metavar='NUMBER',
        type=int,
        nargs='+',
        dest='pass_num',
        help='Pass(es) to plot (defaults to all).')
    parser.add_argument(
        '--decaps-only',
        action='store_true',
        help='Plot only tiles in nominal DECaPS footprint.')
    parser.add_argument(
        '--decaps-extended-only',
        action='store_true',
        help='Plot only tiles in extended DECaPS footprint.')
    parser.add_argument(
        '--weather',
        type=str,
        default=os.path.join(script_dir, '..', 'data', 'badweather.txt'),
        help='File containing weather information.')
    parser.add_argument(
        '--reject-conditions',
        type=str,
        nargs='+',
        default=('bad', 'marginal'),
        help='Reject exposures taken under the given weather conditions.')
    parser.add_argument(
        '--bad-exposures',
        type=str,
        default=os.path.join(script_dir, '..', 'data', 'badexp.txt'),
        help='File containing indices of bad exposures.')
    args = parser.parse_args()

    # Expand plan filenames
    plan_fname = []
    if args.plan:
        for fn_pattern in args.plan:
            plan_fname += glob(fn_pattern)

    bands = 'grizY'

    # Load plans
    plan_tiles = {b:[] for b in bands}

    for fn in plan_fname:
        with open(fn, 'r') as f:
            data = json.load(f)
        try:
            for row in data:
                b = row['filter']
                plan_tiles[b].append(int(row['object'].split('_')[1]))
        except ValueError as err:
            print('In {:s}: "object" entry does not conform to pattern "DECaPS_[tile]_[band]"'.format(fn))
            raise err
        except KeyError as err:
            print('In {:s}: missing key'.format(fn))
            raise err

    # Load information from tile file
    n_bands = len(bands)

    with fits.open(args.tilefile) as hdulist:
        data = hdulist[1].data[:]
        
    badtiles = badweather.check_badexp(data, badexpfn=args.bad_exposures)
    
    if args.weather is not None:
        badtiles = badtiles | badweather.check_bad(data, args.weather, reject=args.reject_conditions)
    
    for k,filt in enumerate('grizy'):
        data[filt+'_done'] = (
            data[filt+'_done'] & (badtiles[:, k] == 0))
    
    # print('\n'.join(textwrap.wrap(str(hdulist[1].header), 80)))

    n_decaps_tiles = np.sum(data['in_decaps'] == 1)

    tile_id = data['tileid'][:]
    tile_sort_idx = np.argsort(tile_id)

    idx_footprint = np.zeros((len(data)), dtype='bool')
    idx_planned = np.zeros((len(data),n_bands), dtype='bool')
    idx_observed = np.empty((len(data),n_bands), dtype='bool')
    pct_observed = np.empty(len(bands), dtype='f8')

    if args.pass_num:
        for p in args.pass_num:
            idx_footprint |= (data['pass'] == p)
    else:
        idx_footprint[:] = 1

    if args.decaps_only:
        # print(np.unique(data['in_decaps']))
        idx_footprint &= (data['in_decaps'] == 3)

    if args.decaps_extended_only:
        idx_footprint &= (data['in_decaps'] != 0)

    for i,b in enumerate(bands):
        idx_observed[:,i] = data['{}_done'.format(b)]

        k_insert = np.searchsorted(
            tile_id,
            plan_tiles[b],
            sorter=tile_sort_idx)
        k_insert[k_insert == len(tile_id)] = -1
        idx_match = (tile_id[k_insert] == plan_tiles[b])
        idx_planned[k_insert[idx_match],i] = 1

        pct_observed[i] = 100. * np.count_nonzero(idx_observed[:,i] & idx_footprint) / np.count_nonzero(idx_footprint)
        print('{}: {:.1f}% observed'.format(b, pct_observed[i]))

    coords = SkyCoord(
        data['ra'],
        data['dec'],
        frame='icrs',
        unit='deg')

    # Set up the figure
    fig = plt.figure(figsize=(8,2*n_bands), dpi=100)

    g = coords.transform_to('galactic')
    gal_l = g.l.wrap_at('120d').deg
    gal_b = g.b.deg

    l_footprint = gal_l[idx_footprint]
    b_footprint = gal_b[idx_footprint]

    xlim = [np.max(l_footprint)+4., np.min(l_footprint)-4.]
    ylim = [np.min(b_footprint)-4., np.max(b_footprint)+4.]

    # Determine plotting opacity for observed pointings
    if args.pass_num:
        alpha_obs = 1. / len(args.pass_num)
    else:
        alpha_obs = 0.33

    # Plot each band
    for i,b in enumerate(bands):
        ax = fig.add_subplot(n_bands,1,1+i)

        # Plot unobserved pointings
        idx = idx_footprint & (~idx_observed[:,i])
        plot_circles(
            ax,
            gal_l[idx],
            gal_b[idx],
            1.,
            edgecolor='k',
            facecolor='none',
            alpha=0.04)

        # Plot observed pointings
        idx = idx_footprint & idx_observed[:,i]
        plot_circles(
            ax,
            gal_l[idx],
            gal_b[idx],
            1.,
            edgecolor='none',
            facecolor='k',
            alpha=alpha_obs)

        # Plot planned pointings
        idx = idx_footprint & idx_planned[:,i]
        plot_circles(
            ax,
            gal_l[idx],
            gal_b[idx],
            1.,
            edgecolor='none',
            facecolor='green',
            alpha=0.75*alpha_obs)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_ylabel(r'${}$'.format(b), fontsize=22, rotation=0, labelpad=15)

    if args.pass_num:
        title = 'Pass ' + ' + '.join([str(i) for i in args.pass_num])
    else:
        title = 'All Passes'
    fig.suptitle(r'$\mathrm{{ {} }}$'.format(title.replace(' ', '\ ')), fontsize=20)

    fig.savefig(args.output, bbox_inches='tight', transparent=False, dpi=100)

    return 0


if __name__ == '__main__':
    main()
