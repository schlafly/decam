"""
Simplified version of DECaLS nightlystrategy.py

strategy: observe the pointing that is getting worst the fastest
  prefer nearby exposures over just getting the single pointing getting worse
  the fastest.
"""

from astropy.io import fits
import ephem
import numpy as np
import json

import pdb

from collections import OrderedDict

# Global constants
R_earth = 6378.1e3  # in meters
# conversion factor s_to_days*secs = days
s_to_days = (1.0/(60.0*60.0*24.0))
days_to_s = (60.0*60.0*24.0)

decam = ephem.Observer()
decam.lon = '-70.806525'
decam.lat = '-30.169661'
decam.elev = 2207.0  # meters
decam.temp = 10.0  # deg celsius; average temp for August
decam.pressure = 780.0  # mbar
# decam.horizon = -np.sqrt(2.0*decam.elev/R_earth)
decam.horizon = -ephem.degrees('12:00:00.0')

exp_time_filters = {'g': 96, 'r': 30, 'i': 30, 'z': 30, 'Y': 30}
filter_symbols = {'g': 's', 'r': 'p', 'i': '<', 'z': 'o', 'Y': '*'}

sun = ephem.Sun()
moon = ephem.Moon()

overheads = 30.


def alt2airmass(alt):
    '''
    Calculate the airmass at the given altitude (in radians).

    Uses the Hardie (1962) interpolation function.
    '''

    if isinstance(alt, list):
        alt[alt < 0.07] = 0.07
    else:
        alt = max(alt, 0.07)

    secz = 1.0 / np.sin(alt)
    seczm1 = secz - 1.0

    airm = (secz - 0.0018167*seczm1 - 0.002875*seczm1**2 -
            0.0008083*seczm1**3)

    return airm


def radec2airmass(obs, ra, dec):
    b = ephem.FixedBody()
    b._ra = np.radians(ra)
    b._dec = np.radians(dec)
    b.compute(obs)
    return alt2airmass(b.alt)


def radec_obj_dist(obs, ra, dec, obj):
    b = ephem.FixedBody()
    b._ra = np.radians(ra)
    b._dec = np.radians(dec)
    b.compute(obs)
    obj.compute(obs)
    return np.degrees(ephem.separation(b, obj))


def radec_report(obs, ra, dec):
    am = radec2airmass(obs, ra, dec)
    ms = radec_obj_dist(obs, ra, dec, ephem.Moon())
    msg = (
        'Airmass: {:.2f}\n'
        'MoonSep: {:.1f} deg'
    ).format(am, ms)
    return msg


def night_start_end(obs):
    obs = obs.copy()
    t_start = obs.next_setting(ephem.Sun())
    obs.date = t_start
    t_end = obs.next_rising(ephem.Sun())
    return t_start, t_end


#####################################################
# Misc.
#####################################################

def ConvertRA(raval):
    hours = np.zeros_like(raval)
    minutes = np.zeros_like(raval)
    seconds = np.zeros_like(raval)

    hours = (raval/360.0)*24.0
    minutes = (hours-np.floor(hours))*60.0
    seconds = (minutes-np.floor(minutes))*60.0

    stringra = []
    for k in range(0, raval.size):
        # print hours[k],minutes[k], seconds[k]
        stringra.append("%02d:%02d:%04.1f" %
                        (hours[k], minutes[k], seconds[k]))

    stringra = np.array(stringra)
    return stringra


#######################################################
def ConvertDec(decval):
    sdd = np.zeros_like(decval)
    minutes = np.zeros_like(decval)
    seconds = np.zeros_like(decval)

    sdd = decval
    pos_sdd = np.fabs(sdd)
    minutes = (pos_sdd-np.floor(pos_sdd))*60.0
    seconds = (minutes-np.floor(minutes))*60.0

    stringdec = []
    for k in range(0, decval.size):
        # print sdd[k],minutes[k], seconds[k]
        stringdec.append("%02d:%02d:%02d" % (sdd[k], minutes[k], seconds[k]))

    stringdec = np.array(stringdec)
    return stringdec
#####################################################
#####################################################


def gc_dist(lon1, lat1, lon2, lat2):
    '''
    Great-circle distance between two points on a sphere.
    Inputs:
        lon1  longitude (in degrees) of the first point
        lat1  latitude (in degrees) of the first point
        lon2  longitude (in degrees) of the second point
        lat2  latitude (in degrees) of the second point
    Output:
        dist  angular distance (in degrees) between the two points.
    '''
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    return np.degrees(
        2. * np.arcsin(
            np.sqrt(
                np.sin(0.5*(lat1-lat2))**2 +
                np.cos(lat1) * np.cos(lat2) * np.sin(0.5*(lon1-lon2))**2
            )
        )
    )


#####################################################
def WriteJSON(pl, fname, chunks=1):
    # Convert the plan into a list of dictionaries
    n_exposures = len(pl['RA'])

    exposure_list = ([
        {
            'expType': 'object',
            'object': 'DECaPS_{:d}_{:s}'.format(pl['TILEID'][k],
                                                pl['filter'][k]),
            'expTime': pl['exp_time'][k],
            'filter': pl['filter'][k],
            'RA': np.mod(pl['RA'][k], 360.),
            # '{:.3f}'.format(np.mod(pl['RA'][k], 360.)),
            'dec': pl['DEC'][k]
            # '{:.3f}'.format(pl['DEC'][k])
        }
        for k in range(n_exposures)
    ])

    # Write to JSON
    chunk_size = int(np.ceil(float(n_exposures) / float(chunks)))

    for c in range(chunks):
        i0 = c * chunk_size
        i1 = (c+1) * chunk_size
        f = open('{:s}_{:02d}.json'.format(fname, c), 'w')
        json.dump(exposure_list[i0:i1], f, indent=2, separators=(',', ': '))
        f.close()


def gen_decam_json(ra, dec, filters, obj_name, fname, dither=False):
    exposures = []

    if dither:
        dither_scale = 0.262 * 4096 / 3600.
        d_ra = (np.array([0.0, -0.5, 0.5]) * dither_scale /
                np.cos(np.radians(dec)))
        d_dec = np.array([0.5, -0.5, -0.5]) * 0.5 * dither_scale
    else:
        d_ra = [0.]
        d_dec = [0.]

    for filt in filters:
        for da, dd in zip(d_ra, d_dec):
            exposures.append({
                'expType': 'object',
                'object': 'DECaPS_{:s}_{:s}'.format(obj_name, filt),
                'expTime': 30.,
                'filter': filt,
                'RA': np.mod(ra+da, 360.),
                'dec': dec+dd
            })

    f = open(fname, 'w')
    json.dump(exposures, f, indent=2, separators=(',', ': '))
    f.close()


def equgal(ra, dec):
    coord = [ephem.Galactic(ephem.Equatorial(ra0*np.pi/180., dec0*np.pi/180.))
             for ra0, dec0 in zip(ra, dec)]
    l = np.array([coord0.lon*180./np.pi for coord0 in coord])
    b = np.array([coord0.lat*180./np.pi for coord0 in coord])
    return l, b


def readTilesTable(filename, expand_footprint=False, rdbounds=None,
                   lbbounds=None, skypass=-1):
    tiles_in = fits.getdata(filename, 1)

    tiles = OrderedDict()
    # Check that required columns exist
    for col in ['TILEID', 'PASS', 'IN_SDSS', 'IN_DES', 'IN_DESI', 'IN_DECAPS',
                'G_DONE', 'R_DONE', 'I_DONE', 'Z_DONE', 'Y_DONE']:
        tiles[col] = tiles_in[col].astype(int)
    for col in ['RA', 'DEC', 'EBV_MED']:
        tiles[col] = tiles_in[col].astype(float)

    # Cut to tiles of interest:
    if expand_footprint:
        I = (tiles['IN_DECAPS'] & 2**1) != 0
    else:
        I = (tiles['IN_DECAPS'] & 2**0) != 0

    if skypass > 0:
        I = I & (tiles['PASS'] == skypass)

    if rdbounds is not None:
        I = (I & (tiles['RA'] > rdbounds[0]) & (tiles['RA'] <= rdbounds[1]) &
             (tiles['DEC'] > rdbounds[2]) & (tiles['DEC'] <= rdbounds[3]))

    if lbbounds is not None:
        lt, bt = equgal(tiles['RA'], tiles['DEC'])
        lt2 = ((lt + 180.) % 360.) - 180.
        I = (I &
             ((((lt > lbbounds[0]) & (lt <= lbbounds[1])) |
              ((lt2 > lbbounds[0]) & (lt2 <= lbbounds[1]))) &
              (bt > lbbounds[2]) & (bt <= lbbounds[3])))

    survey = OrderedDict([(k, v[I]) for k, v in tiles.items()])

    # H:M:S and D:M:S strings
    survey['RA_STR'] = ConvertRA(survey['RA'])
    survey['DEC_STR'] = ConvertDec(survey['DEC'])

    return tiles, survey


def GetNightlyStrategy(obs, survey_centers, filters, nightfrac=1.,
                       minmoonsep=40.):
    """date: UT; if time is not set, the next setting of the sun following start
    of that date is the start of the plan; awkward when the night starts just
    before midnight UT, as it does in March in Chile!
    """

    # tonightsplan = OrderedDict()
    tonightsplan = {}
    orig_keys = survey_centers.keys()
    for key in orig_keys:
        tonightsplan[key] = []

    tonightsplan['airmass'] = []
    tonightsplan['approx_time'] = []
    tonightsplan['approx_datetime'] = []
    tonightsplan['moon_sep'] = []
    tonightsplan['moon_alt'] = []
    tonightsplan['filter'] = []
    tonightsplan['exp_time'] = []
    tonightsplan['lst'] = []

    # Get start and end time of night
    sn = obs.date
    en = obs.next_rising(ephem.Sun())
    lon = (en-sn) * days_to_s * nightfrac

    # Make sure the Sun isn't up
    sun.compute(obs)
    moon.compute(obs)
    if sun.alt > 0:
        print 'WARNING: sun is up?!'

    # Report night start/end times
    print 'Date: {}'.format(obs.date)
    print 'Length of night: {} s'.format(lon)
    print 'Start time of plan (UT): {}'.format(sn)
    print 'End time of night (UT): {}'.format(en)
    print 'Plan length: {} hours'.format(lon / 60. / 60.)

    for f in 'grizY':
        col = 'used_tile_{:s}'.format(f)
        survey_centers[col] = (
            survey_centers['{:s}_DONE'.format(f.capitalize())].copy())

    time_elapsed = 0.0
    filterorder = 1

    while time_elapsed < lon:
        start_obsdate = sn + time_elapsed*s_to_days
        obs.date = start_obsdate

        if obs.date > en:
            break

        sun.compute(obs)
        moon.compute(obs)

        # compute derivative of airmass for each exposure
        airmass = np.zeros((survey_centers['RA'].size, 2), dtype='f8')
        moonsep = np.zeros((survey_centers['RA'].size), dtype='f8')

        for j in range(survey_centers['RA'].size):
            tile_str = ','.join([
                str(survey_centers['TILEID'][j]),
                'f',
                survey_centers['RA_STR'][j],
                survey_centers['DEC_STR'][j],
                '20'
            ])
            this_tile = ephem.readdb(tile_str)
            moon.compute(obs)
            this_tile.compute(obs)
            moonsep[j] = ephem.separation((this_tile.az, this_tile.alt),
                                          (moon.az, moon.alt))*180./np.pi
            for k, dt in enumerate((0., s_to_days)):
                obs.date = start_obsdate + dt
                this_tile.compute(obs)
                airmass[j, k] = alt2airmass(float(this_tile.alt))

        obs.date = start_obsdate  # reset date

        # Rate of change in airmass^3 (per second)
        dairmass = airmass[:, 1]**3. - airmass[:, 0]**3.

        # Exclude tiles with terrible airmass
        exclude = (airmass[:, 0] > 5) | (moonsep < minmoonsep)

        # Exclude tiles that have been observed before
        filters_done = np.ones(len(survey_centers['RA']), dtype='bool')

        for f in filters:
            filters_done = (filters_done &
                            survey_centers['used_tile_{:s}'.format(f)])

        exclude = exclude | filters_done

        # Bail if there's nothing left to observe
        if np.all(exclude):
            print 'Ran out of tiles to observe before night was done!'
            print 'Minutes left in night: {:5.1f}'.format(
                (lon-time_elapsed)/60.)
            break

        # Determine slew time for each possible exposure
        if len(tonightsplan['RA']) > 1:
            slew = np.clip(
                gc_dist(
                    tonightsplan['RA'][-1], tonightsplan['DEC'][-1],
                    survey_centers['RA'], survey_centers['DEC']
                ) - 2.,
                0.,
                np.inf
            )
        else:
            slew = 0

        # Select tile based on airmass rate of change and slew time
        nexttile = np.argmax(dairmass - 0.0001*slew - 1.e10*exclude)

        delta_t, n_exp = pointing_plan(
            tonightsplan,
            orig_keys,
            survey_centers,
            nexttile,
            filters[::filterorder],
            obs
        )

        time_elapsed += delta_t

        filterorder = -filterorder

        if len(tonightsplan['RA']) > n_exp:
            slew = gc_dist(
                tonightsplan['RA'][-1], tonightsplan['DEC'][-1],
                tonightsplan['RA'][-n_exp-1], tonightsplan['DEC'][-n_exp-1]
            )
            slewtime = 3.0 * max(slew-2.0, 0.0)
            time_elapsed += slewtime
            if slewtime > 0:
                print 'time spent slewing: {:.1f}'.format(slewtime)

    numleft = np.sum(exclude == 0)
    print 'Plan complete, {:d} observations, {:d} remaining.'.format(
        len(tonightsplan['RA']), numleft)

    keys = tonightsplan.keys()
    return np.rec.fromarrays([tonightsplan[k] for k in keys], names=keys)


def pointing_plan(tonightsplan, orig_keys, survey_centers, nexttile, filters,
                  obs):
    time_elapsed = 0
    n_exp = 0

    for f in filters:
        if survey_centers['used_tile_{:s}'.format(f)][nexttile] == 1:
            continue

        n_exp += 1

        survey_centers['used_tile_{:s}'.format(f)][nexttile] = 1

        tile_str = ','.join([
            str(survey_centers['TILEID'][nexttile]),
            'f',
            survey_centers['RA_STR'][nexttile],
            survey_centers['DEC_STR'][nexttile],
            '20'
        ])
        this_tile = ephem.readdb(tile_str)
        this_tile.compute(obs)

        # Compute airmass
        airm = alt2airmass(float(this_tile.alt))

        # Compute moon separation
        moon.compute(obs)
        moon_dist = ephem.separation(
            (this_tile.az, this_tile.alt),
            (moon.az, moon.alt)
        )
        moon_alt = np.degrees(moon.alt)

        # Add this exposure to tonight's plan
        for key in orig_keys:
            tonightsplan[key].append(survey_centers[key][nexttile])

        tonightsplan['exp_time'].append(exp_time_filters[f])
        tonightsplan['approx_datetime'].append(obs.date)
        tonightsplan['airmass'].append(airm)
        tonightsplan['approx_time'].append(obs.date)
        tonightsplan['filter'].append(f)
        tonightsplan['moon_sep'].append(moon_dist)
        tonightsplan['moon_alt'].append(moon_alt)
        tonightsplan['lst'].append(np.degrees(obs.sidereal_time()))

        delta_t = exp_time_filters[f] + overheads
        time_elapsed += delta_t
        obs.date = obs.date + delta_t * s_to_days

    return time_elapsed, n_exp


def plot_plan(plan, date, survey_centers=None, filename=None):
    from matplotlib import pyplot as plt

    fig = plt.figure()

    fig.suptitle(r'$\mathrm{{ Plan \ for \ {} }}$'.format(date), fontsize=18)

    for i, lb in enumerate([False, True]):
        ax = fig.add_subplot(3, 1, i+1)

        if survey_centers is not None:
            coords = (survey_centers['RA'], survey_centers['DEC'])
            if lb:
                coords = equgal(*coords)
                coords[0][:] = ((coords[0] + 180.) % 360.) - 180.
            ax.plot(coords[0], coords[1], 'o', markeredgecolor='none',
                    markerfacecolor='lightgray', markersize=20, zorder=-1)

        coords = (plan['RA'], plan['DEC'])

        if lb:
            coords = equgal(*coords)
            coords[0][:] = ((coords[0] + 180.) % 360.) - 180.

        ax.plot(coords[0], coords[1], '-')

        if lb:
            ax.set_xlabel(r'$\ell$')
            ax.set_ylabel(r'$b$')
        else:
            ax.set_xlabel(r'$\mathrm{RA}$')
            ax.set_ylabel(r'$\delta$')

        startday = np.floor(np.min(plan['approx_time']))

        for f in 'grizY':
            m = plan['filter'] == f
            ax.scatter(coords[0][m], coords[1][m],
                       c=plan['approx_time'][m]-startday, edgecolor='none',
                       marker=filter_symbols[f], facecolor='none', s=50)

    ax = fig.add_subplot(3, 2, 5)
    ax.plot(plan['approx_time']-startday, plan['airmass'])
    ax.set_xlabel('hours since {} UT'.format(ephem.Date(startday)))
    ax.set_ylabel('airmass')
    ax = fig.add_subplot(3, 2, 6)
    ax.plot(plan['approx_time']-startday, np.degrees(plan['moon_sep']))
    ax.set_xlabel('hours since {} UT'.format(ephem.Date(startday)))
    ax.set_ylabel('moon separation')

    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    if filename is not None:
        fig.savefig(filename + '.png')


def write_plan_schedule(plan, fname):
    with open(fname + '_schedule.log', 'w') as f:
        f.write('# TILEID     date-time\n')
        for tid, t in zip(plan['TILEID'], plan['approx_time']):
            f.write('  {:<8d}   {}\n'.format(tid, ephem.Date(t)))


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Plan night.',
        epilog='EXAMPLE: %(prog)s 2016-03-13 decaps-tiles.fits gr plan')
    parser.add_argument('night', type=str, help='night, YYYY-MM-DD')
    parser.add_argument('tilefile', type=str, help='file name of tiles')
    parser.add_argument('filters', type=str, help='filters to run')
    parser.add_argument('outfile', type=str, help='filename to write')
    parser.add_argument('--time', '-t', type=str, default=None,
                        help='time of night to start, 00:00:00.00 (UT)')
    parser.add_argument('--pass', type=int, default=1, dest='skypass',
                        help='Specify pass (dither) number (1,2, or 3)')
    parser.add_argument('--expand-footprint', action='store_true',
                        help='Use tiles outside nominal footprint')
    parser.add_argument(
        '--rd-bounds', metavar='deg', type=float, nargs=4, default=None,
        help=('use only tiles in ra/dec range, specified as '
              '(ramin, ramax, decmin, decmax)'))
    parser.add_argument(
        '--lb-bounds', metavar='deg', type=float, nargs=4, default=None,
        help=('use only tiles in lb range, specified as '
              '(lmin, lmax, bmin, bmax)'))
    parser.add_argument('--chunks', metavar='N', type=int, default=1,
                        help='Split the plan up into N chunks.')
    parser.add_argument('--nightfrac', type=float, default=1.,
                        help='fraction of the night to plan')
    parser.add_argument('--moonsep', type=float, default=40.,
                        help='minimum moon separation to consider')

    args = parser.parse_args()

    tilestable = readTilesTable(
        args.tilefile,
        expand_footprint=args.expand_footprint,
        rdbounds=args.rd_bounds,
        lbbounds=args.lb_bounds,
        skypass=args.skypass
    )

    # Set start time of observing plan
    obs = decam.copy()
    if args.time is None:
        obs.date = args.night
        obs.date = obs.next_setting(ephem.Sun())
    else:
        obs.date = args.night + ' ' + args.time

    plan = GetNightlyStrategy(obs, tilestable[1], args.filters, args.nightfrac,
                              minmoonsep=args.moonsep)
    plot_plan(plan, args.night, filename=args.outfile)
    WriteJSON(plan, args.outfile, chunks=args.chunks)
    write_plan_schedule(plan, args.outfile)


if __name__ == "__main__":
    main()
