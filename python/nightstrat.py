"""
Simplified version of DECaLS nightlystrategy.py

strategy: observe the pointing that is getting worst the fastest
  prefer nearby exposures over just getting the single pointing getting worse
  the fastest.
"""

import pyfits
import ephem
import numpy
import numpy as np

import pdb

from collections import OrderedDict

# Global constants
R_earth = 6378.1e3 # in meters
# conversion factor s_to_days*secs = days
s_to_days = (1.0/(60.0*60.0*24.0))
days_to_s = (60.0*60.0*24.0)

decam = ephem.Observer()
decam.lon = '-70.806525'
decam.lat = '-30.169661'
decam.elev = 2207.0 # meters
decam.temp = 10.0 # deg celsius; average temp for August
decam.pressure = 780.0 # mbar
#decam.horizon = -np.sqrt(2.0*decam.elev/R_earth)
decam.horizon = -ephem.degrees('12:00:00.0')

exp_time_filters = {'g':96, 'r':30, 'i':30, 'z':30, 'y':30 }
filter_symbols = {'g':'+', 'r':'<', 'i':'p', 'z':'o', 'y':'x'}

sun = ephem.Sun()
moon = ephem.Moon()

overheads = 30.

#####################################################
# Observing Functions, Pyephem, etc
#####################################################

def get_twilight(obs):
    '''
    Compute the twilight times for an observer. If the night
    has already begun for observer, set start of night to observer's
    time.
    '''
    sun = ephem.Sun()

    start_night = obs.next_setting(sun)
    end_night = obs.next_rising(sun)

    if start_night > end_night:
        start_night = obs.date

    return start_night, end_night


#####################################################
def GetAirmass(al):
    if isinstance(al, list):
        al[al < 0.07] = 0.07
    else:
        al = 0.07 if al < 0.07 else al
    secz = 1.0/np.sin(al)
    seczm1 = secz-1.0
    airm = secz-0.0018167*seczm1-0.002875*seczm1**2-0.0008083*seczm1**3
    return airm

#####################################################
def GetLST(obs):
    lst_deg = float(obs.sidereal_time())*180.0/np.pi
    return lst_deg

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
    for k in range(0,raval.size):
        #print hours[k],minutes[k], seconds[k]
        stringra.append("%02d:%02d:%04.1f" % (hours[k], minutes[k], seconds[k]))

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
    for k in range(0,decval.size):
        #print sdd[k],minutes[k], seconds[k]
        stringdec.append("%02d:%02d:%02d" % (sdd[k], minutes[k], seconds[k]))

    stringdec = np.array(stringdec)
    return stringdec
#####################################################
#####################################################

def gc_dist(lon1, lat1, lon2, lat2):
    from numpy import sin, cos, arcsin, sqrt

    lon1 = np.radians(lon1); lat1 = np.radians(lat1)
    lon2 = np.radians(lon2); lat2 = np.radians(lat2)

    return np.degrees(2*arcsin(sqrt( (sin((lat1-lat2)*0.5))**2 + cos(lat1)*cos(lat2)*(sin((lon1-lon2)*0.5))**2 )));


#####################################################
def WriteJSON(pl, outfilename):
    jf = open(outfilename,'w')
    jf.write('['+ '\n')

    ntot = len(pl['RA'])

    for k in range(0,ntot):
        jf.write(' {'+'\n')

        jf.write('  "seqid": '+'"'+'%d' % (1)+'",'+'\n')
        jf.write('  "seqnum": '+ '%d' % (k+1) +','+'\n')
        jf.write('  "seqtot": '+'%d' % ntot+','+'\n')
        jf.write('  "expType": '+'"object",'+'\n')
        jf.write('  "object": '+ '"DECaPS_'+str(pl['TILEID'][k])+'_'+pl['filter'][k]+'",'+'\n')
        jf.write('  "expTime": '+'%d' % pl['exp_time'][k]+','+'\n')
        jf.write('  "filter": '+'"'+pl['filter'][k]+'",'+'\n')

        ra_w = pl['RA'][k] % 360.

        jf.write('  "RA": '+'%.3f' % ra_w+','+'\n')
        jf.write('  "dec": '+'%.3f' % pl['DEC'][k]+'\n')
        if k == len(pl['RA'])-1:
            jf.write(' }'+'\n')
        else:
            jf.write(' },'+'\n')



    jf.write(']')
    jf.close()

    # Check that we wrote valid JSON by opening & parsing
    f = open(outfilename)
    import json
    json.load(f)
    f.close()


def equgal(ra, dec):
    coord = [ephem.Galactic(ephem.Equatorial(ra0*numpy.pi/180., dec0*numpy.pi/180.))
             for ra0, dec0 in zip(ra, dec)]
    l = numpy.array([coord0.lon*180./numpy.pi for coord0 in coord])
    b = numpy.array([coord0.lat*180./numpy.pi for coord0 in coord])
    return l, b

def readTilesTable(filename, expand_footprint=False, rdbounds=None,
                   lbbounds=None, skypass=-1):
    tiles_in = pyfits.getdata(filename, 1)

    tiles = OrderedDict()
    # Check that required columns exist
    for col in ['TILEID','PASS','IN_SDSS','IN_DES','IN_DESI','IN_DECAPS',
                'G_DONE','R_DONE', 'I_DONE', 'Z_DONE', 'Y_DONE']:
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
        I = (I & (lt > lbbounds[0]) & (lt <= lbbounds[1]) &
             (bt > lbbounds[2]) & (bt <= lbbounds[3]))

    survey = OrderedDict([(k,v[I]) for k,v in tiles.items()])

    # H:M:S and D:M:S strings
    survey['RA_STR']  = ConvertRA(survey['RA'])
    survey['DEC_STR'] = ConvertDec(survey['DEC'])

    return tiles,survey


def GetNightlyStrategy(obs, survey_centers, filters):
    """
    date: UT; if time is not set, the next setting of the sun following start of that date is
          the start of the plan; awkward when the night starts just before midnight UT, as it does in March in Chile!
    """

    tonightsplan = OrderedDict()
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
    sn, en = get_twilight(obs)
    obs.date = sn

    lon = (en-sn) * days_to_s

    # Make sure the Sun isn't up
    sun.compute(obs)
    moon.compute(obs)
    if sun.alt > 0:
        print 'WARNING: sun is up?!'

    print 'Date: {}'.format(obs.date)
    print 'Length of night: {} s'.format(lon)
    print 'Start time of plan (UT): {}'.format(sn)
    print 'End time of night (UT): {}'.format(en)

    for f in 'grizy':
        col = 'used_tile_{:s}'.format(f)
        survey_centers[col] = survey_centers['{:s}_DONE'.format(f.capitalize())].copy()

    time_elapsed = 0.0
    filterorder = 1

    while time_elapsed < lon:
        obs.date = sn+time_elapsed*s_to_days
        start_obsdate = obs.date

        if obs.date > en:
            break

        sun.compute(obs)
        moon.compute(obs)

        airmass = np.zeros_like(survey_centers['RA'])
        airmassp = np.zeros_like(survey_centers['RA'])

        # compute derivative of airmass for each exposure
        for k in range(0,survey_centers['RA'].size):
            this_tile = ephem.readdb(str(survey_centers['TILEID'][k])+','+'f'+','+
                                     survey_centers['RA_STR'][k]+','+
                                     survey_centers['DEC_STR'][k]+','+'20')
            this_tile.compute(obs)
            airmass[k] = GetAirmass(float(this_tile.alt))
        obs.date = obs.date + 1./24./60./60.
        for k in range(0,survey_centers['RA'].size):
            this_tile = ephem.readdb(str(survey_centers['TILEID'][k])+','+'f'+','+
                                     survey_centers['RA_STR'][k]+','+
                                     survey_centers['DEC_STR'][k]+','+'20')
            this_tile.compute(obs)
            airmassp[k] = GetAirmass(float(this_tile.alt))
        obs.date = sn+time_elapsed*s_to_days # reset date
        # tile getting worse the fastest
        dairmass = airmassp - airmass
        exclude = numpy.zeros(len(survey_centers['RA']), dtype='bool')
        exclude = exclude | (airmass > 5)
        for f in filters:
            exclude = exclude | survey_centers['used_tile_%s' % f]
        if numpy.all(exclude):
            print 'Ran out of tiles to observe before night was done!'
            print 'Minutes left in night: %5.1f' % ((lon-time_elapsed)/60.)
            break
        if len(tonightsplan['RA']) > 1:
            slew = numpy.clip(gc_dist(tonightsplan['RA'][-1], tonightsplan['DEC'][-1],
                                      survey_centers['RA'], survey_centers['DEC'])-2, 0., numpy.inf)
        else:
            slew = 0
        nexttile = numpy.argmax(dairmass-slew*0.00001-exclude*1.e10)
        deltat, nexp = pointing_plan(tonightsplan, orig_keys, survey_centers, nexttile, filters[::filterorder], obs)
        time_elapsed += deltat
        #print survey_centers['RA_STR'][nexttile], survey_centers['DEC_STR'][nexttile]
        filterorder = -filterorder
        if len(tonightsplan['RA']) > nexp:
            slew = gc_dist(tonightsplan['RA'][-1], tonightsplan['DEC'][-1],
                           tonightsplan['RA'][-nexp-1], tonightsplan['DEC'][-nexp-1])
            slewtime = 3.0*max(slew-2.0, 0.0)
            time_elapsed += slewtime
            if slewtime > 0:
                print 'time spent slewing', slewtime

    numleft = numpy.sum(exclude == 0)
    print 'Plan complete, %d observations, %d remaining.' % (len(tonightsplan['RA']), numleft)
    keys = tonightsplan.keys()
    return numpy.rec.fromarrays([tonightsplan[k] for k in keys], names=keys)


def pointing_plan(tonightsplan, orig_keys, survey_centers, nexttile, filters, obs):
    time_elapsed = 0
    nexp = 0
    for f in filters:
        if survey_centers['used_tile_%s'%f][nexttile] == 1:
            continue
        nexp += 1
        survey_centers['used_tile_%s'%f][nexttile] = 1
        for key in orig_keys:
            tonightsplan[key].append(survey_centers[key][nexttile])
        tonightsplan['exp_time'].append(exp_time_filters[f])
        tonightsplan['approx_datetime'].append(obs.date)
        this_tile = ephem.readdb(str(survey_centers['TILEID'][nexttile])+','+'f'+','+
                                 survey_centers['RA_STR'][nexttile]+','+
                                 survey_centers['DEC_STR'][nexttile]+','+'20')
        this_tile.compute(obs)
        airm = GetAirmass(float(this_tile.alt))
        moon.compute(obs)
        moon_dist = ephem.separation((this_tile.az,this_tile.alt),(moon.az,moon.alt))
        moon_alt = moon.alt*180.0/np.pi

        tonightsplan['airmass'].append(airm)
        #pdb.set_trace()
        tonightsplan['approx_time'].append(obs.date)
        tonightsplan['filter'].append(f)
        tonightsplan['moon_sep'].append(moon_dist)
        tonightsplan['moon_alt'].append(moon_alt)
        tonightsplan['lst'].append(GetLST(obs))

        deltt = exp_time_filters[f] + overheads
        time_elapsed += deltt
        obs.date = obs.date + deltt*s_to_days
    return time_elapsed, nexp

def plot_plan(plan, survey_centers=None, filename=None):
    from matplotlib import pyplot as p
    p.clf()
    for i, lb in enumerate([False, True]):
        p.subplot(3,1,i+1)
        if survey_centers is not None:
            coords = (survey_centers['RA'], survey_centers['DEC'])
            if lb:
                coords = equgal(*coords)
            p.plot(coords[0], coords[1], 'o', markeredgecolor='none',
                   markerfacecolor='lightgray', markersize=20, zorder=-1)
        coords = (plan['RA'], plan['DEC'])
        if lb:
            coords = equgal(*coords)
        p.plot(coords[0], coords[1], '-')
        if lb:
            p.xlabel('l')
            p.ylabel('b')
        else:
            p.xlabel('l')
            p.ylabel('b')

        startday = numpy.floor(numpy.min(plan['approx_time']))
        for f in 'grizy':
            m = plan['filter'] == f
            p.scatter(coords[0][m], coords[1][m], c=plan['approx_time'][m]-startday,
                      marker=filter_symbols[f], facecolor='none', s=100)

    p.subplot(3,2,5)
    p.plot(plan['approx_time']-startday, plan['airmass'])
    p.xlabel('hours since %s UT' % ephem.Date(startday))
    p.ylabel('airmass')
    p.subplot(3,2,6)
    p.plot(plan['approx_time']-startday, plan['moon_sep']*180/numpy.pi)
    p.xlabel('hours since %s UT' % ephem.Date(startday))
    p.ylabel('moon separation')
    if filename is not None:
        p.savefig(filename)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plan night.',
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
    parser.add_argument('--rd-bounds', metavar='deg', type=float, nargs=4, default=None,
                        help='use only tiles in ra/dec range, specified as (ramin, ramax, decmin, decmax)')
    parser.add_argument('--lb-bounds', metavar='deg', type=float, nargs=4, default=None,
                        help='use only tiles in lb range, specified as (lmin, lmax, bmin, bmax)')
    args = parser.parse_args()

    tilestable = readTilesTable(args.tilefile, expand_footprint=args.expand_footprint,
                                rdbounds=args.rd_bounds,
                                lbbounds=args.lb_bounds,
                                skypass=args.skypass)

    decam.date = '{} {}'.format(
        args.night,
        '' if args.time == None else args.time
    )

    plan = GetNightlyStrategy(decam, tilestable[1], args.filters)
    plot_plan(plan, filename=args.outfile+'.png')
    WriteJSON(plan, args.outfile+'.json')





if __name__ == "__main__":
    main()
