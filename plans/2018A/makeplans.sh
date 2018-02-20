NIGHTSTRAT=$DECAM_DIR/python/nightstrat.py
PLOTCOVERAGE=$DECAM_DIR/python/plot_coverage.py
TILEFILE=$DECAM_DIR/data/decaps-tiles.fits
WEATHERFILE=$DECAM_DIR/data/badweather.txt
BADEXP=$DECAM_DIR/data/badexp.txt

# 02-02: second half night.  Start: 04:57:09.  All bright.
# 02-03: second half night.  Start: 04:57:15.  All bright.

# probably just izY as far as we can, right?

#python $NIGHTSTRAT 2018-02-03 $TILEFILE izY plan20180202_1_1izY --pass 1 --lb-bounds -80 -53 -20 20 --weatherfile $WEATHERFILE --nightfrac -0.5 --expand-footprint --optimize_ha  # intentional 02-03 for Feb 2; this half night starts on Feb 3 UT
#python $NIGHTSTRAT 2018-02-04 $TILEFILE izY plan20180203_1_2izY --pass 2 --lb-bounds -80 -53 -20 20 --weatherfile $WEATHERFILE --nightfrac -0.5 --expand-footprint --optimize_ha  # intentional 02-03 for Feb 2; this half night starts on Feb 3 UT


# 02-25: full night. moonset at 06:39, roughly Q3.
# 02-26: full night. moonset at 07:43, roughly Q3.
# 02-27: full night. moonset at 08:50, ~end of night.

# okay, so continue izY on similar footprints until moonset, then gr on original footprint?

# 02-25
python $NIGHTSTRAT 2018-02-25 $TILEFILE izY plan20180225_1_2izY --pass 2 --lb-bounds -105 -75 -20 20 --weatherfile $WEATHERFILE --endtime 06:04 --expand-footprint --optimize_ha
python $NIGHTSTRAT 2018-02-25 $TILEFILE izY plan20180225_2_2izY --pass 2 --lb-bounds -60 -50 -20 20 --weatherfile $WEATHERFILE --time +06:04 --endtime +06:39 --expand-footprint --optimize_ha
python $NIGHTSTRAT 2018-02-25 $TILEFILE gr plan20180225_3_1gr --pass 1 --lb-bounds -55 -42 -20 20 --weatherfile $WEATHERFILE --time +06:39 --expand-footprint --optimize_ha

# 02-26
python $NIGHTSTRAT 2018-02-26 $TILEFILE izY plan20180226_1_1izY --pass 1 --lb-bounds -113 -75 -20 20 --weatherfile $WEATHERFILE --endtime +07:43 --expand-footprint --optimize_ha
python $NIGHTSTRAT 2018-02-26 $TILEFILE gr plan20180226_2_2gr --pass 2 --lb-bounds -55 -46 -20 20 --weatherfile $WEATHERFILE --time +07:43 --expand-footprint --optimize_ha

# 02-27
python $NIGHTSTRAT 2018-02-27 $TILEFILE izY plan20180227_1_3izY --pass 3 --lb-bounds -105 -60 -20 20 --weatherfile $WEATHERFILE --endtime +08:50 --expand-footprint --optimize_ha
python $NIGHTSTRAT 2018-02-27 $TILEFILE gr plan20180227_3_3gr --pass 3 --lb-bounds -55 -51 -20 20 --weatherfile $WEATHERFILE --time +08:50 --expand-footprint --optimize_ha # ~half of this time is after 12 degree twilight; we could forget about the blue bands if we wanted.

# Plot overall coverage
for pass in {1..3}; do
    outFname=coverage_thru_2018-02-27_p${pass}.png
    python $PLOTCOVERAGE $TILEFILE -o $outFname -p plan2018022?_?_*.json --weather $WEATHERFILE --bad-exposures $BADEXP --decaps-extended-only --pass $pass
done

# Plot coverage on each night
for day in {25..27}; do
    outFname=coverage_2018-02-${day}.png
    python $PLOTCOVERAGE $TILEFILE -o $outFname -p plan201802${day}_?_*.json --weather $WEATHERFILE --bad-exposures $BADEXP --decaps-extended-only
done

# where do we stand?  izY mostly covered for early half of the footprint.  The May runs can (need to!) pick up some of the izY second half footprint.
# gr is basically empty, except for a few patches toward the middle of the footprint.  There will be more than enough left for the May time.
# izY: pass 1 to -53, 2 to -45, 3 to -60


# following set of runs are in May.
# 05-08: second half.  Start: 04:39.  moon rises 05:21, early in half night.
# 05-09: second half.  Start: 04:39.  moon rises 06:16, early in half night.
# 05-10: second half.  Start: 04:39.  moon rises 07:12, near Q3.
# 05-11: second half.  Start: 04:39.  moon rises 08:10, near Q3.
# 05-18: full.  moon sets at 01:16, near Q1.
# 05-19: full.  moon sets at 02:20, near Q1.  
# 05-20: full.  moon sets at 03:25, near Q1.5.


## 05-08:
#python $NIGHTSTRAT 2018-05-08 $TILEFILE gr plan20180508_1_1gr --pass 2 --lb-bounds -42 -39 -20 20 --weatherfile $WEATHERFILE --time 04:39 --endtime 05:21 --expand-footprint --optimize_ha
#python $NIGHTSTRAT 2018-05-08 $TILEFILE izY plan20180508_2_3izY --pass 3 --lb-bounds -45 -22 -20 20 --weatherfile $WEATHERFILE --time 05:21 --expand-footprint --optimize_ha
#
## 05-09
#python $NIGHTSTRAT 2018-05-09 $TILEFILE gr plan20180509_1_2gr --pass 2 --lb-bounds -46 -38 -20 20 --weatherfile $WEATHERFILE --time 04:39 --endtime 06:16 --expand-footprint --optimize_ha
#python $NIGHTSTRAT 2018-05-09 $TILEFILE izY plan20180509_2_1izY --pass 1 --lb-bounds -30 -7 -20 20 --weatherfile $WEATHERFILE --time 06:16 --expand-footprint --optimize_ha
#
## 05-10
#python $NIGHTSTRAT 2018-05-10 $TILEFILE gr plan20180510_1_3gr --pass 3 --lb-bounds -51 -40 -20 20 --weatherfile $WEATHERFILE --time 04:39 --endtime 07:12 --expand-footprint --optimize_ha
#python $NIGHTSTRAT 2018-05-10 $TILEFILE izY plan20180510_2_2izy --pass 2 --lb-bounds -20 0 -20 20 --weatherfile $WEATHERFILE --time 07:12 --expand-footprint --optimize_ha
#
## 05-11
## there's still plenty of sky to observe here, but it gets kind of crazy to plan it.  We know the moon is not a problem.
#
## 05-19
## one example night to check for moon problems.
#python $NIGHTSTRAT 2018-05-19 $TILEFILE izY plan20180519_1_3izY --pass 3 --lb-bounds -60 -45 -20 20 --weatherfile $WEATHERFILE --endtime 02:20 --expand-footprint --optimize_ha
#python $NIGHTSTRAT 2018-05-19 $TILEFILE gr plan20180519_2_2gr --pass 2 --lb-bounds -38.5 5 -20 20 --weatherfile $WEATHERFILE --time 02:20 --expand-footprint --optimize_ha
## also ~2 hours of imaging on the other side of the footprint above.  Should be fine.
