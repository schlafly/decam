basedir=${DECAM_DIR}

NIGHTSTRAT=${basedir}/python/nightstrat.py
PLOTCOVERAGE=${basedir}/python/plot_coverage.py
TILEFILE=${basedir}/data/decaps-tiles.fits
WEATHERFILE=${basedir}/data/badweather.txt
BADEXP=${basedir}/data/badexp.txt

# Times for all the nights:
# 01-30: moonris: 06:35, Q2: 04:56, 18 twi: 08:42
# 01-31: moonris: 07:21, Q2: 04:57, 18 twi: 08:43

# 01-30:
python $NIGHTSTRAT 2019-01-30 $TILEFILE gr plan_2019y01m30d_1_2gr --pass 2 --lb-bounds -125 -60 -20 20 --weatherfile $WEATHERFILE --time +04:56 --endtime +06:35 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-30 $TILEFILE izY plan_2019y01m30d_2_2izY --pass 2 --lb-bounds -125 -30 -20 20 --weatherfile $WEATHERFILE --time +06:35 --expand-footprint --optimize_ha --assumeplans .

# 01-31:
python $NIGHTSTRAT 2019-01-31 $TILEFILE gr plan_2019y01m31d_1_3gr --pass 1 --lb-bounds -125 -60 -20 20 --weatherfile $WEATHERFILE --time +04:57 --endtime +07:21 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-31 $TILEFILE izY plan_2019y01m31d_2_3izY --pass 1 --lb-bounds -125 -30 -20 20 --weatherfile $WEATHERFILE --time +07:21 --expand-footprint --optimize_ha --assumeplans .

#Plot coverage on each night
for day in {30,31}; do
    outFname=coverage_2019y01m${day}d.png
    python $PLOTCOVERAGE $TILEFILE -o $outFname -p plan_2019y01m${day}d_?_*.json --weather $WEATHERFILE --bad-exposures $BADEXP --decaps-extended-only
done

#Plot total coverage for first run
x='plan_2019y01m*.json'
y=$(echo $x | find . -name "plan_2019y01m*.json")
outFname=totalcoverage.png
python $PLOTCOVERAGE $TILEFILE -o $outFname -p echo $y --weather $WEATHERFILE --bad-exposures $BADEXP --decaps-extended-only

