NIGHTSTRAT=$DECAM_DIR/python/nightstrat.py
PLOTCOVERAGE=$DECAM_DIR/python/plot_coverage.py
TILEFILE=$DECAM_DIR/data/decaps-tiles.fits
WEATHERFILE=$DECAM_DIR/data/badweather.txt
BADEXP=$DECAM_DIR/data/badexp.txt

## 05-08:
python $NIGHTSTRAT 2018-05-08 $TILEFILE gr plan20180508_1_1gr --pass 1 --lb-bounds -38.7 -35.65 -20 20 --weatherfile $WEATHERFILE --time +04:39 --endtime +05:21 --expand-footprint --optimize_ha --chunk_size 50  #0.7 hr dark
python $NIGHTSTRAT 2018-05-08 $TILEFILE iz plan20180508_2_2iz --pass 2 --lb-bounds -62 -18 -20 20 --weatherfile $WEATHERFILE --time +05:21 --expand-footprint --optimize_ha --chunk_size 50  #5 hours bright
#
## 05-09
python $NIGHTSTRAT 2018-05-09 $TILEFILE gr plan20180509_1_1gr --pass 1 --lb-bounds -35.65 -29 -20 20 --weatherfile $WEATHERFILE --time +04:39 --endtime +06:16 --expand-footprint --optimize_ha --chunk_size 50  #1.6 hours dark
python $NIGHTSTRAT 2018-05-09 $TILEFILE iz plan20180509_2_1iz --pass 1 --lb-bounds -57 5 -20 0 --weatherfile $WEATHERFILE --time +06:16 --expand-footprint --optimize_ha --chunk_size 50  #4 hours bright
#
## 05-10
python $NIGHTSTRAT 2018-05-10 $TILEFILE gr plan20180510_1_1gr --pass 1 --lb-bounds -29 -16.4 -20 20 --weatherfile $WEATHERFILE --time +04:39 --endtime +07:12 --expand-footprint --optimize_ha --chunk_size 50  #2.5 hours dark
python $NIGHTSTRAT 2018-05-10 $TILEFILE iz plan20180510_2_1iz --pass 1 --lb-bounds -45 5 0 20 --weatherfile $WEATHERFILE --time +07:12 --expand-footprint --optimize_ha --chunk_size 50  #3.2 hours bright

##05-11
python $NIGHTSTRAT 2018-05-11 $TILEFILE gr plan20180511_1_1gr --pass 1 --lb-bounds -16.4 5 -20 20 --weatherfile $WEATHERFILE --time +04:39 --endtime +08:10 --expand-footprint --optimize_ha --chunk_size 50  #3.5 hours dark
python $NIGHTSTRAT 2018-05-11 $TILEFILE iz plan20180511_2_2iz --pass 2 --lb-bounds -16.5 5 -20 20 --weatherfile $WEATHERFILE --time +08:10 --expand-footprint --optimize_ha --chunk_size 50  #2.2 hours bright

# Plot overall coverage
#for pass in {1..3}; do
#    outFname=coverage_thru_2018-05-10_p${pass}.png
#    python $PLOTCOVERAGE $TILEFILE -o $outFname -p plan201805??_?_*.json --weather $WEATHERFILE --bad-exposures $BADEXP --decaps-extended-only --pass $pass
#done

#Plot coverage on each night
for day in {08,09,10,11}; do
    outFname=coverage_2018-05-${day}.png
    python $PLOTCOVERAGE $TILEFILE -o $outFname -p plan201805${day}_?_*.json --weather $WEATHERFILE --bad-exposures $BADEXP --decaps-extended-only
done

#Plot total coverage for first run
x='plan201805*.json'
y=$(echo $x | find . -name "plan201805*.json")
outFname=totalcoverage.png
python $PLOTCOVERAGE $TILEFILE -o $outFname -p echo $y --weather $WEATHERFILE --bad-exposures $BADEXP --decaps-extended-only

