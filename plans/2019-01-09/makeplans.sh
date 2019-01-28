
# Times for all the nights:
# 01-09: moonset: 02:22, Q2: 04:51, 18 twi: 08:16 
# 01-10: moonset: 02:55, Q2: 04:51, 18 twi: 08:16 
# 01-11: moonset: 03:27, Q2: 04:51, 18 twi: 08:18 
# 01-12: moonset: 03:59, Q2: 04:52, 18 twi: 08:19 
# 01-13: moonset: 04:31, Q2: 04:52, 18 twi: 08:20 
# 01-14: moonset: 05:05, Q2: 04:52, 18 twi: 08:21 


# 01-09
python $NIGHTSTRAT 2019-01-09 $TILEFILE gr plan20190109_1_1gr --pass 1 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +04:51 --endtime +08:16 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-09 $TILEFILE izY plan20190109_2_1izY --pass 1 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +08:16 --expand-footprint --optimize_ha --assumeplans .

# 01-10
python $NIGHTSTRAT 2019-01-10 $TILEFILE gr plan20190110_1_2gr --pass 2 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +04:51 --endtime +08:16 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-10 $TILEFILE izY plan20190110_2_2izY --pass 2 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +08:16 --expand-footprint --optimize_ha --assumeplans .

# 01-11
python $NIGHTSTRAT 2019-01-11 $TILEFILE gr plan20190111_1_3gr --pass 3 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +04:51 --endtime +08:18 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-11 $TILEFILE izY plan20190111_2_3izY --pass 3 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +08:18 --expand-footprint --optimize_ha --assumeplans .

# 01-12
python $NIGHTSTRAT 2019-01-12 $TILEFILE gr plan20190112_1_1gr --pass 1 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +04:52 --endtime +08:19 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-12 $TILEFILE izY plan20190112_2_1izY --pass 1 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +08:19 --expand-footprint --optimize_ha --assumeplans .


# 01-13
python $NIGHTSTRAT 2019-01-13 $TILEFILE gr plan20190113_1_2gr --pass 2 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +04:52 --endtime +08:20 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-13 $TILEFILE izY plan20190113_2_2izY --pass 2 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +08:20 --expand-footprint --optimize_ha --assumeplans .


# 01-14
python $NIGHTSTRAT 2019-01-14 $TILEFILE izY plan20190114_1_3izY --pass 3 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +04:52 --endtime +05:05 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-14 $TILEFILE gr plan20190114_2_3gr --pass 3 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +05:05 --endtime +08:21 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-14 $TILEFILE izY plan20190114_3_3izY --pass 3 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +08:21 --expand-footprint --optimize_ha --assumeplans .

# 01-15: moonset: 05:42, Q2: 04:53, 18 twi: 08:22
# 01-16: moonset: 06:24, Q2: 04:53, 18 twi: 08:23
# 01-17: moonset: 07:12, Q2: 04:54, 18 twi: 08:24
# 01-18: moonset: 08:07, Q2: 04:54, 18 twi: 08:26

# 01-15
python $NIGHTSTRAT 2019-01-15 $TILEFILE izY plan20190115_1_1izY --pass 1 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +04:53 --endtime +05:42 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-15 $TILEFILE gr plan20190115_2_1gr --pass 1 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +05:42 --endtime +08:22 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-15 $TILEFILE izY plan20190115_3_1izY --pass 1 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +08:22 --expand-footprint --optimize_ha --assumeplans .

# 01-16:
python $NIGHTSTRAT 2019-01-16 $TILEFILE izY plan20190116_1_2izY --pass 2 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +04:53 --endtime +06:24 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-16 $TILEFILE gr plan20190116_2_2gr --pass 2 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +06:24 --endtime +08:23 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-16 $TILEFILE izY plan20190116_3_2izY --pass 2 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +08:23 --expand-footprint --optimize_ha --assumeplans .

# 01-17:
python $NIGHTSTRAT 2019-01-17 $TILEFILE izY plan20190117_1_3izY --pass 3 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +04:54 --endtime +07:12 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-17 $TILEFILE gr plan20190117_2_3gr --pass 3 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +07:12 --endtime +08:24 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-17 $TILEFILE izY plan20190117_3_3izY --pass 3 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +08:24 --expand-footprint --optimize_ha --assumeplans .

# 01-18:
python $NIGHTSTRAT 2019-01-18 $TILEFILE izY plan20190118_1_3izY --pass 3 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +04:54 --endtime +08:07 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-18 $TILEFILE gr plan20190118_2_3gr --pass 3 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +08:07 --endtime +08:26 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-18 $TILEFILE izY plan20190118_3_3izY --pass 3 --lb-bounds -125 -80 -20 20 --weatherfile $WEATHERFILE --time +08:26 --expand-footprint --optimize_ha --assumeplans .


# 01-30: moonris: 06:35, Q2: 04:57, 18 twi: 08:40
# 01-31: moonris: 07:05, Q2: 04:57, 18 twi: 08:41

# 01-30:
python $NIGHTSTRAT 2019-01-30 $TILEFILE gr plan20190130_1_2gr --pass 2 --lb-bounds -125 -60 -20 20 --weatherfile $WEATHERFILE --time +04:57 --endtime +06:35 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-30 $TILEFILE izY plan20190130_2_2izY --pass 2 --lb-bounds -125 -30 -20 20 --weatherfile $WEATHERFILE --time +06:35 --expand-footprint --optimize_ha --assumeplans .

# 01-31:
python $NIGHTSTRAT 2019-01-31 $TILEFILE gr plan20190131_1_3gr --pass 1 --lb-bounds -125 -60 -20 20 --weatherfile $WEATHERFILE --time +04:57 --endtime +07:05 --expand-footprint --optimize_ha --assumeplans .
python $NIGHTSTRAT 2019-01-31 $TILEFILE izY plan20190131_2_3izY --pass 1 --lb-bounds -125 -30 -20 20 --weatherfile $WEATHERFILE --time +07:05 --expand-footprint --optimize_ha --assumeplans .

#Plot coverage on each night
for day in {09,10,11,12,13,14,15,16,17,18,30,31}; do
    outFname=coverage_2019-01-${day}.png
    python $PLOTCOVERAGE $TILEFILE -o $outFname -p plan201901${day}_?_*.json --weather $WEATHERFILE --bad-exposures $BADEXP --decaps-extended-only
done

#Plot total coverage for first run
x='plan201801*.json'
y=$(echo $x | find . -name "plan201801*.json")
outFname=totalcoverage.png
python $PLOTCOVERAGE $TILEFILE -o $outFname -p echo $y --weather $WEATHERFILE --bad-exposures $BADEXP --decaps-extended-only

