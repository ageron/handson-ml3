
# https://www.kaggle.com/c/nyc-taxi-trip-duration

OUT = ../../out
TEMP = /tmp/constant
ORIG_DATASET = $(TEMP)/nyc-taxi-trip-duration.zip
TRAIN = $(TEMP)/train.csv
OUTPUT = \
 $(TRAIN) \
 $(TEMP)/train-10k.csv \
 $(TEMP)/taxi.db \

all: $(OUTPUT)

$(TEMP):
	mkdir -p $@

$(ORIG_DATASET):
	cd $(shell dirname $@) && kaggle competitions download -c nyc-taxi-trip-duration
	touch $@

$(TEMP)/train.zip: $(TEMP) $(ORIG_DATASET)
	unzip -d $(TEMP) -u $(ORIG_DATASET)
	touch $@

$(TRAIN): $(TEMP)/train.zip
	unzip -d $(TEMP) -u $<
	touch $@
	test $$(wc -l < $@) -eq 1458645

ZONE_URL = 'https://data.cityofnewyork.us/api/geospatial/d3c5-ddgc?method=export&format=Shapefile'
$(TEMP)/train-10k.csv: $(TRAIN)
	head -n 10001 $<  > $@
	cd $(TEMP) && wget -S -O taxi_zones.zip  $(ZONE_URL)

export PYTHONPATH += :../../..

$(TEMP)/taxi.db: $(TEMP)/train-10k.csv
	./etl.py $<

clean:
	rm -f $(OUTPUT)

very-clean: clean
	rm -rf $(TEMP)
