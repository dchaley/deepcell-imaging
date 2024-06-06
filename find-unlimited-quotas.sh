#!/bin/bash

services=`gcloud services list --enabled --format="value(config.name)"`

for x in $services;
do
  echo "Checking service $x"
  # echo "Hello $x"
  echo `gcloud alpha services quota list --consumer=projects/deepcell-401920 --service=$x --flatten="consumerQuotaLimits[]" --format="csv(consumerQuotaLimits.metric,consumerQuotaLimits.quotaBuckets.effectiveLimit)" --filter="consumerQuotaLimits.quotaBuckets.effectiveLimit=9223372036854775807"`
done
