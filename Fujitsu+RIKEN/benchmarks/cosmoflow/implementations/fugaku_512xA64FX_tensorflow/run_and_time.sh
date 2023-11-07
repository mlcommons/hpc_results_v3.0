#!/bin/bash

cd ../implementation_fugaku/cosmoflow-benchmark

# closed division
for _ in {1..9}; do
  ./Submit_mult 16x12x48 8x4x16 5:00:00 configs/cosmo_closed_512.yaml vol1 set1 txz ram
done

# open division
#for _ in {1..9}; do
#  ./Submit_mult 16x12x48 8x4x16 5:00:00 configs/cosmo_open_512.yaml vol1 set1 txz ram
#done

# touch /2ndfs/ra010011/cosmoflow/ready_flag/jobs/set1
