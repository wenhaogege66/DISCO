#!/bin/bash
# Convert DISCO Yelp data to SASRec format.
# Run this once before training.
# Output: SASRec/python/data/Yelp.txt

set -e
cd /home/sjj/wenhao

conda run -n DDBC python DISCO/datasets/Yelp/convert_to_sasrec.py --dataset Yelp
