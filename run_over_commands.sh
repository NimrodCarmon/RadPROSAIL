#!/bin/bash

# Read the file line by line
while read -r line; do
  sbatch -N 1 -c 40 --mem=180G --wrap="$line"
  sleep 2s
done < "/scratch/carmon/lcluc_fire/data/commands.sh"
