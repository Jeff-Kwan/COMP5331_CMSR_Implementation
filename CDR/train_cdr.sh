#! /bin/bash

target_datasets=("ca" "de" "fr" "in" "jp" "mx" "uk")
models=("DCDCSR" "NATR" "SSCDR")

for model in ${models[@]}; do
    for target in ${target_datasets[@]}; do
        python run_recbole_cdr.py --model $model --target-data $target
    done
done