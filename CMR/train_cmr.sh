#! /bin/bash

# Market Unaware
python train_base_all.py  --data_dir DATA2/proc_data --exp_name base_single_model --cuda --data_sampling_method concat --batch_size 1024 --num_epoch 25 --exp_output forec_single_model/base-forec_single_model_concat.json
# MAML
python train_maml.py  --experiment_type single_model --data_dir DATA2/proc_data --exp_name base_single_model --cuda --data_sampling_method concat --num_epoch 25 --exp_output forec_single_model/maml-forec_single_model_concat.json
# FOREC
python train_forec.py  --experiment_type single_model --data_dir DATA2/proc_data --exp_name base_single_model --cuda --data_sampling_method concat  --num_epoch 25 --exp_output forec_single_model/forec-forec_single_model_concat.json
# Market Aware
python train_base_all.py  --data_dir DATA2/proc_data --exp_name base_single_model --cuda --data_sampling_method concat --batch_size 1024 --num_epoch 25 --market_aware --exp_output forec_single_model/base-forec_single_model_concat_market_aware.json
