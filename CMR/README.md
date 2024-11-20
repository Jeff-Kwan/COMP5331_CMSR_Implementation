# Code for cross market models

### Environment setup

1. Install `python 3.7.10`. 
2. Install requirements via `pip install -r requirements.txt`

### Reproducing experiments
1. Run the commands experiments using the instructions in [RUN.md](RUN.md)
2. Create a directory `raw_results` at the repo root, and move `forec_eval_single`, `forec_eval_all`, 
`forec_eval_all_market_aware`, and `forec_single_model` into `raw_results`.
3. Run [the results nb](results.ipynb) 


## Global Model Experiments

```
# Market Unaware
python train_base_all.py  --data_dir DATA2/proc_data --exp_name base_single_model --cuda --data_sampling_method concat --batch_size 1024 --num_epoch 25 --exp_output forec_single_model/base-forec_single_model_concat.json
# MAML
python train_maml.py  --experiment_type single_model --data_dir DATA2/proc_data --exp_name base_single_model --cuda --data_sampling_method concat --num_epoch 25 --exp_output forec_single_model/maml-forec_single_model_concat.json
# FOREC
python train_forec.py  --experiment_type single_model --data_dir DATA2/proc_data --exp_name base_single_model --cuda --data_sampling_method concat  --num_epoch 25 --exp_output forec_single_model/forec-forec_single_model_concat.json
# Market Aware
python train_base_all.py  --data_dir DATA2/proc_data --exp_name base_single_model --cuda --data_sampling_method concat --batch_size 1024 --num_epoch 25 --market_aware --exp_output forec_single_model/base-forec_single_model_concat_market_aware.json
```