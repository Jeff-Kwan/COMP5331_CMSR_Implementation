# COMP5331_CMSR_Implementation
Implementation &amp; Further Study Based on "Pre-Training with Transferable Attention for Addressing Market Shifts in Cross-Market Sequential Recommendation" by Wang et al. 2024

finetune: python finetune.py --config_files='properties/CATSR.yaml properties/market.yaml' --params_file='model.hyper' --output_file='hyper_example.result' --weight_path='saved/CATSR-us-200.pth' --tool='Hyperopt' --dataset='us'
