# COMP5331_CMSR_Implementation
Implementation &amp; Further Study Based on "Pre-Training with Transferable Attention for Addressing Market Shifts in Cross-Market Sequential Recommendation" by Wang et al. 2024

## Requirements

- `recbole==1.1.1`
- `python==3.8.5`
- `cudatoolkit==11.3.1`
- `pytorch==1.12.1`
- `pandas==1.3.0`
- `transformers==4.18.0`

## Dataset Preparation

1. **Preprocessed XMRec Dataset**  
   Copy the preprocessed XMRec dataset from FOREC or MA. Place the data file into the `data` directory.  
   Example: `data/ca_5core.txt`

2. **Amazon Meta Dataset**  
   Download the Amazon meta dataset for the **Electronics** category. Use the **metadata** file.  
   Place the dataset in the `data/Amazon/metadata` directory.  
   Example: `data/Amazon/metadata/meta_Electronics.json.gz`

## Data Processing

Navigate to the `data` directory and process the dataset:
```bash
cd data
python data_process.py


finetune: python finetune.py --config_files='properties/CATSR.yaml properties/market.yaml' --params_file='model.hyper' --output_file='hyper_example.result' --weight_path='saved/CATSR-us-200.pth' --tool='Hyperopt' --dataset='us'
