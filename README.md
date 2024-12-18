### Implementation & Further Study for HKUST COMP5331 Based on "Pre-Training with Transferable Attention for Addressing Market Shifts in Cross-Market Sequential Recommendation" by Wang et al. 2024
This branch includes both the version 1: original implementation with some minor code modifications; and the version2: implementation with the transformer algorithm architecture modifications (all the files ending with "2").

## Requirements

- `recbole==1.1.1`
- `python==3.8.5`
- `cudatoolkit==11.3.1`
- `pytorch==1.12.1`
- `pandas==1.3.0`
- `transformers==4.18.0`

## Dataset Preparation

1. **Preprocessed XMRec Dataset**  
   Copy the preprocessed XMRec dataset from [FOREC](https://github.com/hamedrab/FOREC/tree/main/DATA/proc_data) or [MA](https://github.com/samarthbhargav/efficient-xmrec/tree/main/DATA2/proc_data). Place the data file into the `data` directory.  
   Example: `data/ca_5core.txt`

2. **[Amazon Meta Dataset](https://nijianmo.github.io/amazon/index.html)**  
   Download the Amazon meta dataset for the **Electronics** category. Use the **metadata** file.  
   Place the dataset in the `data/Amazon/metadata` directory.  
   Example: `data/Amazon/metadata/meta_Electronics.json.gz`

## Data Processing

Navigate to the `data` directory and process the dataset:
```bash
cd data
python data_process.py
```

## Pretrain US Market using the original algorithm
```bash
python pretrain.py
```

## Finetune with hypertuning (for example with 'ca') using the original algorithm
```bash
python finetune.py --config_files='properties/CATSR.yaml properties/market.yaml' --params_file='model.hyper' --output_file='hyper_example.result' --weight_path='saved/CATSR-us-200.pth' --tool='Hyperopt' --dataset='ca'
```

## Pretrain US Market using the new algorithm
```bash
python pretrain2.py
```

## Finetune with hypertuning (for example with 'ca') using the new algorithm
```bash
python finetune2.py --config_files='properties/CATSR.yaml properties/market.yaml' --params_file='model.hyper' --output_file='hyper_example.result' --weight_path='saved/CATSR-us-200.pth' --tool='Hyperopt' --dataset='ca'
```
