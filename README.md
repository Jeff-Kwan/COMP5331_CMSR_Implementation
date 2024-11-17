# COMP5331_CMSR_Implementation
Implementation &amp; Further Study Based on "Pre-Training with Transferable Attention for Addressing Market Shifts in Cross-Market Sequential Recommendation" by Wang et al. 2024

### 1. Copy preprocessed XMRec dataset from [FOREC](https://github.com/hamedrab/FOREC/tree/main/DATA/proc_data) or [MA](https://github.com/samarthbhargav/efficient-xmrec/tree/main/DATA2/proc_data)
Put data file into ```data``` directory. For example: ```data/ca_5core.txt```

### 2. Download [Amazon meta dataset](https://nijianmo.github.io/amazon/index.html)
Category: Electronics

For data process of S3Rec/SASRec/CATSR:
    Data: metadata

    Put dataset into ```data/Amazon/metadata``` directory. For example ```data/Amazon/metadata/meta_Electronics.json.gz```
For data process of UniSRec:
    Data: metadata,ratings

    ```
    data/
    raw/
        Metadata/
        meta_Electronics.json.gz
        Ratings/
        Electronics.csv
    ```


### 3. Process data
```
cd data
python data_process.py
```

### 2. Process downstream datasets

```bash
cd data
python UnisRec_data_process.py
python process_unis_atomic.py
```

### 4. Pretrain ```us``` market

For SASRec-bert and UniSRec-bert:
Pretrain:
1. Adjust the model_name  and the dataset in pretrain_dist.sh for distrubuted training
2. For distrubuted training, Run:
    ```
    bash pretrain_dist.sh  
    ```
    For Single GPU:
    ```
    python pretrain.py --model_name=UniSRec
    ```

    

### 5. Fine-tune
```
bash finetune.sh
```
### 6. File discription
- Directory CDR:

- Directory data:  
    - data_process.py:
        The Processing file for generating data files (i.e., .inter, .item) used in CAT-SR, SASRec, and S3Rec.  
    - process_unis_atomic.py:
        The Processing file for generating atomic data files used  in UniSRec.  
    - UnisRec_data_process.py:
        The Processing file for generating data files used (i.e., .inter, .item, .feat1CLS etc.) in UniSRec.  
    - utils.py: 
        some utils for data_process.py  

- Directory properties:  
    - The configuration files for baseline1 have file names with the model name as the prefix and a unified suffix of .yaml.  


- Directory UniSRec_model:
    - config.py:
        Self-defined Config class for some RecBole configuration.
    - dataloader.py:
        Two self-defined dataloader class for UnisRec's dataloader
    - dataset.py:
        Two self-defined dataset class to adapt to UnisRec model
    - transform.py:
        A File for PLM embedding transform in dataLoader.
    - UnisRec.py:
        The file define the model structure of UniSRec.


- finetune_baseline.py:
    The file implement the finetune method for the models in baseline 1.
- finetune_baseline.sh:
    A shell script file defines the parameters for running finetune.py

- pretrain_baseline.py:
    The file implement the pretrain method for the models in baseline 1, only using one gpu.  
- pretrain_dist.py:
    The file implement the pretrain method for the models in baseline 1, using multiple gpus.  
- pretrain_dist.sh:
    A shell script file defines the parameters for running pretrain_dist.py.  
- S3Rec.py:
    The file define the model structure of S3Rec.
- SASRec.py:
    The file define the model structure of SASRec.

### 7. Operating System
    Linux




