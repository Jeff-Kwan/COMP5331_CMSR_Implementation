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
Take finetune Canada(ca) as an example
```
bash finetune.sh