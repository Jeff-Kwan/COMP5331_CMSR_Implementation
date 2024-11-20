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


### 3 Process data
```
cd data
python data_process.py
```

### 4 Process downstream datasets

```bash
cd data
python UnisRec_data_process.py
python process_unis_atomic.py
```

### 5. Pretrain ```us``` market

For SASRec-bert and UniSRec-bert:
Pretrain:
1. Adjust the model_name  and the dataset in pretrain_dist.sh for distrubuted training
2. For distrubuted training, Run:
    ```
    bash pretrain_dist.sh  
    ```
    For Single GPU:
    ```
    python pretrain_baseline.py --model_name=UniSRec
    ```

    

### 6. Fine-tune
```
bash finetune_baseline.sh
```

### 7. Run CDR Baselines

1. Create a symbolic link to the parent directory in the CDR directory, make sure the rating column contains in the processed data.
```
cd CDR
ln -s ../dataset .
```
2. Run the CDR experiment
```
./train_cdr.sh
```
### 8. Run CMR Baselines

1. Create a symbolic link to the parent directory in the CMR directory, make sure the rating column contains in the processed data.
```
cd CMR
ln -s ../dataset .
```
2. Run the CMR experiment
```
./train_cmr.sh
```


### 9. File discription

- Directory `data`:  
    - `data_process.py`:
        The Processing file for generating data files (i.e., .inter, .item) used in CAT-SR, SASRec, and S3Rec.  
    - `process_unis_atomic.py`:
        The Processing file for generating atomic data files used  in UniSRec.  
    - `UnisRec_data_process.py`:
        The Processing file for generating data files used (i.e., .inter, .item, .feat1CLS etc.) in UniSRec.  
    - `utils.py`: 
        some utils for data_process.py  

- Directory `properties`:  
    - The configuration files for baseline1 have file names with the model name as the prefix and a unified suffix of `.yaml`.  

- Directory `CDR`:
    - `run_recbole_cdr.py`:
        The file to run the CDR experiment.
    - `properties`:
        The configuration files for CDR have file names with the model name as the prefix, including DCDCSR, NATR and SSCDR and a unified suffix of `.yaml`.
    - `recbole_cdr`: Adapted from RecBole, the file defines the CDR dataset, dataloader, config, model and trainer, since cross domain model uses source and target domain data which is different from the original RecBole pipeline.

- Directory `UniSRec_model`:
    - `config.py`:
        Self-defined Config class for some RecBole configuration.
    - `dataloader.py`:
        Two self-defined dataloader class for UnisRec's dataloader
    - `dataset.py`:
        Two self-defined dataset class to adapt to UnisRec model
    - `transform.py`:
        A File for PLM embedding transform in dataLoader.
    - `UnisRec.py`:
        The file define the model structure of UniSRec.


- `finetune_baseline.py`:
    The file implement the finetune method for the models in baseline 1.
- `finetune_baseline.sh`:
    A shell script file defines the parameters for running finetune.py

- `pretrain_baseline.py`:
    The file implement the pretrain method for the models in baseline 1, only using one gpu.  
- `pretrain_dist.py`:
    The file implement the pretrain method for the models in baseline 1, using multiple gpus.  
- `pretrain_dist.sh`:
    A shell script file defines the parameters for running pretrain_dist.py.  
- `S3Rec.py`:
    The file define the model structure of S3Rec.
- `SASRec.py`:
    The file define the model structure of SASRec.

### 10. Environments
1. Single Domain Models
    - Operating System: **Ubuntu 20.04.5 LTS**
    - python version: **3.11.10**
    - Dependencies:
    ```
    absl-py                  2.1.0
    certifi                  2024.8.30
    charset-normalizer       3.4.0
    colorama                 0.4.4
    colorlog                 4.7.2
    filelock                 3.16.1
    fsspec                   2024.10.0
    grpcio                   1.67.0
    huggingface-hub          0.26.1
    idna                     3.10
    Jinja2                   3.1.4
    joblib                   1.4.2
    kmeans-pytorch           0.3
    Markdown                 3.7
    MarkupSafe               3.0.2
    mpmath                   1.3.0
    networkx                 3.4.2
    numpy                    1.26.4
    nvidia-cublas-cu12       12.4.5.8
    nvidia-cuda-cupti-cu12   12.4.127
    nvidia-cuda-nvrtc-cu12   12.4.127
    nvidia-cuda-runtime-cu12 12.4.127
    nvidia-cudnn-cu12        9.1.0.70
    nvidia-cufft-cu12        11.2.1.3
    nvidia-curand-cu12       10.3.5.147
    nvidia-cusolver-cu12     11.6.1.9
    nvidia-cusparse-cu12     12.3.1.170
    nvidia-nccl-cu12         2.21.5
    nvidia-nvjitlink-cu12    12.4.127
    nvidia-nvtx-cu12         12.4.127
    packaging                24.1
    pandas                   2.2.3
    pip                      24.2
    plotly                   5.24.1
    protobuf                 5.28.3
    python-dateutil          2.9.0.post0
    pytz                     2024.2
    PyYAML                   6.0.2
    recbole                  1.2.0
    regex                    2024.9.11
    requests                 2.32.3
    safetensors              0.4.5
    scikit-learn             1.5.2
    scipy                    1.14.1
    setuptools               75.1.0
    six                      1.16.0
    sympy                    1.13.1
    tabulate                 0.9.0
    tenacity                 9.0.0
    tensorboard              2.18.0
    tensorboard-data-server  0.7.2
    texttable                1.7.0
    thop                     0.1.1-2209072238
    threadpoolctl            3.5.0
    tokenizers               0.20.1
    torch                    2.5.0
    tqdm                     4.66.5
    transformers             4.46.0
    triton                   3.1.0
    typing_extensions        4.12.2
    tzdata                   2024.2
    urllib3                  2.2.3
    Werkzeug                 3.0.6
    wheel                    0.44.0
    ```

2. Cross domain models
    - Operating System: **Ubuntu 20.04.5 LTS**
    - python version: **3.8.20**
    - Dependencies:
    ```
    absl-py                  2.1.0
    cachetools               5.5.0
    certifi                  2024.8.30
    charset-normalizer       3.4.0
    colorama                 0.4.4
    colorlog                 4.7.2
    filelock                 3.16.1
    fsspec                   2024.10.0
    google-auth              2.36.0
    google-auth-oauthlib     1.0.0
    grpcio                   1.67.1
    idna                     3.10
    importlib_metadata       8.5.0
    Jinja2                   3.1.4
    joblib                   1.4.2
    Markdown                 3.7
    MarkupSafe               2.1.5
    mpmath                   1.3.0
    networkx                 3.1
    numpy                    1.24.4
    nvidia-cublas-cu12       12.1.3.1
    nvidia-cuda-cupti-cu12   12.1.105
    nvidia-cuda-nvrtc-cu12   12.1.105
    nvidia-cuda-runtime-cu12 12.1.105
    nvidia-cudnn-cu12        9.1.0.70
    nvidia-cufft-cu12        11.0.2.54
    nvidia-curand-cu12       10.3.2.106
    nvidia-cusolver-cu12     11.4.5.107
    nvidia-cusparse-cu12     12.1.0.106
    nvidia-nccl-cu12         2.20.5
    nvidia-nvjitlink-cu12    12.6.77
    nvidia-nvtx-cu12         12.1.105
    oauthlib                 3.2.2
    pandas                   2.0.3
    pip                      24.3.1
    protobuf                 5.28.3
    pyasn1                   0.6.1
    pyasn1_modules           0.4.1
    python-dateutil          2.9.0.post0
    pytz                     2024.2
    PyYAML                   6.0.2
    recbole                  1.0.1
    requests                 2.32.3
    requests-oauthlib        2.0.0
    rsa                      4.9
    scikit-learn             1.3.2
    scipy                    1.6.0
    setuptools               75.3.0
    six                      1.16.0
    sympy                    1.13.3
    tensorboard              2.14.0
    tensorboard-data-server  0.7.2
    threadpoolctl            3.5.0
    torch                    2.4.1
    tqdm                     4.67.0
    triton                   3.0.0
    typing_extensions        4.12.2
    tzdata                   2024.2
    urllib3                  2.2.3
    Werkzeug                 3.0.6
    wheel                    0.45.0
    zipp                     3.20.2
    ```
3. Cross market models
- Operating System: **Ubuntu 20.04.5 LTS**
    - python version: **3.8.20**
    - Dependencies:
    ```
    appnope==0.1.2
    argcomplete==2.0.0
    argon2-cffi==21.3.0
    argon2-cffi-bindings==21.2.0
    attrs==21.4.0
    backcall==0.2.0
    bleach==4.1.0
    boto==2.49.0
    certifi==2021.10.8
    cffi==1.15.0
    charset-normalizer==2.0.10
    cloudpickle==2.0.0
    crcmod==1.7
    cryptography==36.0.1
    cycler==0.11.0
    debugpy==1.5.1
    decorator==5.1.0
    defusedxml==0.7.1
    entrypoints==0.3
    fasteners==0.16.3
    fonttools==4.28.5
    gcs-oauth2-boto-plugin==3.0
    google-apitools==0.5.32
    google-reauth==0.1.1
    gsutil==5.5
    gym==0.21.0
    httplib2==0.20.2
    idna==3.3
    importlib-metadata==4.10.0
    importlib-resources==5.4.0
    ipykernel==6.6.1
    ipython==7.31.0
    ipython-genutils==0.2.0
    ipywidgets==7.6.5
    jedi==0.18.1
    Jinja2==3.0.3
    joblib==1.1.0
    jsonschema==4.3.3
    jupyter==1.0.0
    jupyter-client==7.1.0
    jupyter-console==6.4.0
    jupyter-core==4.9.1
    jupyterlab-pygments==0.1.2
    jupyterlab-widgets==1.0.2
    kiwisolver==1.3.2
    learn2learn==0.1.6
    MarkupSafe==2.0.1
    matplotlib==3.5.1
    matplotlib-inline==0.1.3
    mistune==0.8.4
    monotonic==1.6
    nbclient==0.5.9
    nbconvert==6.4.0
    nbformat==5.1.3
    nest-asyncio==1.5.4
    notebook==6.4.6
    numpy==1.21.4
    oauth2client==4.1.3
    packaging==21.3
    pandas==1.3.4
    pandocfilters==1.5.0
    parso==0.8.3
    pexpect==4.8.0
    pickleshare==0.7.5
    Pillow==9.0.0
    prometheus-client==0.12.0
    prompt-toolkit==3.0.24
    ptyprocess==0.7.0
    pyasn1==0.4.8
    pyasn1-modules==0.2.8
    pycparser==2.21
    Pygments==2.11.1
    pyOpenSSL==21.0.0
    pyparsing==3.0.6
    pyrsistent==0.18.0
    python-dateutil==2.8.2
    pytrec-eval==0.5
    pytz==2021.3
    pyu2f==0.1.5
    pyzmq==22.3.0
    qpth==0.0.15
    qtconsole==5.2.2
    QtPy==2.0.0
    requests==2.27.0
    retry-decorator==1.1.1
    rsa==4.7.2
    scikit-learn==1.0.2
    scipy==1.7.3
    Send2Trash==1.8.0
    six==1.16.0
    terminado==0.12.1
    testpath==0.5.0
    threadpoolctl==3.0.0
    torch==1.10.0
    torchvision==0.11.1
    tornado==6.1
    tqdm==4.62.3
    traitlets==5.1.1
    typing-extensions==4.0.1
    urllib3==1.26.7
    wcwidth==0.2.5
    webencodings==0.5.1
    widgetsnbextension==3.5.2
    zipp==3.7.0
```

