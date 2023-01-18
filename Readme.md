# Installation

`pip install -r requirements.txt`

# Download Pre-trained Models

best_model_pooling_512

https://drive.google.com/drive/folders/1fFjncBNo3-2dmJjz3CwySWB3XH288BQ1?usp=sharing

best_model_pooling_768_pca_512

https://drive.google.com/drive/folders/1_eJ9qqbl_KpA8vfAEtjN3xVSPu47beAP?usp=sharing

# How to Run Demo

`python demo.py`

You can play around with commented out lines and sample inputs.

# How to Train Only Transformer Model

`python train_transformer.py`

# How to Train Transformer + PCA Model

`python train_transformer_pca.py`

# How to Evaluate a Model
    
`python test.py`

    usage: test.py [-h] [-m MODEL] [-d DATASET] [--device DEVICE]
    
    options:
      -h, --help            show this help message and exit
      -m MODEL, --model MODEL
      -d DATASET, --dataset DATASET
      --device DEVICE
