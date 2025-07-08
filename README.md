# MASTER: A Multi-modal Foundation Model For Human Activity Recognition
This is a repo for Ubicomp/IMWUT 2025 paper: " <a href="!!!!!our paper link here!!!!!"> MASTER: A Multi-modal Foundation Model For Human Activity Recognition </a>".

# Update timeline
* [x] 2025.7 Upload a simple vision of MASTER which can run on uci dataset
* [ ] other dataset configs and running example
* [ ] multi-dataset running example


# Paper Overview
Coming soon

# Project Strcuture
```
|--config_files // Folder for storing running settings for different datasets
    |--uci-Configs.py //running settings for uci dataset

|--data // Folder for storing intermediate processed data
    |--uci  //uci dataset
        |--train_0.01
            |--train_label.pt //divided labeled train data at label rate 0.01
            |--train_unlabel.pt //divided unlabeled train data at label rate 0.01
        |--train_0.2
            |--train_label.pt //divided labeled train data at label rate 0.2
            |--train_unlabel.pt //divided unlabeled train data at label rate 0.2
        |--test.py  //all test data
        |--train.py //all train data

|--data_preprocessing // Folder of data preprocessing file
    |--UCI HAR Dataset //original uci dataset download from https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
    |--preprocess_uci.py // data preprocessing file of uci dataset

|--modality_process // Folder of modality files, each one contains a preprocess function and a feature encoder
    |--__init__.py
    |--IMU.py
    |--lidar.py
    |--mmwave.py
    |--skeleton.py
    |--wificsi.py

|--models // Folder of implementation of modules in MASTER, you can read them in the following order
    |--model.py // model definition
    |--feature_extraction.py // a module that cut input data into patches and then extracts features
    |--embedding.py // a module that add position embeddings and modality embeddings to feature patches
    |--self_learning.py // a module that integrate tokens of all modalities into one sentence, do mask language modeling and transformer computing
    |--transformer.py // our implementation of transformer
    |--recoverlayer.py // the recover layer used in self_learning
    |--predictor.py // the prediction layer used in fine_tune
    |--train_loss.py //our definition of loss computing

|--dataloader.py // The code file of dataloader
|--main.py // Main code file, just run it
|--train.py // The code file of Trainer
|--utils.py // Some defined utility functions
```

# Running Requirements
The code has been tested in the following environment:  
```
numpy 1.24.3  
pandas 2.0.3  
scikit-learn 1.3.0  
torch 1.12.1  
einops 0.6.1  
tqdm 4.66.1
```

# Quick Start
* We provide all running related files for the UCI dataset in the initial version of the uploaded code.
* You can run this to train MASTER by self supervised learning with unlabeled data(80% of all train data):
    ```bash
    python main.py --training_mode s --selected_dataset uci --label_rate 0.2 --seed 123 --device cuda --cuda_no 0 --experiment_description test --run_description uci
    ```
* Then run the following code to fine tune MASTER on labeled data(1% of all train data):
    ```bash
    python main.py --training_mode f --selected_dataset uci --label_rate 0.01 --seed 123 --device cuda --cuda_no 0 --experiment_description test --run_description uci
    ```
* In addition to the two training modes mentioned above, you can also explore align_learning mode and align_miss_modality mode on your own.

# Run on Other Datasets or Multi-Datasets
* If you want to run MASTER on other datasets, you need to provide all the following files:
  * configs -- see uci_Configs.py
  * data -- see preprocess_uci for an example
  * modality_process -- if you need to add new modalities, don't forget to change feature_extraction.py too
* You can also replace the feature encoder of each modality with your preferred version.


* If you want to run MASTER on multi-datasets, our dataloader cat support. 
* You only need to provide multiple datasets in the format of Python List to our Load_Dataset function.

# Repositories Related to This Project
This repository implement based on the following repository:  
<a href="https://github.com/emadeldeen24/TS-TCC">Time-Series Representation Learning via Temporal and Contextual Contrasting</a>  
Thanks very much for its open-source code contribution.

We are also very grateful for the following repositories of baseline methods mentioned in the paper and give a link to them below:  
<a href="https://github.com/xmouyang/Cosmo"> Cosmo: Contrastive Fusion Learning with Small Data for Multimodal Human Activity Recognition</a>  
<a href="https://github.com/wdkhuans/STMAE"> Spatial-Temporal Masked Autoencoder for Multi-Device Wearable Human Activity Recognition</a>  
<a href="https://github.com/I-ESC/Project-Babel"> Babel: A Scalable Pre-trained Model for Multi-Modal Sensing via Expandable Modality Alignment</a>   
<a href="https://github.com/maxxu05/relcon"> RelCon: Relative Contrastive Learning for a Motion Foundation Model for Wearable Data</a>  
In our paper, we do experiments of attnsense and CMC models on the implementation by the authors of Cosmo.  
(And at the time of writing our paper, the repository of Babel had not yet been made public,  
so the relevant experiments of the BABEL model in our paper were based on our own implementation version)

# Citation
The code of this project are made available for non-commercial, academic research only. If you would like to use the code of this project, please cite the following paper:
```
!!!!!our citation here!!!!!
```
    
