# Abstract_Classifier

## Problem
Classify given set of Pubmed abstracts (biomedical literature abstracts) into four classes:
- Abstracts containing Drug adverse events
- Abstracts containing Congenital anomalies
- Abstracts containing both (a) and (b)
- Others

Dataset: Pubmed (https://pubmed.ncbi.nlm.nih.gov/)

## Required Libraries 
- python 3
- numpy 
- tenforflow
- keras
- sklearn
- pandas
- bs4
- requests
- matplotlib
- scipy

## Download Data
Code `data_download.py` will downlod all the required data in four classes
- Each class includes 700 examples
- Class other has two time more examples (1400) to keep all classes ballanced 
```
$ python data_download.py
```

## Train the model
To train the model run the following
```
$ python NLP_Classification.py --task train
```
To evaluate the model performance the the following
```
$ python NLP_Classification.py --task test
```
## Conclution 
There are a large number of possibility to train such a model, however the following are important to mention
- Generally Neural Netwrok based models performes better
- First a tokenizer generates arrays from text
- Second an Embedding layer generates array representation for each sequesnce 
- WE can use RNN (simple or LSTM), CNN or attention based models 
- My results is a one possibility

<p align="center"> <img src="https://github.com/nimahamidi/Abstract_Classifier/blob/master/Figure_1.png" width="70%"> </p>



