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
$ python NLP_classification.py --task train
```
To evaluate the model performance the the following
```
$ python NLP_classification.py --task test
```
