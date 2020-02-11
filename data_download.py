import numpy as np
import pandas as pd
import random
import sys
import requests
from bs4 import BeautifulSoup


def param_dict ():
    '''
    return: returns a dictionary of all the parameters required for download
    '''
    param_dict = {}

    param_a = {
    "db" : "pubmed",
    "term" : "hasabstract AND drug-related side effects and adverse reactions[mesh] NOT Congenital Abnormalities[mesh]",
    "RetMax":10000
    }
    param_b = {
    "db" : "pubmed",
    "term" : "hasabstract AND Congenital Abnormalities[mesh] NOT drug-related side effects and adverse reactions[mesh] ",
    "RetMax":10000
    }
    param_c = {
    "db" : "pubmed",
    "term" : "hasabstract AND drug-related side effects and adverse reactions[mh] AND Congenital Abnormalities[mh]",
    "RetMax":10000
    }
    param_d = {
    "db" : "pubmed",
    "term" : "hasabstract NOT (drug-related side effects and adverse reactions[mh] OR Congenital Abnormalities[mh])",
    "RetMax":10000
    }

    param_dict['A'] = param_a
    param_dict['B'] = param_b
    param_dict['C'] = param_c
    param_dict['D'] = param_d

    return param_dict

def get_ids(parameters):
    '''
    parameters: desired parameters for download
    return: the list of ids
    '''
    base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
    response = requests.get(base,params=parameters)
    parser = BeautifulSoup(response.content,'xml')
    ids = parser.find_all('Id')
    ids_comb = [td.get_text() for td in ids]
    #print('url = {}'.format(response.url))
    return ids_comb

def fetch_doc(id_list):
    '''
    id_list: the list of ids of class
    return: the list of abstract
    '''
    n = len(id_list)
    chunk = (id_list[i:i + 200] for i in range(0, n, 200))
    ids_comb = []
    for index in chunk:
        param_doc = {
            "db" : "pubmed",
            "id":','.join(index),
            "rettype":"abstract",
            "retmode":"xml"
            }
        base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
        response = requests.get(base,params=param_doc)
        parser = BeautifulSoup(response.content,'xml')
        ids = parser.find_all('Abstract')
        ids_comb += [td.get_text().strip() for td in ids]
    return ids_comb

def process (param, sample_size):
    '''
    cls: recieve class to download data for it
    return: sample_size number of abstracts for the input class
    '''
    id_list = get_ids(param)
    random.seed(6)
    list_sample = random.sample(id_list,k=sample_size)
    abstract = fetch_doc(list_sample)
    return abstract

def main(classes, param_dict):
    '''
    classes: All 4 classes that we want to download abstract from them
    output: saves all the abstracts into a csv file
    '''

    topics = ['Drug adverse events', 'Congenital anomalies']
    df_list = []
    sample_size = 700

    for cls in classes:
        print("Class {} is downloading ...".format(cls))
        if cls == 'A':
            topic = [topics[0]]
        elif cls == 'B':
            topic = [topics[1]]
        elif cls == 'C':
            topic = topics
        elif cls == 'D':
            #sample_size = 2 * sample_size
            topic = ['Others']
        abstract = process (param_dict[cls], sample_size)
        class_df = pd.DataFrame(list(zip(abstract, [topic]*sample_size)), columns=['abstract','topics'])
        print("Class {} was downloaded ...".format(cls))
        df_list.append(class_df)

    df = pd.concat(df_list)
    df.to_csv("data.csv",index=False)


if __name__ == "__main__":

    param_dict = param_dict()
    classes = ['A', 'B', 'C', 'D']
    main(classes, param_dict)
