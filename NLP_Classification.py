'''-----------------------------------
           Data Prepration
------------------------------------'''
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow.keras.utils as ku

def read_data(path):
    '''
    path: name of the csv file including all the downloaded data
    return: returns data line by line and one single corpus and labels
    '''
    corpus = []
    labels = []
    data = pd.read_csv(path)
    data['topics'] = data['topics'].apply(literal_eval)
    for line, label in zip (data['abstract'], data['topics']):
        corpus.append(line)
        labels.append(label)
    return data, corpus, labels

def data_prep (path):
    '''
    path: name of the csv file including all the downloaded data
    return: prepares and splits data into test (holdout) and train
    '''
    data, corpus, labels = read_data(path)

    # make labels into multi-label binary form [i, i, i] -> i (0 or 1)
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)

    # tokenizing abstracts into array of integers
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    # create input sequences using list of tokens
    input_sequences = []
    for line in data['abstract']:
        token_list = tokenizer.texts_to_sequences([line])[0]
        input_sequences.append(token_list)

    # pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='post', truncating='post'))

    X_train, X_test, y_train, y_test = train_test_split(input_sequences,
                                                        labels,
                                                        stratify=labels,
                                                        random_state=42,
                                                        test_size=0.1, shuffle=True)

    return X_train, X_test, y_train, y_test, total_words

'''-----------------------------------
          Building models
-----------------------------------'''
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
import matplotlib.pyplot as plt
import numpy as np

tokenizer = Tokenizer()

def clasifier (total_words, model_type):
    '''
    total_words: the size of our dictionary
    model_type: the code provides several models such as
                just dense layer, with convolution, with LSTM, with attentio
    returns: based on type the function returns the built model
    '''

    if model_type == 'basic':
        model = Sequential()
        model.add(Embedding(total_words, 100))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(3, activation='sigmoid'))

    elif model_type == 'CNN':
        model = Sequential()
        model.add(Embedding(total_words, 100))
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(3, activation='sigmoid'))

    return model

'''-----------------------------------
          Training models
-----------------------------------'''

def train(model_type):
    '''
    model_type: the code provides several models such as
                just dense layer, with convolution, with LSTM, with attention
    output: writes the trained model into weight.h5
    '''

    X_train,_ , y_train,_ , total_words = data_prep('data.csv')

    split = int(len(X_train)*0.9)
    partial_X_train = X_train[:split]
    partial_y_train = y_train[:split]
    X_val = X_train[split:]
    y_val = y_train[split:]

    model = clasifier (total_words, model_type)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()

    model_weights = 'weights.h5'
    call_back = ModelCheckpoint(model_weights,
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=True)

    history = model.fit(partial_X_train,
                        partial_y_train,
                        epochs=40,
                        callbacks=[call_back],
                        validation_data=(X_val, y_val),
                        verbose=1)

'''-----------------------------------
          Evaluating models
-----------------------------------'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import sys
import argparse

def eveluate(model_type):
    '''
    model_type: the code provides several models such as
                just dense layer, with convolution, with LSTM, with attention
    output: plots ROC curve for micro and macro averaging
    '''
    n_classes = 3
    model_weights = 'weights.h5'
    _, X_test,_ , y_test, total_words = data_prep('data.csv')
    model = clasifier (total_words, model_type)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.load_weights(model_weights)
    y_score = model.predict(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(0,3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of ROC to multi-class')
    plt.legend(loc="lower right")
    plt.show()

'''------------------------
        main function
------------------------'''
def main (task, model_type):
    '''
    task: required task including train, test and evaluation
    model_type: the code provides several models such as
                just dense layer, with convolution, with LSTM, with attention
    output: calls the intended task
    '''
    if task == "train":
        train(model_type)
    elif task == "test":
        eveluate(model_type)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='traing/testing/evaluation', required=True)
    args = parser.parse_args()
    task = args.task
    main(task, 'CNN')
