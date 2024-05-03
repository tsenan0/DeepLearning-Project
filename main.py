################################
# File: main.py                #
# Purpose: File that trains    #
# and tests the model          #
################################
import pandas as pd
import numpy as np
## must use scipy version 1.10.1
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

## import the datasets
data_train = pd.read_csv("./data/train.csv")
data_test = pd.read_csv("./data/test.csv")

## preprocess the dataset
# drop the domain column
data_train.drop('domain', axis=1, inplace=True)
data_test.drop('domain', axis=1, inplace=True)

# make all scores 1 when formal, and 0 when not formal
data_train['avg_score'] = data_train['avg_score'].apply(lambda x: 1 if x>=0 else 0)
data_test['avg_score'] = data_test['avg_score'].apply(lambda x: 1 if x>=0 else 0)

# make word embeddings
model = Word2Vec(data_train['sentence'], min_count = 1)
weights = model.wv.vectors
vocab_len, embed_len = weights.shape

data_train['sentence'] = data_train['sentence'].apply(lambda x: [model.wv.get_vector(word) for word in x])

## train the model
maxlen = max(data_train['sentence'].apply(len))
data_train['sentence'] = [pad_sequences(data_train['sentence'], padding='post', maxlen=maxlen)]

x_train = np.stack(data_train['sentence'].values)
y_train = data_train['avg_score'].values

print(x_train)
print(y_train)

## test the model

## show results