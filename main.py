################################
# File: main.py                #
# Purpose: File that trains    #
# and tests the model          #
################################
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import confusion_matrix, classification_report
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
data_train['sentence'] = data_train['sentence'].apply(lambda x: [model.wv.get_vector(word) for word in x])
data_test['sentence'] = data_test['sentence'].apply(lambda x: [model.wv.get_vector(word) for word in x if word in model.wv.key_to_index])

## train the model
maxlen = max(data_train['sentence'].apply(len))
padded_sequences_train = pad_sequences(data_train['sentence'], padding='post', maxlen=maxlen)
padded_sequences_test = pad_sequences(data_test['sentence'], padding='post', maxlen=maxlen)

x_train = np.stack(padded_sequences_train)
y_train = data_train['avg_score'].values

x_test = np.stack(padded_sequences_test)
y_test = data_test['avg_score'].values

# Create a Keras model
model = Sequential()
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy*100}')

# get the model's predictions on the test set
y_pred = model.predict(x_test)
y_pred = np.round(y_pred)  #round the predictions to 0 or 1

# generate a confusion matrix, testing our classifier to see how many right and wrong predictions it made
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# generate a classification report, report of each precision, recall, f1-score and support
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)