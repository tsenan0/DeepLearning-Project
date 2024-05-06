################################
# File: main.py                #
# Purpose: File that trains    #
# and tests the model          #
################################
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer 
from gensim.models import Word2Vec
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, Flatten, Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras import regularizers
import os

word2vec_model = None
maxlen = None

# Function to ask the user for their choice
def user_choice():
  choice = input("Do you want to 'train' the model or 'use' the model? (train/use): ").lower()
  while choice not in ['train', 'use']:
    choice = input("Invalid input. Please type 'train' to train the model or 'use' to use the model: ").lower()
  return choice

# Function to save the model
def save_model(model):
  model.save('trained_model.keras')
  print("Model saved to 'trained_model.keras'")

# Function to load the model
def load_trained_model():
  if os.path.exists('trained_model.keras'):
    return load_model('trained_model.keras')
  else:
    print("No trained model found. Please train the model first.")
    exit()

# Function to test the model with user input
def test_with_input(model):
  user_input = input("Enter your sentence to test: ")
  # Preprocess and convert user input to the same format as the training data
  processed_input = preprocess_user_input(user_input)
  prediction = model.predict(processed_input)
  print(f"The model predicts: {'Formal' if prediction[0][0] >= 0.5 else 'Not Formal'}")
  print(f"Prediction: {prediction[0][0]}")

# Function to preprocess user input
def preprocess_user_input(user_input):
  # Initialize and fit the tokenizer
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts([user_input])

  # Convert sentences to sequences of integers
  sequences = tokenizer.texts_to_sequences([user_input])

  # Pad the sequences
  padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')

  # Convert sequences to word vectors
  word_vectors = np.zeros((1, maxlen, word2vec_model.vector_size))
  for i, seq in enumerate(padded_sequences[0]):
      if seq != 0:
          token = tokenizer.index_word[seq]
          if token in word2vec_model.wv.key_to_index:
              word_vectors[0, i] = word2vec_model.wv.get_vector(token)

  return word_vectors
  
def main():
  global x_test, y_test, word2vec_model, maxlen

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
  word2vec_model = Word2Vec(data_train['sentence'], min_count = 1)
  data_train['sentence'] = data_train['sentence'].apply(lambda x: [word2vec_model.wv.get_vector(word) for word in x])
  data_test['sentence'] = data_test['sentence'].apply(lambda x: [word2vec_model.wv.get_vector(word) for word in x if word in word2vec_model.wv.key_to_index])

  ## train the model
  maxlen = max(data_train['sentence'].apply(len))
  padded_sequences_train = pad_sequences(data_train['sentence'], padding='post', maxlen=maxlen, dtype='float32')
  padded_sequences_test = pad_sequences(data_test['sentence'], padding='post', maxlen=maxlen, dtype='float32')

  # Reshape the data to be 3D
  x_train = np.stack(padded_sequences_train)
  x_test = np.stack(padded_sequences_test)

  # Ensure the target variable is a numpy array
  y_train = data_train['avg_score'].values
  y_test = data_test['avg_score'].values

  user_decision = user_choice()
  
  if user_decision == 'train':

  
    # Create a Keras model
    # Define the model
    model = Sequential()

    # Add an LSTM layer
    model.add(LSTM(128, return_sequences=True))

    # Add a Dropout layer for regularization
    model.add(Dropout(0.5))

    # Add another LSTM layer
    model.add(LSTM(64))

    # Add a Dense layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
  
    # Train the model
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    save_model(model)
  elif user_decision == 'use':
    model = load_trained_model()
    
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

if __name__ == "__main__":
  main()