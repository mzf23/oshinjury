#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam, SGD, RMSprop

# Load the pre-trained GloVe model
glove_model = KeyedVectors.load_word2vec_format('path_to_glove_file', binary=False)

# Load your text data from a CSV file
data = pd.read_csv('text_data.csv')
text_data = data['text'].tolist()
labels = data['label'].tolist()

# Preprocess text data
def preprocess_text(sen):
    sentence = sen.lower()
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

preprocessed_texts = [preprocess_text(text) for text in text_data]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_texts, stratify=labels, random_state=42, test_size=0.2)

# Create TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Generate GloVe vectors
def generate_glove_vectors(texts):
    glove_vectors = []

    for text in texts:
        tokens = text.split()
        doc_vector = np.zeros((len(tokens), 100))

        for i, token in enumerate(tokens):
            if token in glove_model:
                doc_vector[i, :] = glove_model[token]

        doc_vector = np.mean(doc_vector, axis=0)
        glove_vectors.append(doc_vector)

    return np.array(glove_vectors)

X_train_glove = generate_glove_vectors(X_train)
X_test_glove = generate_glove_vectors(X_test)

# Define a function to create the Bi-LSTM model with hyperparameters
def create_model(units=128, dense_units=10, dropout=0.2, batch_size=32, epochs=20, activation='relu', optimizer='adam'):
    model = keras.Sequential()
    model.add(Embedding(input_dim=X_train_tfidf.shape[1], output_dim=100, input_length=X_train_tfidf.shape[1]))
    model.add(Bidirectional(LSTM(units, return_sequences=True)))
    model.add(Bidirectional(LSTM(units)))
    model.add(Dropout(dropout))
    model.add(Dense(dense_units, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Define hyperparameter space
param_dist = {
    'units': [128, 256, 512],
    'dense_units': [10, 20, 30],
    'dropout': [0.2, 0.3, 0.4],
    'batch_size': [32, 64, 128],
    'epochs': [20, 25, 30],
    'activation': ['relu', 'tanh', 'sigmoid'],
    'optimizer': [Adam(), SGD(), RMSprop()]
}

# Wrap the Keras model with scikit-learn's KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=0)

# Create RandomizedSearchCV instance
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=10)

# Fit the model to the data
random_search.fit(X_train_tfidf, y_train)

# Print the best hyperparameters and corresponding accuracy
print("Best Hyperparameters: ", random_search.best_params_)
print("Best Accuracy: ", random_search.best_score_)

# Optionally, you can get the best model and evaluate it on your test data
best_model = random_search.best_estimator_.model
y_pred = best_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy: ", accuracy)

