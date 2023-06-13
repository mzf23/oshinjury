#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Define the input shape and parameters
maxlen = 200
embedding_dim = 100

# Create the input layers
input_1 = Input(shape=(maxlen,))
input_2 = Input(shape=(5,))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)

# Preprocess input_1 using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X1_train)
input_1_tfidf = tfidf_vectorizer.transform([input_1_text])
input_1_seq = pad_sequences(tokenizer.texts_to_sequences([input_1_text]), padding='post', maxlen=maxlen)

# Get the pre-trained GloVe embeddings
def get_glove_embeddings_matrix():
    # Load and return the pre-trained GloVe embeddings matrix
    # Replace this with your own code to load the pre-trained GloVe embeddings
    return np.zeros((len(tfidf_vectorizer.get_feature_names()), embedding_dim))

glove_embedding_matrix = get_glove_embeddings_matrix()

# Embedding layer for input_1_tfidf using GloVe embeddings
input_1_embed = Embedding(len(tfidf_vectorizer.get_feature_names()), embedding_dim,
                          weights=[glove_embedding_matrix], trainable=False)(input_1_tfidf)

# Dense layer for input_2
input_2_scaled = Dense(10)(input_2)

# Concatenate input_1_embed and input_2_scaled
concat_layer = concatenate([input_1_embed, input_2_scaled])

# LSTM layer for concatenated inputs
Bilstm_layer = Bidirectional(LSTM(256, return_sequences=False))(concat_layer)

# Additional layers
dropout_rate = 0.2
dropout_layer = Dropout(dropout_rate)(Bilstm_layer)
dense_layer = Dense(10, activation='relu')(dropout_layer)

# Output layer
output = Dense(1, activation='sigmoid')(dense_layer)

# Create the model
model = Model(inputs=[input_1, input_2], outputs=output)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
num_epochs = 25
model.fit(x=[X1_train, X2_train], y=y_train, batch_size=64, epochs=num_epochs, verbose=1, validation_split=0.2)

# Evaluate the model on the testing set
test_loss, test_accuracy = model.evaluate(x=[X1_test, X2_test], y=y_test, verbose=1)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

