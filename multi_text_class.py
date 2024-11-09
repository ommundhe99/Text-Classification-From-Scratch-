#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('labeled3.csv')

# Ensure all entries in the text column are strings
data['text'] = data['text'].astype(str)

# Handle any missing values (if any)
data.dropna(subset=['text', 'label'], inplace=True)

# Data Cleaning function
def clean_text(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    return text

# Apply the cleaning function to the text column
data['text'] = data['text'].apply(clean_text)

# Preprocessing
max_words = 20000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data['text'].values)
sequences = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(sequences, maxlen=max_len)

# Convert labels to categorical format
Y = pd.get_dummies(data['label']).values

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=64)

# Load GloVe embeddings
embedding_index = {}
with open('glove.6B.300d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

# Create an embedding matrix
embedding_dim = 300
word_index = tokenizer.word_index
num_words = min(max_words, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():
    if i >= max_words:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Model Implementation with Bidirectional LSTM and Dropout
model = Sequential()
model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))  # 4 classes for the 4 emotions

optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Training the Model
epochs = 10  
batch_size = 128

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), verbose=2)

# Evaluate the Model
Y_pred = model.predict(X_test, batch_size=batch_size, verbose=2)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

# Classification report
print(classification_report(Y_true, Y_pred_classes, target_names=['anger', 'happy', 'sadness', 'neutral']))

# Example to check the model
example_text = ["I am so happy and joyful today!"]
example_text_cleaned = [clean_text(text) for text in example_text]
example_sequences = tokenizer.texts_to_sequences(example_text_cleaned)
example_padded = pad_sequences(example_sequences, maxlen=max_len)

example_pred = model.predict(example_padded)
example_pred_class = np.argmax(example_pred, axis=1)

emotion_labels = ['anger', 'happy', 'sadness', 'neutral']
predicted_emotion = emotion_labels[example_pred_class[0]]

print(f"Example text: {example_text[0]}")
print(f"Predicted emotion: {predicted_emotion}")


# In[ ]:




