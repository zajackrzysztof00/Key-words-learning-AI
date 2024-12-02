import numpy as np
import pandas as pd
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
import json

# Open and read the JSON file
with open('testing_data.json', 'r') as file:
    data = json.load(file)
    
test_texts = data['text']
# Load the trained model
model = load_model('text_classification_model.keras')

# Recreate the Tokenizer and Label Encoder used in training
# Normally you would save these during training, but here we're creating them for testing.
# Use the same tokenizer configuration as during training.
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Tokenize and pad test texts
test_sequences = tokenizer.texts_to_sequences(test_texts)
max_sequence_length = 15  # This should match the max length used during training
test_data_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Make predictions
predictions = model.predict(test_data_padded)
predicted_classes = np.argmax(predictions, axis=1)
labels = ['class1', 'class2']
plt.plot(predicted_classes)
plt.show()
print(predicted_classes)
for text, predicted_class in zip(test_texts,predicted_classes):
    print(f"Text: {text} \nPredicted Label: {labels[predicted_class]}\n")
