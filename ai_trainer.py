from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization,Conv1D, MaxPooling2D, Flatten, GlobalMaxPool1D
from keras._tf_keras.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import pickle
import json

# Open and read the JSON file
with open('dataset.json', 'r') as file:
    data = json.load(file)

data = pd.DataFrame(data)
# Encode labels
label_encoder = LabelEncoder()
data['label_encoded'] = label_encoder.fit_transform(data['label'])
num_classes = len(label_encoder.classes_)
encoder_df = data[['label','label_encoded']].copy()
encoder_df = encoder_df.drop_duplicates()
with open('encoder_key.json', 'w') as file:
    encoder_df.to_json(file,orient="records", index=False)
# Tokenize text data
tokenizer = Tokenizer(num_words=1000)  # Limit vocabulary size
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
word_index = tokenizer.word_index
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# Pad sequences to ensure uniform input length
max_sequence_length = max(len(seq) for seq in sequences)

X = pad_sequences(sequences, maxlen=max_sequence_length)
y = to_categorical(data['label_encoded'], num_classes=num_classes)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
embedding_dim = 50

model = Sequential([
    Embedding(input_dim=1000, output_dim=embedding_dim, input_length=max_sequence_length),
    GlobalMaxPool1D(),
    Dense(100, activation='relu'),
    Dense(10, activation='relu'),
    Dense(num_classes, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# Compile the model
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=2,
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate model on validation data
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Save the trained model
model.save('text_classification_model.keras')
model.summary()
