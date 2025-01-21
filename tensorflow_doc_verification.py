import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import json

# Step 1: Load and Prepare Dataset
with open("dataset.json", "r") as file:
    data = json.load(file)

# Extract texts and labels
texts = [item["text"] for item in data]
labels = [item["label"] for item in data]

# Encode labels
label_map = {label: idx for idx, label in enumerate(set(labels))}
y = np.array([label_map[label] for label in labels])

# Tokenize texts
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=50, padding="post")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Build the Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=50),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(len(label_map), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Step 3: Train the Model
print("Training the model...")
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 4: Convert the Model to TFLite Format
print("Converting model to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = "text_classification_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved as '{tflite_model_path}'")

# Step 5: Test the TFLite Model
print("Testing the TFLite model...")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input for testing
input_text = "Can you find the medical reports?"
input_sequence = tokenizer.texts_to_sequences([input_text])
input_sequence = pad_sequences(input_sequence, maxlen=50, padding="post")

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_sequence)

# Run inference
interpreter.invoke()

# Get predictions
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_label = list(label_map.keys())[np.argmax(output_data)]
print(f"Input Text: {input_text}")
print(f"Predicted Label: {predicted_label}")