import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import tensorflow as tf
import joblib

# Load data
def load_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# Load the dataset
data = load_data("dataset.json")
df = pd.DataFrame(data)

# Separate features and labels
X = df['text']
y = df['label']

# Vectorize text data
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X).toarray()

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.4, random_state=42)

# Build TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(set(y_encoded)), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")

keras_model_path = "my_model.keras"
model.save(keras_model_path)
print(f"Model saved in .keras format at '{keras_model_path}'")


converter = tf.lite.TFLiteConverter.from_keras_model(model)


converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]


converter.experimental_enable_resource_variables = True


converter._experimental_lower_tensor_list_ops = False


tflite_model = converter.convert()


tflite_model_path = "new_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
print(f"Model saved in TensorFlow Lite format at '{tflite_model_path}'")
# Save the TensorFlow Lite model
with open("text_classifier_model.tflite", "wb") as f:
    f.write(tflite_model)
print("Model successfully converted to TFLite format.")

# Save the vectorizer and label encoder for inference
joblib.dump(vectorizer, "text_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("Vectorizer and label encoder saved.")

# Function for TFLite inference
def classify_text(input_text):
    # Load the vectorizer and label encoder
    loaded_vectorizer = joblib.load("text_vectorizer.pkl")
    loaded_label_encoder = joblib.load("label_encoder.pkl")

    # Vectorize input text and convert to float32
    input_vectorized = loaded_vectorizer.transform([input_text]).toarray().astype('float32')

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="text_classifier_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_vectorized)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = loaded_label_encoder.inverse_transform([predictions.argmax()])
    return predicted_label[0]

# Main script
if __name__ == "__main__":
    new_text = input("Enter a sentence to classify: ")
    result = classify_text(new_text)
    print(f"Predicted Label: {result}")