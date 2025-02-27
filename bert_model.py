import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split


def load_data(json_file):
    """Loads dataset from a JSON file."""
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data


# Load dataset
data = load_data("dataset.json")
df = pd.DataFrame(data)

# Ensure labels are correctly mapped to integers
label_mapping = {"DOCUMENT_RETRIEVAL": 0, "CONVERSATION": 1}
df["label"] = df["label"].map(label_mapping)

# Map unique doc types to integer values
doc_types = df["doc_type"].unique()
doc_type_mapping = {doc: idx for idx, doc in enumerate(doc_types)}
df["doc_type"] = df["doc_type"].map(doc_type_mapping)

# Train-test split
train_texts, test_texts, train_labels, test_labels, train_doc_types, test_doc_types = train_test_split(
    df["text"].tolist(), df["label"].tolist(), df["doc_type"].tolist(), test_size=0.2, random_state=42
)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(
    pd.DataFrame({"text": train_texts, "label": train_labels, "doc_type": train_doc_types}))
test_dataset = Dataset.from_pandas(pd.DataFrame({"text": test_texts, "label": test_labels, "doc_type": test_doc_types}))

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    """Tokenizes text data with proper truncation and padding."""
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


# Tokenize datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Load pre-trained model
num_labels = len(label_mapping)  # Number of labels (DOCUMENT_RETRIEVAL, CONVERSATION)
num_doc_types = len(doc_type_mapping)  # Number of document types

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                            num_labels=num_labels + num_doc_types)

# Ensure PyTorch uses GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Use data collator for automatic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

if __name__ == "__main__":
    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

    # Save model, tokenizer, and mappings
    model.save_pretrained("./distilbert_doc_retrieve_classifier")
    tokenizer.save_pretrained("./distilbert_doc_retrieve_classifier")

    with open("./distilbert_doc_retrieve_classifier/label_mapping.json", "w") as f:
        json.dump(label_mapping, f)

    with open("./distilbert_doc_retrieve_classifier/doc_type_mapping.json", "w") as f:
        json.dump(doc_type_mapping, f)


    def test_model(sentences, model_path="./distilbert_doc_retrieve_classifier"):
        """Tests the trained model on new sentences."""
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        model.to(device)

        with open(f"{model_path}/label_mapping.json", "r") as f:
            label_mapping = json.load(f)
        label_mapping = {v: k for k, v in label_mapping.items()}  # Reverse mapping

        with open(f"{model_path}/doc_type_mapping.json", "r") as f:
            doc_type_mapping = json.load(f)
        doc_type_mapping = {v: k for k, v in doc_type_mapping.items()}  # Reverse mapping

        results = []
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to GPU if available

            with torch.no_grad():  # Disable gradient calculation for inference
                outputs = model(**inputs)

            predicted_classes = torch.argmax(outputs.logits, dim=1).tolist()

            if len(predicted_classes) == 2:
                predicted_label = label_mapping.get(predicted_classes[0], "UNKNOWN")
                predicted_doc_type = doc_type_mapping.get(predicted_classes[1], "UNKNOWN")
            else:
                predicted_label = label_mapping.get(predicted_classes[0], "UNKNOWN")
                predicted_doc_type = "UNKNOWN"

            results.append((sentence, predicted_label, predicted_doc_type))

        return results


    def save_incorrect_data(incorrect_data):
        """Saves incorrect predictions for further review."""
        file_path = "incorrect_predictions.json"

        try:
            with open(file_path, "r") as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = []

        existing_data.append(incorrect_data)

        with open(file_path, "w") as f:
            json.dump(existing_data, f, indent=4)


    def save_correct_data(correct_data):
        """Saves incorrect predictions for further review."""
        file_path = "correct_predictions.json"

        try:
            with open(file_path, "r") as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = []

        existing_data.append(correct_data)

        with open(file_path, "w") as f:
            json.dump(existing_data, f, indent=4)


    # Interactive user input
    print("\nEnter sentences to classify. Type 'exit' to stop.")
    while True:
        user_input = input("Enter text: ").strip()
        if user_input.lower() == "exit":
            print("Exiting program.")
            break

        predictions = test_model([user_input])
        for text, pred_label, pred_doc_type in predictions:
            print(f"Text: '{text}' | Predicted Class: {pred_label} | Document Type: {pred_doc_type}\n")

            # Ask the user if the prediction is correct
            user_feedback = input("Is this prediction correct? (Y/N): ").strip().lower()

            if user_feedback == "n":
                # Ask for correct label (D for DOCUMENT_RETRIEVAL, C for CONVERSATION)
                while True:
                    correct_label_input = input(
                        "Enter the correct label (D for DOCUMENT_RETRIEVAL, C for CONVERSATION): ").strip().upper()
                    if correct_label_input in ["D", "C"]:
                        correct_label = "DOCUMENT_RETRIEVAL" if correct_label_input == "D" else "CONVERSATION"
                        break
                    print("Invalid input! Please enter 'D' or 'C'.")

                # Ask for correct document type
                correct_doc_type = input("Enter the correct doc_type (or NONE if not applicable): ").strip().upper()

                incorrect_data = {
                    "text": text,
                    "label": correct_label,
                    "doc_type": correct_doc_type
                }

                save_incorrect_data(incorrect_data)
                print("Your correction has been saved for future training!\n")
            else:
                correct_doc_type = input("Enter the correct doc_type (or NONE if not applicable): ").strip().upper()

                correct_data = {
                    "text": text,
                    "label": pred_label,
                    "doc_type": correct_doc_type,

                }

                save_correct_data(correct_data)
                print("Your correction has been saved for future training!\n")
