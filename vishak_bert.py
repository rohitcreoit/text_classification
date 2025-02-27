from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, pipeline


# Load your JSONL dataset
dataset = load_dataset("json", data_files="dataset.json")

# Split into train and test sets
dataset = dataset["train"].train_test_split(test_size=0.2)

print(dataset)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # Base BERT model
    num_labels=2          # Number of classes (DOCUMENT_RETRIEVAL and CONVERSATION)
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data collator ensures proper padding for batches during training
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Convert labels to integers if they're not already
def convert_labels(example):
    example["label"] = 0 if example["label"] == "DOCUMENT_RETRIEVAL" else 1
    return example

tokenized_dataset = tokenized_dataset.map(convert_labels)

# Setup configuration for the fine-tuning process
training_args = TrainingArguments(
    output_dir="./results",            # Directory to save model checkpoints
    eval_strategy="epoch",            # Use the correct argument name
    learning_rate=2e-5,               # Learning rate for optimizer
    per_device_train_batch_size=8,    # Batch size for training
    per_device_eval_batch_size=8,     # Batch size for evaluation
    num_train_epochs=3,               # Number of training epochs
    weight_decay=0.01,                # Weight decay for regularization
    logging_dir="./logs",             # Directory for logs
    logging_steps=10,                 # Log after every 10 steps
)

trainer = Trainer(
    model=model,                         # Pre-trained BERT model
    args=training_args,                  # Training arguments
    train_dataset=tokenized_dataset["train"],  # Training dataset
    eval_dataset=tokenized_dataset["test"],    # Evaluation dataset
    tokenizer=tokenizer,                 # Tokenizer for decoding
    data_collator=data_collator          # Data collator for padding
)

trainer.train()

results = trainer.evaluate()
print(results)

trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# test the fine-tuned model
# Load the fine-tuned model
classifier = pipeline("text-classification", model="./fine_tuned_model", tokenizer="./fine_tuned_model")

# Test the classifier
test_sentences = [
    "I wanna see my blood test reports.",
    "How do I understand these reports?",
    "Can I have a look into my mammogram reports?",
    "Who are you?",
    "How can you assist me?"
]

# Define label mapping
label_map = {
    0: "DOCUMENT_RETRIEVAL",
    1: "CONVERSATION"
}

for sentence in test_sentences:
    result = classifier(sentence)
    # Extract the numerical label and map it to the string label
    label = label_map[int(result[0]["label"].split("_")[-1])]
    score = result[0]["score"]
    print(f"Input: {sentence}")
    print(f"Predicted Label: {label}, Confidence: {score:.2f}\n")