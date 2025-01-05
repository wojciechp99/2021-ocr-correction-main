import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import os

os.environ["WANDB_DISABLED"] = "true"

# Load the training data
in_train_path = '../train/inTrain.tsv'
expected_train_path = '../train/expectedTrain.tsv'
in_test_path = '../dev-0/in.tsv'
expected_test_path = '../dev-0/expected.tsv'

# Load training data
in_train = pd.read_csv(in_train_path, sep='\t', names=['DocumentID', 'PageNumber', 'Year', 'Text'])
expected_train = pd.read_csv(expected_train_path, sep='\t', names=['Text'])

# Merge training data
train_data = pd.DataFrame({
    'input_text': in_train['Text'],
    'expected_text': expected_train['Text']
})

# Ensure data is clean
train_data['input_text'] = train_data['input_text'].fillna('').astype(str)
train_data['expected_text'] = train_data['expected_text'].fillna('').astype(str)

print("1")
# Select the first 20 samples from the training data
train_text_number = 20
print(f"Model jest trenowany na {train_text_number} tekstach")
train_data_subset = train_data.head(train_text_number)


# Truncate long text
def truncate_text(text, max_len=128):
    return text[:max_len] if len(text) > max_len else text


# Apply truncation to avoid warnings
train_data_subset.loc[:, 'input_text'] = train_data_subset['input_text'].apply(lambda x: truncate_text(x, max_len=128))
train_data_subset.loc[:, 'expected_text'] = train_data_subset['expected_text'].apply(
    lambda x: truncate_text(x, max_len=128))

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_data_subset.rename(columns={"input_text": "text", "expected_text": "label"}))

# Load T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
print("2")


# Tokenize the dataset
def tokenize_function(examples):
    # Tokenizowanie tekstów wejściowych
    model_inputs = tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=128
    )
    # Tokenizowanie etykiet (label)
    labels = tokenizer(
        examples["label"], truncation=True, padding="max_length", max_length=128, return_tensors="pt"
    )
    # Upewnij się, że etykiety są poprawnie ustawione
    model_inputs["labels"] = labels["input_ids"].squeeze().tolist() if "input_ids" in labels else []
    return model_inputs


# Tokenizacja i walidacja danych
tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

# Usuń puste etykiety
tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["labels"]) > 0)

# Debugging: Wyświetl przykładowe dane
print("Przykład tokenizowanych danych:")
print(tokenized_dataset[0])

print("3")
# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    report_to=[]
)

# Define Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)
print("4")
# Train the model
trainer.train()

# Load test data
in_test = pd.read_csv(in_test_path, sep='\t', names=['DocumentID', 'PageNumber', 'Year', 'Text'])
expected_test = pd.read_csv(expected_test_path, sep='\t', names=['Text'])

# Ensure test data is clean
in_test['Text'] = in_test['Text'].fillna('').astype(str)
print("5")


# Apply the model to test data
def generate_corrections(input_text):
    input_ids = tokenizer("popraw błąd: " + input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


in_test['Corrected_Text'] = in_test['Text'].apply(generate_corrections)
print("6")
# Save the output
output_path = '../result/corrected_output.tsv'
in_test[['DocumentID', 'PageNumber', 'Year', 'Corrected_Text']].to_csv(output_path, sep='\t', index=False)

print(f"Corrected output saved to {output_path}")
