import os
import csv
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BertForMaskedLM
from datasets import Dataset

# Suppress parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load and preprocess the dataset
def load_data(file_path):
    data = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            data.append({'doc_id': row[0], 'page_num': row[1], 'year': row[2], 'ocr_text': row[3]})
    return pd.DataFrame(data)

def preprocess_data(df, tokenizer, max_length=128):
    def tokenize_function(examples):
        inputs = tokenizer(examples['ocr_text'], max_length=max_length, truncation=True, padding='max_length')
        inputs['labels'] = inputs['input_ids'].copy()  # For language modeling
        return inputs

    dataset = Dataset.from_pandas(df)
    return dataset.map(tokenize_function, batched=True)

# Fine-tune the model
def fine_tune_model(dataset, model_name="dkleczek/bert-base-polish-uncased-v1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name, is_decoder=False)

    # Split dataset into training and evaluation sets
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=500,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    return model, tokenizer

# Use the trained model for inference
def correct_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Load data
    file_path = "../dev-0/in.tsv"  # Update this with your file path
    df = load_data(file_path)

    # Prepare data for BERT
    model_name = "dkleczek/bert-base-polish-uncased-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = preprocess_data(df, tokenizer)

    # Fine-tune the BERT model
    model, tokenizer = fine_tune_model(dataset, model_name)

    # Test the model on a sample OCR text
    sample_text = "ca Wtem\\n\\nwśród postów i ciężkich umartwień, nad\\ngrobem własną ręką wykopanym..."
    corrected_text = correct_text(model, tokenizer, sample_text)
    print("Corrected Text:", corrected_text)
