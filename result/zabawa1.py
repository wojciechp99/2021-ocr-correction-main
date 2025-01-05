import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from difflib import SequenceMatcher
import numpy as np

# Load the training data
in_train_path = '../train/inTrain.tsv'
expected_train_path = '../train/expectedTrain.tsv'
in_test_path = '../dev-0/in.tsv'
expected_test_path = '../dev-0/expected.tsv'

# Load training data
in_train = pd.read_csv(in_train_path, sep='\t', names=['DocumentID', 'PageNumber', 'Year', 'Text'])
expected_train = pd.read_csv(expected_train_path, sep='\t', names=['Text'])

# Load test data
in_test = pd.read_csv(in_test_path, sep='\t', names=['DocumentID', 'PageNumber', 'Year', 'Text'])
expected_test = pd.read_csv(expected_test_path, sep='\t', names=['Text'])

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


# Helper function to generate character n-grams
def generate_ngrams(text, n=3):
    return [text[i:i + n] for i in range(len(text) - n + 1)]


# Create a feature set from n-grams
def create_features(input_texts, target_texts):
    features = []
    labels = []
    for input_text, target_text in zip(input_texts, target_texts):
        sm = SequenceMatcher(None, input_text, target_text)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != 'equal':
                features.append(input_text[i1:i2])
                labels.append(target_text[j1:j2])
    return features, labels


# Generate training features and labels
features, labels = create_features(train_data_subset['input_text'], train_data_subset['expected_text'])
print("2")
# Create a text correction model
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 2))
model = LogisticRegression(max_iter=1000)

# Create a pipeline
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', model)
])

# Fit the model
pipeline.fit(features, labels)
print("3")


# Correction function
def correct_text(input_text, model):
    corrected_text = input_text
    sm = SequenceMatcher(None, input_text, input_text)  # Dummy for iteration
    corrections = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != 'equal':
            segment = input_text[i1:i2]
            predicted = model.predict([segment])
            corrections.append((i1, i2, predicted[0]))
    # Apply corrections
    for start, end, replacement in sorted(corrections, key=lambda x: -x[0]):
        corrected_text = corrected_text[:start] + replacement + corrected_text[end:]
    return corrected_text


print("4")
in_test['Text'] = in_test['Text'].fillna('').astype(str)
# Apply the model to test data
test_data = in_test['Text'].apply(lambda x: correct_text(x, pipeline))

# Save the output
output_path = '../result/corrected_output.tsv'
test_data.to_csv(output_path, sep='\t', index=False, header=False)

print(f"Corrected output saved to {output_path}")

# Load expected test data
expected_data = pd.read_csv(expected_test_path, sep='\t', names=['Text'])

# Ensure both datasets are aligned in length
if len(test_data) != len(expected_data):
    print(f"Warning: Mismatch in lengths. Model output: {len(test_data)}, Expected: {len(expected_data)}")
else:
    # Compare the outputs
    differences = []
    for idx, (model_output, expected_output) in enumerate(zip(test_data, expected_data['Text'])):
        if model_output.strip() != expected_output.strip():  # Ignore leading/trailing spaces
            differences.append({
                "Index": idx,
                "Model_Output": model_output,
                "Expected_Output": expected_output
            })

    # Display differences
    if differences:
        print(f"Differences found in {len(differences)} entries out of {len(test_data)}:")
        for diff in differences[:10]:  # Show first 10 differences
            print(
                f"Index {diff['Index']}:\nModel Output: {diff['Model_Output']}\nExpected: {diff['Expected_Output']}\n")
    else:
        print("No differences found between model output and expected data.")
