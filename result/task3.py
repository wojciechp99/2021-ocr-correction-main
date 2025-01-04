import csv
import pandas as pd

if __name__ == '__main__':
    file_path = "dev-0/in.tsv"
    data = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')  # Użyj tabulacji jako separatora
        for row in reader:
            data.append({'doc_id': row[0], 'page_num': row[1], 'year': row[2], 'ocr_text': row[3]})

        # Convert to pandas DataFrame for easier processing
        df = pd.DataFrame(data)

        # Example: Display the first row
        print(df.head())
