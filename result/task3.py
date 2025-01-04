import csv
import pandas as pd

if __name__ == '__main__':
    file_path = "../dev-0/in.tsv"
    data = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            data.append({'doc_id': row[0], 'page_num': row[1], 'year': row[2], 'ocr_text': row[3]})

        df = pd.DataFrame(data)
        pd.set_option('display.max_colwidth', 700)

        print(df.head())
