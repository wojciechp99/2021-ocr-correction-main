# import csv
# import re
#
# # Przykładowe reguły konwersji (rozszerz według potrzeb)
# import pandas as pd
#
# HISTORICAL_TO_MODERN = {
#     r"rzeczÿ": "rzeczy",  # Przykład zamiany historycznej pisowni
#     r"wszyſtko": "wszystko",  # "ſ" na "s"
#     r"niemá": "nie ma",  # Dodanie spacji w archaicznych formach
#     r"é": "e",  # Zamiana é na e
#     r"ćwicźenie": "ćwiczenie",  # Korekta pisowni
# }
#
#
# def correct_text(text, rules):
#     for pattern, replacement in rules.items():
#         text = re.sub(pattern, replacement, text)
#     return text
#
#
# def ocr_post_correction(input_text):
#     # 1. Usuwanie znaków specjalnych (opcjonalnie)
#     text_cleaned = re.sub(r"[^a-zA-Z0-9ąćęłńóśźżſ \n]", "", input_text)
#
#     # 2. Poprawa historycznej pisowni na podstawie reguł
#     corrected_text = correct_text(text_cleaned, HISTORICAL_TO_MODERN)
#
#     # 3. Opcjonalna poprawa na podstawie słownika współczesnego języka
#     # (Można zaimplementować, np. przy użyciu biblioteki PyHunspell lub podobnych)
#
#     return corrected_text
#
#
# if __name__ == '__main__':
#     file_path = "../dev-0/in.tsv"
#     data = []
#     with open(file_path, mode='r', encoding='utf-8') as file:
#         reader = csv.reader(file, delimiter='\t')
#         for row in reader:
#             data.append({'doc_id': row[0], 'page_num': row[1], 'year': row[2], 'ocr_text': row[3]})
#
#         df = pd.DataFrame(data)
#         pd.set_option('display.max_colwidth', 700)
#         print(df.head())
#     # Testowanie
#     sample_text = """
#     Rzeczÿ ściągnionych do końca wszyſtko, niemá nadziei, że tamto ćwicźenie będzie dobre.
#     """
#
#     corrected_sample = ocr_post_correction(sample_text)
#     print("Poprawiony tekst:")
#     print(corrected_sample)

# import csv
# import pandas as pd
# if __name__ == '__main__':
#     file_path = "corrected_output.tsv"
#     data = []
#     with open(file_path, mode='r', encoding='utf-8') as file:
#         reader = csv.reader(file, delimiter='\t')
#         for row in reader:
#             data.append({'ocr_text': row[0]})
#
#         df = pd.DataFrame(data)
#         pd.set_option('display.max_colwidth', 700)
#         print(df.head(10))