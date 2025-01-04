import csv

if __name__ == '__main__':
    with open('../dev-0/in.tsv', mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')  # Użyj tabulacji jako separatora
        for row in reader:
            print(row)
            break
