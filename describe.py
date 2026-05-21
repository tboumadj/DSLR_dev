import sys
from utils.load import load_csv, is_numeric_column
from utils.stats import describe_feature, print_describe
from utils.test.test import pandas_feature

def main():
    #if len(sys.argv) != 2:
    #    print("\033[33m#### ---> Python describe.py datasets/dataset_train.csv ####\033[0m")
    #    sys.exit(1)

    #filepath = sys.argv[1]
    filepath = './datasets/dataset_train.csv'
    dataset = load_csv(filepath)

    print("\033[32m#### All file column: ####\033[0m")
    for col, vals in dataset.items():
        clean = [x for x in vals if x == x]
        print(f"{col:40s}  total={len(vals)}  valid={len(clean)}  missing={len(vals)-len(clean)}")

    print("\033[32m#### Only numeric column now: ####\033[0m")
    EXCLUDE_COLS = ['Index']
    numeric_cols = [col for col, vals in dataset.items() if is_numeric_column(vals) and col not in EXCLUDE_COLS]
    print(f"{len(numeric_cols)} colonnes numériques :")
    for col in numeric_cols:
        print(f"  {col}")

    #Stat
    stats = {}
    print("\033[32m#### Stats by feature: ####\033[0m")
    for col in numeric_cols:
        stats[col] = describe_feature(dataset[col])

    print_describe(stats)

    #Test + diff with pandas
    pandas_feature(stats, filepath)

if __name__ == '__main__':
    main()