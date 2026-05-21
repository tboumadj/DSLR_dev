import sys
from utils.load import load_csv, is_numeric_column, dataset_to_dataframe
from utils.stats import print_graph_sns

def main():
    #if len(sys.argv) != 3:
    #    print("\033[33m#### ---> Python histogram.py datasets/dataset_train.csv Feature ####\033[0m")
    #    sys.exit(1)

#-------- Loader
    #filepath = sys.argv[1]
    filepath = './datasets/dataset_train.csv'
    dataset = load_csv(filepath)

#-------Test SNS
    EXCLUDE      = ['Index']
    numeric_cols = [
        col for col, vals in dataset.items()
        if is_numeric_column(vals) and col not in EXCLUDE
    ]
    dataframe = dataset_to_dataframe(dataset, numeric_cols)
    #print_graph_sns(dataframe)

if __name__ == '__main__':
    main()