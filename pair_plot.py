import sys
from utils.load import load_csv, is_numeric_column, dataset_to_dataframe
from utils.stats import print_graph_sns

def main():
#-------- Loader
    filepath = './datasets/dataset_train.csv'
    dataset = load_csv(filepath)

#-------Test SNS
    EXCLUDE      = ['Index']
    numeric_cols = [
        col for col, vals in dataset.items()
        if is_numeric_column(vals) and col not in EXCLUDE
    ]
    dataframe = dataset_to_dataframe(filepath, numeric_cols)
    print_graph_sns(dataframe, numeric_cols)

if __name__ == '__main__':
    main()