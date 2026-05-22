import sys
from utils.load import load_csv, get_hist_house, is_numeric_column
from utils.stats import print_graph_hist
from utils.test.test import find_homogeneous_feature

def main():
    #if len(sys.argv) != 3:
    #    print("\033[33m#### ---> Python histogram.py datasets/dataset_train.csv Feature ####\033[0m")
    #    sys.exit(1)

#-------- Loader
    #filepath = sys.argv[1]
    filepath = './datasets/dataset_train.csv'
    dataset = load_csv(filepath)

    #feat = sys.argv[2]
    feat = "Arithmancy"

#----TEST-----
    EXCLUDE      = ['Index']
    numeric_cols = [
        col for col, vals in dataset.items()
        if is_numeric_column(vals) and col not in EXCLUDE
    ]
    best_feat = find_homogeneous_feature(filepath, dataset, numeric_cols)
    data_house = get_hist_house(filepath, dataset, best_feat)
    print_graph_hist(data_house, best_feat)

#---------------

    # data_house = get_hist_house(filepath, dataset, feat)
    # print_graph_hist(data_house, feat)

if __name__ == '__main__':
    main()
