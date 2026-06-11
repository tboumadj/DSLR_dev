import sys
from utils.load import load_csv, get_hist_house, is_numeric_column
from utils.stats import print_graph_hist
from utils.test.test import find_homogeneous_feature

def main():
#-------- Loader
    filepath = './datasets/dataset_train.csv'
    dataset = load_csv(filepath)

    feat = "Arithmancy"

#----TEST-----
    EXCLUDE      = ['Index']
    numeric_cols = [
        col for col, vals in dataset.items()
        if is_numeric_column(vals) and col not in EXCLUDE
    ]
    best_feat = find_homogeneous_feature(filepath, dataset, numeric_cols)
    #data_house = get_hist_house(filepath, dataset, best_feat)
    #print_graph_hist(data_house, best_feat)

#---------------

    data_house = get_hist_house(filepath, dataset, feat)
    print_graph_hist(data_house, feat)

if __name__ == '__main__':
    main()
