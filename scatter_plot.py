import sys
from utils.load import load_csv, get_xy, get_xy_house, is_numeric_column
from utils.stats import print_graph_scatter, print_graph_test, find_correlation

def main():
    # if len(sys.argv) != 4:
    #     print("\033[33m#### ---> Python scatter_plot.py datasets/dataset_train.csv Feature1 Feature2 ####\033[0m")
    #     sys.exit(1)

#-------Loader
    #filepath = sys.argv[1]
    filepath = './datasets/dataset_train.csv'
    dataset = load_csv(filepath)
    
    #Defense Against the Dark Arts vs Astronomy ok
    #feat1 = sys.argv[2]
    #feat2 = sys.argv[3]
    feat1 = "Defense Against the Dark Arts"
    feat2 = "Astronomy"
#-----------------

    data_house = get_xy_house(filepath, dataset, feat1, feat2)

    #test corellation
    EXCLUDE_COLS = ['Index']
    numeric_cols = [col for col, vals in dataset.items() if is_numeric_column(vals) and col not in EXCLUDE_COLS]
    find_correlation(dataset, numeric_cols)

    #print_graph_scatter(data_house, feat1, feat2)
    print_graph_test(data_house, feat1, feat2)


if __name__ == '__main__':
    main()
    
