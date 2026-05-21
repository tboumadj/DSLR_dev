import sys
from utils.load import load_csv, get_hist_house
from utils.stats import print_graph_hist

def main():
    # if len(sys.argv) != 4:
    #     print("\033[33m#### ---> Python histogram.py datasets/dataset_train.csv Feature ####\033[0m")
    #     sys.exit(1)

    #-------- Loader
    #filepath = sys.argv[1]
    filepath = './datasets/dataset_train.csv'
    dataset = load_csv(filepath)

    #feat = sys.argv[2]
    feat = "Arithmancy"
    #---------------

    data_house = get_hist_house(filepath, dataset, feat)
    print_graph_hist(data_house, feat)

if __name__ == '__main__':
    main()