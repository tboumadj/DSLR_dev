import sys
from utils.load import load_csv, get_xy_house
from utils.stats import print_graph_scatter

def main():
#-------Loader
    filepath = './datasets/dataset_train.csv'
    dataset = load_csv(filepath)
    
    feat1 = "Astronomy"
    feat2 = "Defense Against the Dark Arts"
#-----------------

    data_house = get_xy_house(filepath, dataset, feat1, feat2)

    print_graph_scatter(data_house, feat1, feat2)


if __name__ == '__main__':
    main()