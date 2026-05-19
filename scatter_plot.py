import sys
from utils.load import load_csv, get_xy, get_xy_house
from utils.stats import print_graph_scatter

def main():
    # if len(sys.argv) != 4:
    #     print("\033[33m#### ---> Python scatter_plot.py datasets/dataset_train.csv Feature1 Feature2 ####\033[0m")
    #     sys.exit(1)

#-------Loader
    filepath = sys.argv[1]
    dataset = load_csv(filepath)
    #Astronomy vs Divination ok
    #feat1 = sys.argv[2]
    #feat2 = sys.argv[3]
    feat1 = "Astronomy"
    feat2 = "Potions"
#-----------------

    data_house = get_xy_house(filepath, dataset, feat1, feat2)

    #Printer + Covariance
    print_graph_scatter(data_house, feat1, feat2)


if __name__ == '__main__':
    main()
    
