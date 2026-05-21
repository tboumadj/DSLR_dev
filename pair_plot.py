import sys
from utils.load import load_csv

def main():
    #if len(sys.argv) != 3:
    #    print("\033[33m#### ---> Python histogram.py datasets/dataset_train.csv Feature ####\033[0m")
    #    sys.exit(1)

#-------- Loader
    #filepath = sys.argv[1]
    filepath = './datasets/dataset_train.csv'
    dataset = load_csv(filepath)

if __name__ == '__main__':
    main()