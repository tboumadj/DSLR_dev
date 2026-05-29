import sys
from utils.load import is_numeric_column, load_csv, dataset_to_dataframe
from utils.train import standardize_feat, extract_X_y

def main():
#--------Loader
    filepath = './datasets/dataset_train.csv'
    dataset = load_csv(filepath)

    print("\033[33m### Run Training Part ... ####\033[0m")
#--------Prep Data
    EXCLUDE      = ['Index',
                    'Arithmancy',
                    'Astronomy',
                    'Care of Magical Creatures']
    valid_feat = [
        col for col, vals in dataset.items()
        if is_numeric_column(vals) and col not in EXCLUDE
    ]
    
    dataframe = dataset_to_dataframe(filepath, valid_feat)
    print(f'valid dataframe : {dataframe.shape}')

#--------Standardisation

    dataframe_stand, params = standardize_feat(dataframe, valid_feat)
    print(f'df_stand : {dataframe_stand.shape}')
    X, y = extract_X_y(dataframe_stand, valid_feat)
 
#------------------Test Print

#---------------
    print("\033[33m### Generate JsonConfig for predict ... ####\033[0m")

if __name__ == '__main__':
    main()
