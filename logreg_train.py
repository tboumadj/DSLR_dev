import sys
from utils.load import is_numeric_column, load_csv, dataset_to_dataframe
from utils.train import normalize_feat, extract_X_y

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

#--------Noralisation

    dataframe_norm, params = normalize_feat(dataframe, valid_feat)
    print(f'df_norm : {dataframe_norm.shape}')
    X, y = extract_X_y(dataframe_norm, valid_feat)
 
#------------------Test Print
    print(f"Élèves             : {len(X)}")
    print(f"Features           : {len(X[0])}")
    print(f"Maisons uniques    : {set(y)}")
    print(f"Premier élève      : {[round(v, 4) for v in X[0]]}")
    print(f"Sa maison          : {y[0]}")

    print(f"\nVérification normalisation :")
    for i, feat in enumerate(valid_feat):
        col  = [X[j][i] for j in range(len(X))]
        mean = sum(col) / len(col)
        std  = (sum((x - mean) ** 2 for x in col) / len(col)) ** 0.5
        print(f"  {feat:<35}  mean={mean:>8.5f}  std={std:>6.4f}")
#---------------
    print("\033[33m### Generate JsonConfig for predict ... ####\033[0m")

if __name__ == '__main__':
    main()
