import json

import numpy as np
from utils.load import is_numeric_column, load_csv, dataset_to_dataframe
from utils.train import standardize_feat, extract_X_y

ITERATION_NUMBER = 100
LEARNING_RATE = 0.01
EXCLUDE = ['Index',
           'Arithmancy',
           'Astronomy',
           'Care of Magical Creatures']

def train_model(valid_feat, X, Y, house):
    
    weights = np.zeros(len(valid_feat) + 1)
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    for i in range (0, ITERATION_NUMBER):

        # add 1 column to datas to include biais to dot product 
        predicts = 1 / (1 + np.exp(-np.dot(X, weights))) 

        gradiant = 1 / len(X) * np.dot(X.T, predicts - (house == Y) ) 

        weights = weights - gradiant
    
    mean_absolute_error = np.sum(np.abs(predicts - (house == Y))) / len(X)
    print(f"Mean absolute error for {house}:", mean_absolute_error)

    return (weights)


def main():

#--------Loader

    filepath = './datasets/dataset_train.csv'
    dataset = load_csv(filepath)

    print("\033[33m### Run Training Part ... ####\033[0m")

#--------Prep Data


    valid_feat = [
        col for col, vals in dataset.items()
        if is_numeric_column(vals) and col not in EXCLUDE
    ]
    
    dataframe = dataset_to_dataframe(filepath, valid_feat)
    print(f'valid dataframe : {dataframe.shape}')

#--------Standardisation

    dataframe_stand, params = standardize_feat(dataframe, valid_feat)
    X, y = extract_X_y(dataframe_stand, valid_feat)

#--------Training 

    Gryffindor = train_model(valid_feat, X, y, "Gryffindor")
    Slytherin = train_model(valid_feat, X, y, "Slytherin")
    Ravenclaw = train_model(valid_feat, X, y, "Ravenclaw")
    Hufflepuff = train_model(valid_feat, X, y, "Hufflepuff")

#--------Output

    weights_dict = {
        "features": valid_feat,
        "Gryffindor": Gryffindor.tolist(),
        "Slytherin": Slytherin.tolist(),
        "Ravenclaw": Ravenclaw.tolist(),
        "Hufflepuff": Hufflepuff.tolist(),
    }

    print("\033[33m### Generate weights.json for predict ... ####\033[0m")

    with open("weights.json", "w") as weights_file:
            json.dump(weights_dict, weights_file)

if __name__ == '__main__':
    main()
