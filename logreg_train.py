import json

import numpy as np
from utils.load import is_numeric_column, load_csv, dataset_to_dataframe
from utils.train import standardize_feat, extract_X_y
import argparse

ITERATION_NUMBER = 50
LEARNING_RATE = 1
EXCLUDE = ['Index',
           'Arithmancy',
           'Astronomy',
           'Divination']

def train_model(valid_feat, X, Y, house, sample_size):
    
    weights = np.zeros(len(valid_feat) + 1)
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    Y_batch = Y
    X_batch = X

    if sample_size != len(X):
         batch_mode = True

    for i in range (0, ITERATION_NUMBER):

        if batch_mode:
            indexs = np.random.choice(len(X), size=sample_size, replace=False)
            X_batch = X[indexs]
            Y_batch = Y[indexs]
            

        # add 1 column to datas to include biais to dot product 
        predicts = 1 / (1 + np.exp(-np.dot(X_batch, weights))) 

        gradiant = 1 / len(X_batch) * np.dot(X_batch.T, predicts - (house == Y_batch) ) 

        weights = weights - LEARNING_RATE * gradiant
    
    return (weights)


def main():

#--------args 
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stochastic", action="store_true", help="Activate stochastic gradient descent")

    parser.add_argument("-b", "--batch-size", action="store", type=int, help="Specify batch size for stochastic gradient descent (default = 16)", default=16)
    args = parser.parse_args()


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

#--------sample size

    sample_size = len(X)

    if (args.stochastic == True):
        if (args.batch_size == 1):
             print("Mode: pure stochastic gradiant descent")
             sample_size = 1
        else: 
             print(f"Mode: mini batch descent gradiant with {args.batch_size} samples")
             sample_size = args.batch_size

#--------Training 

    Gryffindor = train_model(valid_feat, X, y, "Gryffindor", sample_size)
    Slytherin = train_model(valid_feat, X, y, "Slytherin", sample_size)
    Ravenclaw = train_model(valid_feat, X, y, "Ravenclaw", sample_size)
    Hufflepuff = train_model(valid_feat, X, y, "Hufflepuff", sample_size)

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
