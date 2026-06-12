import json
import sys

import numpy as np
import pandas as pd

from utils.load import dataset_to_dataframe_predict
from utils.train import standardize_feat_predict

FILEPATH = './datasets/dataset_test.csv'

#-------------------------TEST-----------
def evaluate_accuracy(true_filepath, pred_filepath="houses.csv"):

    df_true = pd.read_csv(true_filepath)
    df_pred = pd.read_csv(pred_filepath)


    true_labels = df_true['Hogwarts House'].tolist()
    pred_labels = df_pred['Hogwarts House'].tolist()

    correct  = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    total    = len(true_labels)
    accuracy = correct / total * 100

    print(f"\033[32m### Model precision: {accuracy:.2f}% ####\033[0m")

    return accuracy
#------------------------------------------------------


def main():

#--------Loader
    
    try: 
        with open("weights.json") as weights_file:
            data = json.load(weights_file)
            features = data["features"]

            Gryffindor = data["Gryffindor"]
            Slytherin = data["Slytherin"]
            Ravenclaw = data["Ravenclaw"]
            Hufflepuff = data["Hufflepuff"]
            feat_standard = data["Feat_Standard"]
    except FileNotFoundError:
        print("weights.json not found, run logreg_train.py first")
        sys.exit(1)

    dataframe = dataset_to_dataframe_predict(FILEPATH, features, feat_standard, keep_houses=False)

    dataframe_stand = standardize_feat_predict(dataframe, features, feat_standard)

    X = np.hstack([np.ones((dataframe_stand.shape[0], 1)), dataframe_stand])

    #--------Predict

    predict_Gryffindor = np.dot(X, Gryffindor)
    predict_Slytherin = np.dot(X, Slytherin)
    predict_Ravenclaw = np.dot(X, Ravenclaw)
    predict_Hufflepuff = np.dot(X, Hufflepuff)

    #--------Output

    houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
    scores = np.column_stack([predict_Gryffindor, predict_Slytherin, predict_Ravenclaw, predict_Hufflepuff])
    predicted = [houses[i] for i in np.argmax(scores, axis=1)]

    with open("houses.csv", "w") as f:
        f.write("Index,Hogwarts House\n")
        for i, house in enumerate(predicted):
            f.write(f"{i},{house}\n")

    print("\033[33m### Successful: prediction writed in houses.csv ####\033[0m")

#---------------test---------------
    evaluate_accuracy(FILEPATH, "houses.csv")


if __name__ == '__main__':
    main()

    

