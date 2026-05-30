import json
import sys

import numpy as np

from utils.load import dataset_to_dataframe
from utils.train import extract_X_y, standardize_feat

FILEPATH = './datasets/dataset_test.csv'


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
    except FileNotFoundError:
        print("weights.json not found, run logreg_train.py first")
        sys.exit(1)

    dataframe = dataset_to_dataframe(FILEPATH, features, keep_houses=False)
    dataframe_stand, params = standardize_feat(dataframe, features)

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


if __name__ == '__main__':
    main()

    

