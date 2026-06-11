import json

import numpy as np
import matplotlib.pyplot as plt
from utils.load import is_numeric_column, load_csv, dataset_to_dataframe
from utils.train import standardize_feat, extract_X_y
import argparse

EPOCHS = 150
LEARNING_RATE = 10
EXCLUDE = ['Index',
           'Arithmancy',
           'Astronomy',
           'Herbology']
DATASET_PATH = './datasets/dataset_train.csv'
PLOT_STATE = None


def compute_cost(X, Y, weights, house):
    y_binary = (Y == house).astype(float)
    predictions = 1 / (1 + np.exp(-np.dot(X, weights)))
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(y_binary * np.log(predictions) + (1 - y_binary) * np.log(1 - predictions))


def init_cost_plot(houses):
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Cost function progression during training")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.grid(True, alpha=0.3)

    plot_state = {
        "fig": fig,
        "ax": ax,
    }

    for house in houses:
        line, = ax.plot([], [], label=house)
        plot_state[house] = {
            "line": line,
            "iterations": [],
            "costs": [],
        }

    ax.legend()
    fig.tight_layout()
    plt.show(block=False)
    return plot_state


def update_cost_plot(house, iteration, cost):
    if PLOT_STATE is None:
        return

    house_state = PLOT_STATE[house]
    house_state["iterations"].append(iteration)
    house_state["costs"].append(cost)
    house_state["line"].set_data(house_state["iterations"], house_state["costs"])

    ax = PLOT_STATE["ax"]
    ax.relim()
    ax.autoscale_view()
    PLOT_STATE["fig"].canvas.draw_idle()
    PLOT_STATE["fig"].canvas.flush_events()
    plt.pause(0.001)

def train_model(valid_feat, X, Y, house, sample_size):
    global PLOT_STATE

    weights = np.zeros(len(valid_feat) + 1)
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    Y_batch = Y
    X_batch = X
    batch_mode = False

    if sample_size != len(X):
         batch_mode = True

    for i in range (0, EPOCHS):

        if batch_mode:
            indices = np.random.choice(len(X), size=sample_size, replace=False)
            X_batch = X[indices]
            Y_batch = Y[indices]

        # add 1 column to datas to include biais to dot product 
        predicts = 1 / (1 + np.exp(-np.dot(X_batch, weights))) 

        gradient = 1 / len(X_batch) * np.dot(X_batch.T, predicts - (house == Y_batch) ) 

        weights = weights - LEARNING_RATE * gradient

        update_cost_plot(house, i + 1, compute_cost(X, Y, weights, house))
    
    return (weights)


def main():

    global PLOT_STATE

#--------Args 

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stochastic", action="store_true", help="Activate stochastic gradient descent")

    parser.add_argument("-b", "--batch-size", action="store", type=int, help="Specify batch size for stochastic gradient descent (default = 16)", default=16)
    args = parser.parse_args()

#--------Loader

    dataset = load_csv(DATASET_PATH)

#--------Prep Data

    valid_feat = [
        col for col, vals in dataset.items()
        if is_numeric_column(vals) and col not in EXCLUDE
    ]
    
    dataframe = dataset_to_dataframe(DATASET_PATH, valid_feat)

#--------Standardisation

    dataframe_stand, params = standardize_feat(dataframe, valid_feat)
    X, y = extract_X_y(dataframe_stand, valid_feat)

#--------Sample size

    sample_size = len(X)

    if (args.stochastic == True):
        if (args.batch_size == 1):
             print("mode: pure stochastic gradient descent")
             sample_size = 1
        else: 
             print(f"mode: mini batch descent gradient with {args.batch_size} samples")
             sample_size = args.batch_size

#--------Training 

    np.random.seed(0)

    houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
    PLOT_STATE = init_cost_plot(houses)

    Gryffindor_w = train_model(valid_feat, X, y, "Gryffindor", sample_size)
    Slytherin_w = train_model(valid_feat, X, y, "Slytherin", sample_size)
    Ravenclaw_w = train_model(valid_feat, X, y, "Ravenclaw", sample_size)
    Hufflepuff_w = train_model(valid_feat, X, y, "Hufflepuff", sample_size)

#--------Print precision  

    X = np.hstack([np.ones((X.shape[0], 1)), X])

    prediction_Gryffindor = np.dot(X, Gryffindor_w)
    prediction_Slytherin = np.dot(X, Slytherin_w)
    prediction_Ravenclaw = np.dot(X, Ravenclaw_w)
    prediction_Hufflepuff = np.dot(X, Hufflepuff_w)

    houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
    scores = np.column_stack([prediction_Gryffindor, prediction_Slytherin, prediction_Ravenclaw, prediction_Hufflepuff])
    predicted = np.array([houses[i] for i in np.argmax(scores, axis=1)])
    precision = (predicted == y).mean()
    print(f"Precision for predict the training data: {(precision * 100):.2f}%")

#--------Output

    weights_dict = {
        "features": valid_feat,
        "Gryffindor": Gryffindor_w.tolist(),
        "Slytherin": Slytherin_w.tolist(),
        "Ravenclaw": Ravenclaw_w.tolist(),
        "Hufflepuff": Hufflepuff_w.tolist(),
    }

    print("\033[33m### Generate weights.json for predict ... ####\033[0m")

    with open("weights.json", "w") as weights_file:
            json.dump(weights_dict, weights_file)

if __name__ == '__main__':
    main()
