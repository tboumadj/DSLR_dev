import json

from matplotlib.ticker import MaxNLocator
import numpy as np
from utils.load import is_numeric_column, load_csv, dataset_to_dataframe
from utils.train import standardize_feat, extract_X_y
import argparse
from matplotlib import pyplot as plt

EPOCHS = 150
LEARNING_RATE = 0.1
EXCLUDE = ['Index',
           'Arithmancy',
           'Astronomy',
           'Divination']
DATASET_PATH = './datasets/dataset_train.csv'

HOUSE_COLORS = {
    'Gryffindor': '#C84B31',
    'Slytherin':  '#2D6A4F',
    'Ravenclaw':  '#1D3557',
    'Hufflepuff': '#E9C46A',
}
HOUSES = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]

class PlotLoss:

    def update_graph(self, house ,loss, iter):
        self.iters[house].append(iter)
        self.loss[house].append(loss)
        self.lines[house].set_data(self.iters[house], self.loss[house])
        self.ax.relim()
        self.ax.autoscale_view(scalex=False, scaley=True)
        plt.pause(0.001)

    def compute_loss(self, X,Y, weights, house):
        prediction = 1 / ( 1 + np.exp(-np.dot(X, weights)))
        y_binary = (house == Y).astype(float)
        epsilon = 1e-15
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        loss = - np.mean(y_binary * np.log(prediction) + (1 - y_binary) * np.log(1 - prediction))
        return (loss)

    def __init__(self, houses):
        plt.ion()
        fig, ax = plt.subplots(figsize=(10,6))
        self.ax = ax
        self.lines =  {}
        self.iters = {}
        self.loss = {}
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Cost function progression during training")
        ax.set_xlim(0, EPOCHS)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune=None))
        for house in houses:
            color = HOUSE_COLORS.get(house)
            self.lines[house], = ax.plot([], [], label=house, color=color, linewidth=2)
            self.iters[house] = []
            self.loss[house] = []
        ax.legend(loc="upper right")
        plt.show(block=False)
        fig.tight_layout()

    def stop_interactive(self):
        plt.ioff()
        plt.show()


def train_model(valid_feat, X, Y, house, sample_size, graph):
    
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
        predictions = 1 / (1 + np.exp(-np.dot(X_batch, weights))) 

        gradient = 1 / len(X_batch) * np.dot(X_batch.T, predictions - (house == Y_batch) ) 

        weights = weights - LEARNING_RATE * gradient
        if graph:
            graph.update_graph(house, graph.compute_loss(X_batch, Y_batch, weights, house), i)
    
    return (weights)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", action="store", type=int, help="Specify batch size for gradient descent")
    parser.add_argument("-g", "--graph", action="store_true", help="Display the Loss progression during training")
    args = parser.parse_args()
    return (args)

def prepare_data(dataset, args):
    valid_feat = [
        col for col, vals in dataset.items()
        if is_numeric_column(vals) and col not in EXCLUDE
    ]
    
    dataframe = dataset_to_dataframe(DATASET_PATH, valid_feat)

    dataframe_stand, params = standardize_feat(dataframe, valid_feat)
    X, y = extract_X_y(dataframe_stand, valid_feat)
    return (X, y, valid_feat)

def get_sample_size(args, X):
    sample_size = len(X)
    if (args.batch_size != None):
        if (args.batch_size == 1):
             print("mode: pure stochastic gradient descent")
             sample_size = 1
        else: 
             print(f"mode: mini batch descent gradient with {args.batch_size} samples")
             sample_size = args.batch_size
    return (sample_size)

def init_graph(args):
    if args.graph:
        return PlotLoss(HOUSES)
    else:
        return None

def start_training(valid_feat, X, y, sample_size, graph):
    np.random.seed(0)
    w = {}

    w[HOUSES[0]] = train_model(valid_feat, X, y, HOUSES[0], sample_size, graph)
    w[HOUSES[1]]= train_model(valid_feat, X, y, HOUSES[1], sample_size, graph)
    w[HOUSES[2]] = train_model(valid_feat, X, y, HOUSES[2], sample_size, graph)
    w[HOUSES[3]] = train_model(valid_feat, X, y, HOUSES[3], sample_size, graph)
    return (w)

def print_precision(X, y, w):

    X = np.hstack([np.ones((X.shape[0], 1)), X])

    prediction_Gryffindor = np.dot(X, w[HOUSES[0]])
    prediction_Slytherin = np.dot(X, w[HOUSES[1]])
    prediction_Ravenclaw = np.dot(X, w[HOUSES[2]])
    prediction_Hufflepuff = np.dot(X, w[HOUSES[3]])

    scores = np.column_stack([prediction_Gryffindor, prediction_Slytherin, prediction_Ravenclaw, prediction_Hufflepuff])
    predicted = [HOUSES[i] for i in np.argmax(scores, axis=1)]
    precision = np.array((predicted == y).mean())
    print(f"\033[32m### Model precision: {(precision * 100):.2f}% ####\033[0m")

def write_output(valid_feat, w):

    weights_dict = {
        "features": valid_feat,
        "Gryffindor": w[HOUSES[0]].tolist(),
        "Slytherin": w[HOUSES[1]].tolist(),
        "Ravenclaw": w[HOUSES[2]].tolist(),
        "Hufflepuff": w[HOUSES[3]].tolist(),
    }

    print("\033[33m### Successfully generated weights.json ####\033[0m")

    with open("weights.json", "w") as weights_file:
            json.dump(weights_dict, weights_file)


def main():

    args = parse_args()
    dataset = load_csv(DATASET_PATH)
    X, y, valid_feat = prepare_data(dataset, args)
    sample_size = get_sample_size(args, X)
    graph = init_graph(args)
    w = start_training(valid_feat, X, y, sample_size, graph)
    if graph:
        graph.stop_interactive()
    print_precision(X, y, w)
    write_output(valid_feat, w)
 

if __name__ == '__main__':
    main()
