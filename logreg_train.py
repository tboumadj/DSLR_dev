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
BOWL_STATE = None


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


def select_bowl_dimensions(valid_feat):
    if len(valid_feat) >= 2:
        return 1, 2, valid_feat[0], valid_feat[1]
    return 0, 1, "bias", valid_feat[0]


def compute_loss_surface(X, Y, house, dim_x, dim_y, grid_x, grid_y):
    y_binary = (Y == house).astype(float)
    base_logits = np.zeros(X.shape[0])
    x_component = X[:, dim_x][:, None, None] * grid_x[None, :, :]
    y_component = X[:, dim_y][:, None, None] * grid_y[None, :, :]

    logits = base_logits[:, None, None] + x_component + y_component
    predictions = 1 / (1 + np.exp(-logits))
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)

    y_binary = y_binary[:, None, None]
    loss = -np.mean(y_binary * np.log(predictions) + (1 - y_binary) * np.log(1 - predictions), axis=0)
    return loss


def init_loss_bowl_plot(houses, valid_feat, X, Y):
    plt.ion()
    dim_x, dim_y, label_x, label_y = select_bowl_dimensions(valid_feat)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    x_values = np.linspace(-15, 15, 60)
    y_values = np.linspace(-15, 15, 60)
    grid_x, grid_y = np.meshgrid(x_values, y_values)

    bowl_state = {
        "fig": fig,
        "axes": axes,
        "dim_x": dim_x,
        "dim_y": dim_y,
        "label_x": label_x,
        "label_y": label_y,
    }

    for ax, house in zip(axes, houses):
        loss_surface = compute_loss_surface(X, Y, house, dim_x, dim_y, grid_x, grid_y)
        contour = ax.contourf(grid_x, grid_y, loss_surface, levels=30, cmap="viridis")
        ax.contour(grid_x, grid_y, loss_surface, levels=15, colors="white", alpha=0.2, linewidths=0.6)
        trajectory_line, = ax.plot([], [], color="red", linewidth=1.8)
        current_point, = ax.plot([], [], marker="o", color="white", markersize=7, linestyle="None")
        gradient_arrow = ax.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1, color="red")

        ax.set_title(f"{house} loss bowl")
        ax.set_xlabel(f"{label_x} weight")
        ax.set_ylabel(f"{label_y} weight")
        ax.set_xlim(x_values[0], x_values[-1])
        ax.set_ylim(y_values[0], y_values[-1])

        bowl_state[house] = {
            "trajectory_line": trajectory_line,
            "current_point": current_point,
            "gradient_arrow": gradient_arrow,
            "iterations": [],
            "x_positions": [],
            "y_positions": [],
            "contour": contour,
        }

    fig.colorbar(contour, ax=axes.tolist(), shrink=0.85, label="Loss")
    fig.tight_layout()
    plt.show(block=False)
    return bowl_state


def update_cost_plot(house, iteration, cost, weights, gradient):
    global PLOT_STATE, BOWL_STATE

    if PLOT_STATE is not None:
        house_state = PLOT_STATE[house]
        house_state["iterations"].append(iteration)
        house_state["costs"].append(cost)
        house_state["line"].set_data(house_state["iterations"], house_state["costs"])

        ax = PLOT_STATE["ax"]
        ax.relim()
        ax.autoscale_view()
        PLOT_STATE["fig"].canvas.draw_idle()
        PLOT_STATE["fig"].canvas.flush_events()

    if BOWL_STATE is not None and house in BOWL_STATE:
        house_state = BOWL_STATE[house]
        dim_x = BOWL_STATE["dim_x"]
        dim_y = BOWL_STATE["dim_y"]

        x_value = weights[dim_x]
        y_value = weights[dim_y]
        house_state["iterations"].append(iteration)
        house_state["x_positions"].append(x_value)
        house_state["y_positions"].append(y_value)
        house_state["trajectory_line"].set_data(house_state["x_positions"], house_state["y_positions"])
        house_state["current_point"].set_data([x_value], [y_value])

        gradient_xy = -np.array([gradient[dim_x], gradient[dim_y]])
        gradient_norm = np.linalg.norm(gradient_xy)
        if gradient_norm > 0:
            gradient_xy = gradient_xy / gradient_norm * 2.0
        house_state["gradient_arrow"].set_offsets(np.array([[x_value, y_value]]))
        house_state["gradient_arrow"].set_UVC([gradient_xy[0]], [gradient_xy[1]])

        BOWL_STATE["fig"].canvas.draw_idle()
        BOWL_STATE["fig"].canvas.flush_events()

    plt.pause(0.001)

def train_model(valid_feat, X, Y, house, sample_size):
    global PLOT_STATE, BOWL_STATE

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

        update_cost_plot(house, i + 1, compute_cost(X, Y, weights, house), weights, gradient)
    
    return (weights)


def main():

    global PLOT_STATE, BOWL_STATE

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
    BOWL_STATE = init_loss_bowl_plot(houses, valid_feat, X, y)

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
