import pandas as pd
import matplotlib.pyplot as plt
from utils.load import get_xy

HOUSE_COLORS = {
    'Gryffindor': '#C84B31',
    'Slytherin':  '#2D6A4F',
    'Ravenclaw':  '#1D3557',
    'Hufflepuff': '#E9C46A',
}

def pandas_feature(stats, filepath):
    df = pd.read_csv(filepath)
    pd_stats = df.describe()

    stat_keys  = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    pd_keys    = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    features = list(stats.keys())

    col_w   = 14
    label_w = 8

    # Tab pandas

    print("\033[32m#### Stats By Pandas ####\033[0m")
    header = ' ' * label_w
    for feat in features:
        header += feat[:col_w].rjust(col_w)
    print(header)

    for label, pd_key in zip(["Count","Mean","Std","Min","25%","50%","75%","Max"], pd_keys):
        row = label.ljust(label_w)
        for feat in features:
            if feat in pd_stats.columns:
                val = pd_stats.loc[pd_key, feat]
                row += f'{val:>{col_w}.6f}'
            else:
                row += 'N/A'.rjust(col_w)
        print(row)

    # Diff
    print("\033[33m#### Diff - Describe vs Pandas ####\033[0m")
    print(' ' * label_w + ''.join(f[:col_w].rjust(col_w) for f in features))

    for label, my_key, pd_key in zip(
        ["Count","Mean","Std","Min","25%","50%","75%","Max"],
        stat_keys, pd_keys
    ):
        row = label.ljust(label_w)
        for feat in features:
            if feat not in pd_stats.columns:
                row += 'N/A'.rjust(col_w)
                continue
            my_val = stats[feat][my_key]
            pd_val = pd_stats.loc[pd_key, feat]
            if my_val is None:
                row += 'None'.rjust(col_w)
            else:
                diff = abs(my_val - pd_val)
                # diff > -10 is ok
                row += f'{diff:>{col_w}.2e}'
        print(row)

#-----------------------Test Linear + cov

def find_correlation(dataset, numeric_cols):
    import itertools

    results = []
    for col_x, col_y in itertools.combinations(numeric_cols, 2):
        x, y = get_xy(dataset, col_x, col_y)
        if len(x) < 2:
            continue

        a, b     = linear_regression(x, y)
        mean_x   = sum(x) / len(x)
        mean_y   = sum(y) / len(y)
        cov      = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        std_x    = (sum((xi - mean_x) ** 2 for xi in x) / (len(x) - 1)) ** 0.5
        std_y    = (sum((yi - mean_y) ** 2 for yi in y) / (len(y) - 1)) ** 0.5

        r = cov / ((len(x) - 1) * std_x * std_y) if std_x and std_y else 0
        results.append((abs(r), r, col_x, col_y))

    # Tri par corrélation décroissante
    results.sort(reverse=True)

    print(f"\n{'r':>8}  {'feat1':<35} {'feat2'}")
    print('-' * 75)
    for _, r, cx, cy in results[:40]: 
        print(f"{r:>8.4f}  {cx:<35} {cy}")


def linear_regression(x, y):
    n = len(x)

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov    = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x  = sum((xi - mean_x) ** 2 for xi in x)

    if var_x == 0:
        return 0, mean_y

    a = cov / var_x
    b = mean_y - a * mean_x

    return a, b

def print_graph_test(data_house, feat1, feat2):
    # Matplotlib
    print("\033[33m#### Matplot Graph ####\033[0m")
    plt.figure(figsize=(20, 12), dpi=120)

    all_x, all_y = [], []
    
    for house, (x, y) in data_house.items():
        color = HOUSE_COLORS.get(house, 'gray')
        plt.scatter(x, y,
                    label=f'{house} ({len(x)})',
                    color=color,
                    s=18,
                    alpha=0.55)

    #Covariance logic 
    all_x.extend(x)
    all_y.extend(y)
    a, b = linear_regression(all_x, all_y)
    x_min = min(all_x)
    x_max = max(all_x)
    plt.plot([x_min, x_max],
            [a * x_min + b, a * x_max + b],
            color='black',
            linewidth=1.5,
            linestyle='--')
    
    #-----------
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.title(f'{feat1} vs {feat2}')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()