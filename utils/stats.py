import matplotlib.pyplot as plt

HOUSE_COLORS = {
    'Gryffindor': '#C84B31',
    'Slytherin':  '#2D6A4F',
    'Ravenclaw':  '#1D3557',
    'Hufflepuff': '#E9C46A',
}

# -----------Merge
def merge_sort(lst):
    if len(lst) <= 1:
        return lst.copy()
    
    mid = len(lst) // 2
    left  = merge_sort(lst[:mid])
    right = merge_sort(lst[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# -------------Percentile
def percentile(sorted_data, p):
    n = len(sorted_data)
    if n == 0:
        return None

    # List position
    index = (p / 100) * (n - 1)

    lower = int(index)
    upper = lower + 1
    frac  = index - lower

    if upper >= n:
        return sorted_data[lower]

    # Interpolation linéaire entre les deux voisins index flottant = (42/100) × (10−1) = 3.780 -> lower = 3  ·  upper = 4  ·  frac = 0.780 -> résultat = data[3] + frac × (data[4] − data[3])
    return sorted_data[lower] + frac * (sorted_data[upper] - sorted_data[lower])

#-----------------------
def describe_feature(column):
    # Count number data valid clear NaN
    clean = [x for x in column if x == x]
    n = len(clean)

    if n == 0:
        return {k: None for k in ["count","mean","std","min","25%","50%","75%","max"]}

    # Tri
    sorted_data = merge_sort(clean)

    count = n

    # Mean
    mean = sum(clean) / n

    # Std (écart-type, version échantillon : divisé par N-1 bessel correction)
    variance = sum((x - mean) ** 2 for x in clean) / (n - 1)
    std = variance ** 0.5

    # Min / Max
    minimum = sorted_data[0]
    maximum = sorted_data[-1]

    # Percentiles
    p25 = percentile(sorted_data, 25)
    p50 = percentile(sorted_data, 50)
    p75 = percentile(sorted_data, 75)

    return {
        "count": count,
        "mean":  mean,
        "std":   std,
        "min":   minimum,
        "25%":   p25,
        "50%":   p50,
        "75%":   p75,
        "max":   maximum,
    }

#--------------Printer
def print_describe(stats_by_feature):
    row_labels = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    stat_keys  = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    features = list(stats_by_feature.keys())

    col_w  = 14   # largeur de chaque colonne de données
    label_w = 6   # largeur de la colonne de labels

    # Head
    header = ' ' * label_w
    for feat in features:
        # Chunk name
        header += feat[:col_w].rjust(col_w)
    print(header)

    # Stats line
    for label, key in zip(row_labels, stat_keys):
        row = label.ljust(label_w)
        for feat in features:
            val = stats_by_feature[feat][key]
            if val is None:
                row += 'NaN'.rjust(col_w)
            else:
                row += f'{val:>{col_w}.6f}'
        print(row)

def print_graph_scatter(data_house, feat1, feat2):
    # Matplotlib
    print("\033[33m#### Matplot Test: ####\033[0m")    
    plt.figure(figsize=(20, 12))
    
    for house, (x, y) in data_house.items():
        color = HOUSE_COLORS.get(house, 'gray')
        plt.scatter(x, y,
                    label=f'{house} ({len(x)})',
                    color=color,
                    s=12,
                    alpha=0.6)

    #-----------
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.title(f'{feat1} vs {feat2}')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

#-----------------------Test Linear + cov
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
    print("\033[33m#### Matplot Test: ####\033[0m")    
    plt.figure(figsize=(20, 12))

    all_x, all_y = [], []
    
    for house, (x, y) in data_house.items():
        color = HOUSE_COLORS.get(house, 'gray')
        plt.scatter(x, y,
                    label=f'{house} ({len(x)})',
                    color=color,
                    s=12,
                    alpha=0.6)

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