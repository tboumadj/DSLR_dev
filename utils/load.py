import pandas as pd

# ---------Loader
def load_csv(filepath):
    df = pd.read_csv(filepath)
    dataset = {}
    
    for col in df.columns:
        dataset[col] = pd.to_numeric(df[col], errors='coerce').tolist()

    return dataset

def load_house_column(filepath, house_col):
    df = pd.read_csv(filepath)
    
    if house_col not in df.columns:
        raise ValueError(f"House '{house_col}' Missing")
    
    return df[house_col].tolist()

# -------------Getter
def is_numeric_column(values):
    clean = [x for x in values if x == x]
    return len(clean) > len(values) / 2

# Get valid pairs
def get_pairs(dataset, feat1, feat2):
    pairs = []
    for x, y in zip(dataset[feat1], dataset[feat2]):
        if x == x and y == y:
            pairs.append((x, y))
    return pairs

def get_xy(dataset, feat1, feat2):
    pairs = get_pairs(dataset, feat1, feat2)
    return [p[0] for p in pairs], [p[1] for p in pairs]

def get_xy_house(filepath, dataset, feat1, feat2):
    houses = load_house_column(filepath, 'Hogwarts House')
    raw_x = dataset[feat1]
    raw_y = dataset[feat2]

    result = {}

    for x, y, h in zip(raw_x, raw_y, houses):
        if x != x or y != y:
            continue
        if h != h:
            continue
        if h not in result:
            result[h] = ([], [])
        result[h][0].append(x)
        result[h][1]. append(y)

    return result

def get_hist_house(filepath, dataset, feat):
    houses = load_house_column(filepath, 'Hogwarts House')
    raw_x = dataset[feat]

    result = {}

    for x, h in zip(raw_x, houses):
        if x != x:
            continue
        if h != h:
            continue
        if h not in result:
            result[h] = []
        result[h].append(x)

    return result

#------------------

def dataset_to_dataframe(filepath, numeric_cols):

    dataset = pd.read_csv(filepath)
    cols_to_keep = numeric_cols + ['Hogwarts House']
    dataclean = dataset[cols_to_keep]
    result = pd.DataFrame(data = dataclean)
    result = result.dropna()

    return result
