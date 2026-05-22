import json

def normalize_feat(df, valid_feat):
    params = {}
    df_norm = df.copy()

    for col in valid_feat:
        mean = df[col].mean()
        std  = df[col].std()
        params[col] = {'mean': mean, 'std': std}
        df_norm[col] = (df[col] - mean) / std if std != 0 else 0.0
    return df_norm, params

def extract_X_y(df, valid_feat):
    X = df[valid_feat].values.tolist()
    y = df['Hogwarts House'].tolist()
    return X, y

def save_config(house):
    return
