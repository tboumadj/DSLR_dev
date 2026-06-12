import json

def standardize_feat_train(df, valid_feat):
    params = {}
    df_stand = df.copy()

    for col in valid_feat:
        mean = df[col].mean()
        std  = df[col].std()
        params[col] = {'mean': mean, 'std': std}
        df_stand[col] = (df[col] - mean) / std if std != 0 else 0.0

    return df_stand, params

def standardize_feat_predict(df, valid_feat, params):
    df_stand = df.copy()

    for col in params:
        mean = params[col]['mean']
        std  = params[col]['std']
        df_stand[col] = (df[col] - mean) / std if std != 0 else 0.0

    return df_stand

def extract_X_y(df, valid_feat):
    X = df[valid_feat].values
    y = df['Hogwarts House'].values
    return X, y
