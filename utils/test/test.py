import pandas as pd

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

