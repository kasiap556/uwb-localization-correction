import os
import glob
import pandas as pd
import numpy as np

def load_data(base_path, folders, subfolder):
    data_frames = []
    for f in folders:
        path = os.path.join(base_path, f, subfolder, '*.csv')
        files = glob.glob(path)
        for x in files:
            df = pd.read_csv(x, header=None)
            data_frames.append(df)
    if not data_frames:
        return pd.DataFrame()
    return pd.concat(data_frames, ignore_index=True)

def prepare_datasets(data_dir):
    train_df = load_data(data_dir, ['f8', 'f10'], 'stat')
    X_train = train_df.iloc[:, 0:2].values.astype(float)
    y_train = train_df.iloc[:, 2:4].values.astype(float)

    test_df = load_data(data_dir, ['f8', 'f10'], 'dyn')
    X_test = test_df.iloc[:, 0:2].values.astype(float)
    y_test = test_df.iloc[:, 2:4].values.astype(float)

    return (X_train, y_train), (X_test, y_test)