import pandas as pd

def load_data(true_path, fake_path):
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)
    return df_true, df_fake
