import pandas as pd

# Function to load the CSV file
def load_data(path):
    df = pd.read_csv(path)
    df = df.fillna(method='ffill')   # fill missing values
    return df

# Function to split into features (X) and target (y)
def split_xy(df, feature, target):
    X = df[[feature]].values
    y = df[target].values
    return X, y
