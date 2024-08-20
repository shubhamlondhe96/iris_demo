# load_data.py
import pandas as pd
from sklearn.datasets import load_iris

def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
