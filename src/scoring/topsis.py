import numpy as np
import pandas as pd

def topsis(df: pd.DataFrame, weights: pd.Series, beneficial: list) -> pd.Series:
    # Normalize
    X = df.values
    norm = X / np.sqrt((X**2).sum(axis=0))
    # Weighted
    V = norm * weights.values
    # Ideals
    pos = V.max(axis=0) * beneficial + V.min(axis=0) * (~np.array(beneficial))
    neg = V.min(axis=0) * beneficial + V.max(axis=0) * (~np.array(beneficial))
    # Distances
    d_pos = np.sqrt(((V - pos)**2).sum(axis=1))
    d_neg = np.sqrt(((V - neg)**2).sum(axis=1))
    return pd.Series(d_neg / (d_pos + d_neg), index=df.index)

if __name__ == '__main__':
    df = pd.DataFrame({'c1':[8,9,7],'c2':[9,3,1]})
    w = pd.Series([0.6,0.4])
    print(topsis(df, w, [True, True]))