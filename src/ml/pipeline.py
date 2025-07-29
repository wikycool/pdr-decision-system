import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def train_cost_model(csv_path: str, output_model: str):
    df = pd.read_csv(csv_path)
    y = df['price']
    X = df.drop(columns=['price', 'part_id'])
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(include='object').columns.tolist()

    preproc = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    pipe = Pipeline([('pre', preproc), ('model', GradientBoostingRegressor())])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, output_model)
    print(f"Model saved to {output_model}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    parser.add_argument('model_out')
    args = parser.parse_args()
    train_cost_model(args.csv, args.model_out)