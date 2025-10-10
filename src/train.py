"""Entrenamiento simple: carga `data/sampleEntry_processed.csv`, entrena un regresor dummy y guarda el modelo."""
import pandas as pd
from sklearn.dummy import DummyRegressor
from joblib import dump


def load_data(path: str):
    return pd.read_csv(path)


def train_baseline(X, y):
    model = DummyRegressor(strategy='mean')
    model.fit(X, y)
    return model

if __name__ == '__main__':
    import os
    in_path = os.path.join('data', 'sampleEntry_processed.csv')
    df = load_data(in_path)
    if 'probability' not in df.columns:
        raise SystemExit('Column `probability` not found. Run preprocess first.')
    X = df[['Id']]
    y = df['probability']
    model = train_baseline(X, y)
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'baseline.joblib')
    dump(model, model_path)
    print(f"Saved baseline model to {model_path}")
