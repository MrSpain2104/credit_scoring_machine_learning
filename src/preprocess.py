"""Preprocesamiento simple: carga CSV de `Recursos/sampleEntry.csv`, guarda una versión procesada en data/"""
import pandas as pd

def load_sample_entry(path: str):
    return pd.read_csv(path)


def simple_process(df: pd.DataFrame):
    # ejemplo: renombrar columnas y normalizar la probabilidad
    df = df.copy()
    if 'Probability' in df.columns:
        df['probability'] = df['Probability']
        df = df.drop(columns=['Probability'])
    return df


if __name__ == '__main__':
    import os
    in_path = os.path.join('Recursos', 'sampleEntry.csv')
    df = load_sample_entry(in_path)
    df2 = simple_process(df)
    out_path = os.path.join('data', 'sampleEntry_processed.csv')
    df2.to_csv(out_path, index=False)
    print(f"Saved processed sample to {out_path}")
