import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

in_path = r'C:\\MachineLearningPG\\Recursos\\cs-training.csv'
out_path = r'C:\\MachineLearningPG\\data\\processed_train.csv'
rep_path = r'C:\\MachineLearningPG\\reports'

os.makedirs(os.path.dirname(out_path), exist_ok=True)
os.makedirs(rep_path, exist_ok=True)

print('Leyendo', in_path)
df = pd.read_csv(in_path)
print('Filas iniciales:', len(df))

cleaned = df.copy()
# 1) NumberOfTimes90DaysLate > 17
if 'NumberOfTimes90DaysLate' in cleaned.columns:
    cleaned = cleaned[cleaned['NumberOfTimes90DaysLate'] <= 17]
# 2) 96/98 in delay cols
cols_delay = [c for c in ['NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTime30-59DaysPastDueNotWorse'] if c in cleaned.columns]
if cols_delay:
    mask_bad = cleaned[cols_delay].isin([96,98]).all(axis=1)
    cleaned = cleaned[~mask_bad]
# 3) RevolvingUtilizationOfUnsecuredLines < 13
if 'RevolvingUtilizationOfUnsecuredLines' in cleaned.columns:
    cleaned = cleaned[cleaned['RevolvingUtilizationOfUnsecuredLines'] < 13]
# 4) DebtRatio <= 97.5 pct
if 'DebtRatio' in cleaned.columns:
    p975 = cleaned['DebtRatio'].dropna().quantile(0.975)
    cleaned = cleaned[cleaned['DebtRatio'] <= p975]

print('Filas tras limpieza:', len(cleaned))

processed = cleaned.copy()
if 'MonthlyIncome' in processed.columns:
    processed['MonthlyIncome_na'] = processed['MonthlyIncome'].isna().astype(int)
    processed['MonthlyIncome'] = processed['MonthlyIncome'].fillna(processed['MonthlyIncome'].median())
if 'NumberOfDependents' in processed.columns:
    processed['NumberOfDependents_na'] = processed['NumberOfDependents'].isna().astype(int)
    try:
        mode_val = processed['NumberOfDependents'].mode().iloc[0]
    except Exception:
        mode_val = 0
    processed['NumberOfDependents'] = processed['NumberOfDependents'].fillna(mode_val)

processed.to_csv(out_path, index=False)
print('Guardado en', out_path)

# quick feature importances
if 'SeriousDlqin2yrs' in processed.columns:
    feat_cols = [c for c in processed.columns if c not in ['SeriousDlqin2yrs', 'Id']]
    X = processed[feat_cols].fillna(0)
    y = processed['SeriousDlqin2yrs']
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    importances.to_csv(os.path.join(rep_path,'feature_importances.csv'))
    print('Feature importances guardadas en', os.path.join(rep_path,'feature_importances.csv'))

print('Hecho')
