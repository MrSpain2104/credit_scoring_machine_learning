# Resumen EDA automatizado

## Origen del dataset
trainingData cargado desde: C:\MachineLearningPG\data\processed_for_modeling.csv

## Ratio de clases
Total de observaciones (train): 150,000
Proporción de casos positivos (SeriousDlqin2yrs=1): 0.0668 (10,026 casos)
{0: 139974, 1: 10026}

## Checks básicos
Número de columnas: 11
Número de filas: 150,000

## Valores faltantes (top)
No se detectaron valores faltantes.

## Outliers (IQR) - top 10 variables con más outliers
- DebtRatio: 31311 outliers (IQR)
- NumberOfTime30-59DaysPastDueNotWorse: 23982 outliers (IQR)
- NumberOfDependents: 13336 outliers (IQR)
- SeriousDlqin2yrs: 10026 outliers (IQR)
- MonthlyIncome: 9149 outliers (IQR)
- NumberOfTimes90DaysLate: 8338 outliers (IQR)
- NumberOfTime60-89DaysPastDueNotWorse: 7604 outliers (IQR)
- NumberOfOpenCreditLinesAndLoans: 3980 outliers (IQR)
- NumberRealEstateLoansOrLines: 793 outliers (IQR)
- RevolvingUtilizationOfUnsecuredLines: 763 outliers (IQR)

## Skewness (top variables sesgadas)
- MonthlyIncome: skew=127.120
- RevolvingUtilizationOfUnsecuredLines: skew=97.631
- DebtRatio: skew=95.157
- NumberOfTime60-89DaysPastDueNotWorse: skew=23.332
- NumberOfTimes90DaysLate: skew=23.087
- NumberOfTime30-59DaysPastDueNotWorse: skew=22.597
- NumberRealEstateLoansOrLines: skew=3.482
- SeriousDlqin2yrs: skew=3.469
- NumberOfDependents: skew=1.626
- NumberOfOpenCreditLinesAndLoans: skew=1.215

## Importancia de features
Top 10 por Mutual Information:
1. RevolvingUtilizationOfUnsecuredLines_log: 0.037102
2. RevolvingUtilizationOfUnsecuredLines: 0.036953
3. RevolvingUtilizationOfUnsecuredLines_clipped: 0.036718
4. NumberOfTimes90DaysLate: 0.036204
5. NumberOfTime30-59DaysPastDueNotWorse: 0.033708
6. NumberOfTime60-89DaysPastDueNotWorse: 0.025062
7. NumberOfDependents: 0.010072
8. age: 0.007214
9. NumberRealEstateLoansOrLines: 0.005896
10. DebtRatio_log: 0.004193
Top 10 por RandomForest (feature_importances):
1. RevolvingUtilizationOfUnsecuredLines: 0.163420
2. Unnamed: 0: 0.142373
3. DebtRatio: 0.140242
4. MonthlyIncome: 0.115227
5. age: 0.105308
6. NumberOfTimes90DaysLate: 0.090011
7. NumberOfOpenCreditLinesAndLoans: 0.077113
8. NumberOfTime30-59DaysPastDueNotWorse: 0.050600
9. NumberOfTime60-89DaysPastDueNotWorse: 0.043700
10. NumberOfDependents: 0.035580

## PCA - varianza explicada (primeros 5 componentes)
- PC1: 0.2995
- PC2: 0.1537
- PC3: 0.1236
- PC4: 0.1018
- PC5: 0.1001

## Recomendaciones de tratamiento y acciones concretas
- Imputar MonthlyIncome por mediana y conservar indicador de NA. Considerar winsorize al 99% si hay valores extremos.
- Aplicar log1p a RevolvingUtilizationOfUnsecuredLines y DebtRatio (ya se creó versión _log en el dataset final).
- Revisar campos con valores 96/98 en columnas de retrasos y tratarlos como NA o errores de entrada.
- Para modelado: usar pipelines con StandardScaler + Logistic/RandomForest; probar class_weight o SMOTE por el desbalance.

Top 10 por RandomForest (recalculado):
1. RevolvingUtilizationOfUnsecuredLines_clipped: 0.130508
2. RevolvingUtilizationOfUnsecuredLines_log: 0.117266
3. RevolvingUtilizationOfUnsecuredLines: 0.113984
4. age: 0.080542
5. MonthlyIncome: 0.077271
6. NumberOfTime30-59DaysPastDueNotWorse: 0.070369
7. DebtRatio: 0.069263
8. DebtRatio_log: 0.068883
9. DebtRatio_clipped: 0.068794
10. NumberOfTimes90DaysLate: 0.066480


## Acción: Regenerado processed_for_modeling.csv
- Dataset regenerado desde: C:\MachineLearningPG\Recursos\cs-training.csv con imputaciones (MonthlyIncome mediana, NumberOfDependents moda/0) y 96/98 tratados como NA en columnas de atraso.
- Nuevas dimensiones: 150,000 filas x 18 columnas.

Top 10 por Mutual Information (recalculado):
1. RevolvingUtilizationOfUnsecuredLines_log: 0.036661
2. RevolvingUtilizationOfUnsecuredLines: 0.036265
3. RevolvingUtilizationOfUnsecuredLines_clipped: 0.036140
4. NumberOfTimes90DaysLate: 0.034845
5. NumberOfTime30-59DaysPastDueNotWorse: 0.032656
6. NumberOfTime60-89DaysPastDueNotWorse: 0.025764
7. NumberOfDependents: 0.011158
8. MonthlyIncome_na: 0.010531
9. age: 0.007981
10. NumberRealEstateLoansOrLines: 0.007284

Top 10 por RandomForest (recalculado, sin Unnamed: 0):
1. RevolvingUtilizationOfUnsecuredLines_clipped: 0.133233
2. RevolvingUtilizationOfUnsecuredLines_log: 0.122832
3. RevolvingUtilizationOfUnsecuredLines: 0.111548
4. age: 0.081033
5. MonthlyIncome: 0.074667
6. DebtRatio: 0.069405
7. DebtRatio_log: 0.068865
8. DebtRatio_clipped: 0.068078
9. NumberOfTime30-59DaysPastDueNotWorse: 0.066866
10. NumberOfTimes90DaysLate: 0.061990
