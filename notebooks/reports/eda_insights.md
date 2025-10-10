# Resumen EDA automatizado

## Ratio de clases
Total de observaciones (train): 150000
Proporción de casos positivos (SeriousDlqin2yrs=1): 0.0668
{0: 139974, 1: 10026}

## Limpieza y filas eliminadas
Filas iniciales: 150000
Filas tras limpieza: 145755
Filas eliminadas: 4245

## Top features (univariada AUC y RandomForest)
Top 10 por AUC univariada:
1. RevolvingUtilizationOfUnsecuredLines: 0.7778
2. NumberOfTime30-59DaysPastDueNotWorse: 0.6895
3. NumberOfTimes90DaysLate: 0.6571
4. NumberOfTime60-89DaysPastDueNotWorse: 0.6217
5. NumberOfDependents: 0.5468
6. DebtRatio: 0.5238
7. Unnamed: 0: 0.5032
8. NumberRealEstateLoansOrLines: 0.4627
9. NumberOfOpenCreditLinesAndLoans: 0.4555
10. MonthlyIncome: 0.4240
Top 10 por RandomForest (feature_importances):
1. RevolvingUtilizationOfUnsecuredLines: 0.1634
2. Unnamed: 0: 0.1424
3. DebtRatio: 0.1402
4. MonthlyIncome: 0.1152
5. age: 0.1053
6. NumberOfTimes90DaysLate: 0.0900
7. NumberOfOpenCreditLinesAndLoans: 0.0771
8. NumberOfTime30-59DaysPastDueNotWorse: 0.0506
9. NumberOfTime60-89DaysPastDueNotWorse: 0.0437
10. NumberOfDependents: 0.0356

DebtRatio 86% quantile: 453.0000
Se ha observado que RevolvingUtilizationOfUnsecuredLines presenta extremos; se recomienda log1p/clipping para modelado.