# from statsmodels.formula.api import ols as OLS
import statsmodels.api as sm
import pandas as pd

# X_name = ["CRIM"]
X_name = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",	"AGE", "DIS", "RAD", "TAX",	"PTRATIO", "B", "LSTAT"]
Y_name = "MEDV"

df = pd.read_csv("BostonHousing_noNaN_forRegressor.csv")
df_x = df[X_name]
df_y = df[Y_name]

new_df = pd.concat([df_x, df_y], axis=1)
print(new_df.head(10))

X = df_x.values
Y = df_y.values.reshape(-1, 1)

results = sm.OLS(Y, sm.add_constant(X)).fit()
print(results.summary())