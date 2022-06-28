from scipy import stats
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def get_statistics(model, intercept=0, coef=[], train_df=None, target_df=None):
    print("Coefficient :", coef)
    print("Intercept :", intercept)

    params = np.append(intercept, coef)
    print("Params:", params)

    # prediction = model.predict(train_df.values.reshape(-1, 1))      # 단변량
    prediction = model.predict(train_df.values)                     # 다변량

    if len(prediction.shape) == 1:
        prediction = np.expand_dims(prediction, axis=1)
    print(train_df.columns)

    new_trainset = pd.DataFrame({"Constant": np.ones(len(train_df.values))}).join(pd.DataFrame(train_df.values))
    print(new_trainset)

    from sklearn.metrics import mean_squared_error
    MSE = mean_squared_error(prediction, target_df.values)
    print("MSE :", MSE)

    variance = MSE * (np.linalg.inv(np.dot(new_trainset.T, new_trainset)).diagonal())       # MSE = (1, ) & else = (n, ) 가 나와야 함.

    std_error = np.sqrt(variance)
    t_values = params / std_error
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(new_trainset) - len(new_trainset.columns) - 1))) for i in t_values]

    std_error = np.round(std_error, 3)
    t_values = np.round(t_values, 3)
    p_values = np.round(p_values, 3)
    params = np.round(params, 4)

    statistics = pd.DataFrame()
    statistics["Coefficients"], statistics["Standard Errors"], statistics["t -values"], statistics["p-values"] = [params, std_error, t_values, p_values]

    return statistics


if __name__ == "__main__":
    # X_name = ["CRIM"]
    X_name = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    Y_name = ["MEDV"]

    model = LinearRegression(fit_intercept=True)

    df = pd.read_csv("BostonHousing_noNaN_forRegressor.csv")
    print("Dataset - Total CSV Dataframe")
    print(df)
    df_x = df[X_name]
    df_y = df[Y_name]
    df_x.columns = [X_name]
    df_y.columns = [Y_name]
    print("Dataset - CSV X value")
    print(df_x)
    print("Dataset - CSV Y value")
    print(df_y, "\n\n")

    # model.fit(df_x.values.reshape(-1,1), df_y.values)   # 단변량
    model.fit(df_x.values, df_y.values)               # 다변량

    _coef = model.coef_
    if len(_coef.shape) == 1:
        _coef = np.expand_dims(_coef, axis=-1)
    coef_list = []
    for index in range(0, _coef.shape[0]):
        for elem in range(0, _coef.shape[1]):
            coef_list.append(float(_coef[index][elem]))

    _intercept = model.intercept_

    statistics_df = get_statistics(model=model, intercept=_intercept, coef=coef_list, train_df=df_x, target_df=df_y)
    print(statistics_df)

    variables = ['intercept', *X_name]
    variables_df = pd.DataFrame(variables, columns=['Variables'])
    total_df = pd.concat([variables_df, statistics_df], axis=1)
    total_df = total_df.set_index("Variables")

    print(total_df)