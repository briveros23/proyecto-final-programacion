from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def regresion_linear_y_evaluacion(df_performance):
    X = df_performance[['Student_Age', 'Additional_work', 'Graduated_high_school_type']]
    y = df_performance['Cumulative_GPA']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Evaluación Modelo Regresión lineal: ")
    print("MSE: ", mse)
    print("r2: ", r2)
    
    return model, mse, r2



