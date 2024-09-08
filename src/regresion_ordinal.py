import statsmodels.api as sm
from sklearn.metrics import accuracy_score
import pandas as pd

def regression_logistic_ordinal(df, target_variable, predictor_variables):
    
    for variable in predictor_variables:
        df[variable] = df[variable].astype('category')
        
    y = df[target_variable] 
    X = df[predictor_variables]
    
    X = sm.add_constant(X)

    model_ordinal = sm.MNLogit(y, X).fit()

    print(model_ordinal.summary())
    
    predicciones = model_ordinal.predict(X)
    predicciones_clases = predicciones.idxmax(axis=1)  
    
    accuracy = accuracy_score(y, predicciones_clases)
    print(f"Accuracy del modelo: {accuracy:.4f}")

    return accuracy

predictor_variables = ['Student_Age', 'Additional_work', 'Graduated_high_school_type']
target_variable = 'Cumulative_GPA'

accuracy = regression_logistic_ordinal(df, target_variable, predictor_variables)

