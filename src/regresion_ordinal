import statsmodels.api as sm
import statsmodels.formula.api as smf

def regresion_ordinal(df_performance):
    model = smf.ols('Cumulative_GPA ~ C(Student_Age) + C(Additional_work) + C(Graduated_high_school_type)', data=df_performance).fit()
    
    print("Resultados Regresión Ordinal: ")
    print(model.summary())
    
    return model

