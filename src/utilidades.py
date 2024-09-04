import pandas as pd
import csv
import prince
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def mapeo_datos(path_datos_ori,path_datos_mapeados,path_mapeo):
    df_performance = pd.read_csv(path_datos_ori)

    with open(path_mapeo, 'r', errors = 'ignore') as csvfile:
        lector_csv = csv.reader(csvfile)
        var_codes = list(lector_csv)

    variables = df_performance.columns[1:]

    mapeo = {}

    for var in variables:
        mapeo[var] = {}
        for row in var_codes:
            if row[0] == var:
                mapeo[var][int(row[1])] = row[2]

    for var in variables:
        df_performance[var] = df_performance[var].replace(mapeo[var])

    df_performance

    df_performance.to_csv(path_datos_mapeados, index=False)


def planos_factoriales(ACM,data,planos=[0,1]):
    # Obtener los perfiles de fila y columna
    row_profiles = ACM.row_coordinates(data)[planos]
    col_profiles = ACM.column_coordinates(data)[planos]
    # ponemos los index como una columna llamada individuo
    row_profiles['nombre'] = row_profiles.index
    col_profiles['nombre'] = col_profiles.index
    # unimos los perfiles de fila y columna
    df = pd.concat([row_profiles, col_profiles])
    # añadimos una columna para diferenciar los perfiles de fila y columna
    df['tipo'] = ['individuo']*len(row_profiles) + ['columna']*len(col_profiles)
    # Cambiamos el nombre de las columnas
    df.columns = ['x','y','nombre','tipo']
    # retornamos el dataframe
    return df

def grafico_planos_factoriales(df,df_valores_porpios,planos=[0,1]):
    # sacamos los pesos de cada eje
    peso1 = df_valores_porpios.loc[planos[0],'% of variance']
    peso2 = df_valores_porpios.loc[planos[1],'% of variance']
    # Crear gráfico de dispersión
    fig = px.scatter(df, x='x', y='y', color='tipo', title='Gráfico de Dispersión con Variable Cualitativa',hover_name='nombre')
    # cambiamos tamaño
    fig.update_layout(width=850, height=800)
    # Añadir etiquetas a las columnas
    fig.update_layout(
        xaxis_title=f'Componente {str(planos[0]+1)} ({str(peso1)}%)',
        yaxis_title=f'Componente {str(planos[1]+1)} ({str(peso2)}%)'
    )

    # Mostrar gráfico
    fig.show()


def entrenamiento_modelo(Modelo,x_entrenamiento,y_entrenamiento):
    model = Modelo
    model.fit(x_entrenamiento, y_entrenamiento)
    return model

def prediccion_modelo(modelo,x_prueba):
    return modelo.predict(x_prueba)

def evaluacion_modelo(y_prueba,y_pred):
    accuracy = accuracy_score(y_prueba, y_pred)
    conf_matrix = confusion_matrix(y_prueba, y_pred)
    class_report = classification_report(y_prueba, y_pred)
    print(f'class_report: {class_report}')
    return [accuracy,conf_matrix]

def grafico_los_coeficientes(modelo,data):
    coefs = modelo.coef_

    # Crear un gráfico para cada clase
    n_classes = coefs.shape[0]
    features = data.feature_names

    plt.figure(figsize=(10, 6))

    for i in range(n_classes):
        plt.barh(np.arange(len(features)) + i * 0.25, coefs[i], height=0.25, label=f'Clase {i}')

    # Configuraciones adicionales del gráfico
    plt.yticks(np.arange(len(features)) + 0.25 * (n_classes - 1) / 2, features)
    plt.xlabel('Coeficiente')
    plt.title('Parámetros de la Regresión Logística para Cada Clase')
    plt.legend()
    plt.grid(True)
    plt.show()


def grafico_de_matriz(matriz,data):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=data.target_names, yticklabels=data.target_names)

    # Configuración de etiquetas y título
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('Matriz de Confusión Multiclase')
    plt.show()