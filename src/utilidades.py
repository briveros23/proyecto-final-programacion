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

def plot_coef_multiclass_logreg(model, feature_names, block_size=6):
    """
    Genera gráficos de coeficientes para un modelo de regresión logística multiclase, mostrando bloques de 4 variables a la vez.

    Parámetros:
    - model: modelo entrenado de LogisticRegression
    - feature_names: lista de nombres de las variables (por ejemplo, de OneHotEncoder)
    - block_size: número de variables por gráfico (por defecto 4)
    """
    # Obtener los coeficientes del modelo
    coef = model.coef_

    # Crear un DataFrame con los coeficientes
    coef_df = pd.DataFrame(coef, columns=feature_names, index=model.classes_)

    # Calcular cuántos bloques de gráficos necesitaremos
    num_features = len(feature_names)
    num_blocks = int(np.ceil(num_features / block_size))
    
    # Iterar sobre los bloques y generar gráficos
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, num_features)

        # Seleccionar las variables de este bloque
        coef_block = coef_df.iloc[:, start_idx:end_idx]

        # Graficar el bloque
        coef_block.T.plot(kind='bar', figsize=(10, 6))

        # Añadir título y etiquetas
        plt.title(f'Coeficientes de la Regresión Logística (Bloque {i+1})')
        plt.ylabel('Valor del Coeficiente')
        plt.xlabel('Variables Independientes')

        # Mostrar la leyenda (categorías de la variable de respuesta)
        plt.legend(title='Categoría de Respuesta', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Ajustar diseño
        plt.tight_layout()

        # Mostrar gráfico
        plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Función para calcular y graficar la matriz de confusión.
    
    Parameters:
    y_true: Valores reales (verdaderos).
    y_pred: Valores predichos por el modelo.
    class_names: Lista de nombres de las clases.
    """
    # Calcular la matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Graficar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    
    # Agregar etiquetas y título
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
