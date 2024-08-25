import pandas as pd
import csv



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
