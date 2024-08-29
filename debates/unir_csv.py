import os
import pandas as pd

def encontrar_archivos_csv(directorio):
    archivos_csv = []  # Lista para almacenar los nombres de los archivos .csv
    for archivo in os.listdir(directorio):
        if archivo.endswith('.csv'):
            archivos_csv.append(archivo)  # Agrega el archivo a la lista si termina en .csv
    return archivos_csv

# Usando '.' para referirse al directorio actual
directorio = '.'
archivos_csv = encontrar_archivos_csv(directorio)



# Leer y concatenar todos los archivos en un solo DataFrame
dataframes = [pd.read_csv(archivo) for archivo in archivos_csv]
comentarios_debates = pd.concat(dataframes, ignore_index=True)

# Guardar el DataFrame resultante en un nuevo archivo CSV
comentarios_debates.to_csv("comentarios_debates.csv", index=False)
