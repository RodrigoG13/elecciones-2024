import pandas as pd
import csv

# Cargar el archivo CSV
archivo_csv = "dataset_etiquetado.csv"  # Reemplaza con la ruta de tu archivo CSV
df = pd.read_csv(archivo_csv)

# Convertir las columnas "xochitl" y "Xóchitl" a tipo cadena y reemplazar NaN por cadena vacía
df['xochitl'] = df['xochitl'].astype(str).replace('nan', '')
df['Xóchitl'] = df['Xóchitl'].astype(str).replace('nan', '')

# Unir las columnas en una sola columna "Xóchitl"
df['Xóchitl'] = df['xochitl'] + df['Xóchitl']

# Eliminar la columna "xochitl" si ya no es necesaria
df = df.drop(columns=['xochitl'])

# Convertir valores numéricos en "Xóchitl" a enteros (sin afectar texto o valores vacíos)
def convertir_a_entero(valor):
    try:
        return int(float(valor))  # Convertir si es numérico
    except (ValueError, TypeError):
        return valor  # Mantener texto o valores vacíos intactos

df['Xóchitl'] = df['Xóchitl'].apply(lambda x: x if x == '' else convertir_a_entero(x))
df['num_debate'] = df['num_debate'].apply(lambda x: x if x == '' else convertir_a_entero(x))
df['num_likes'] = df['num_likes'].apply(lambda x: x if x == '' else convertir_a_entero(x))
df['Claudia'] = df['Claudia'].apply(lambda x: x if x == '' else convertir_a_entero(x))
df['Ninguno'] = df['Ninguno'].apply(lambda x: x if x == '' else convertir_a_entero(x))
df['Maynez'] = df['Maynez'].apply(lambda x: x if x == '' else convertir_a_entero(x))

# Convertir otras columnas numéricas a int, pero solo si no tienen valores vacíos
for columna in ['Claudia', 'Maynez', 'Xóchitl', 'Ninguno']:
    df[columna] = df[columna].apply(lambda x: int(float(x)) if pd.notna(x) and x != '' else x)

# Guardar el DataFrame actualizado, asegurando que cada valor esté entrecomillado
archivo_salida = "dataset_etiquetado2.csv"
df.to_csv(archivo_salida, index=False, quoting=csv.QUOTE_ALL, quotechar='"', encoding='utf-8')

print("El archivo CSV ha sido guardado correctamente sin rellenar columnas vacías con ceros.")
