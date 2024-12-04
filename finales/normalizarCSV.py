import pandas as pd

# Nombre del archivo CSV original
input_file = "politica.csv"
# Nombre del archivo CSV normalizado
output_file = "politicaN.csv"

# Leer el archivo CSV
df = pd.read_csv(input_file)

# Convertir todo el texto a minúsculas
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# Guardar el archivo normalizado
df.to_csv(output_file, index=False)

print(f"El archivo normalizado se ha guardado como: {output_file}")
