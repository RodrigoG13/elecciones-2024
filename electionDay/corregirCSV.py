import csv

name = "22a23"
# Archivo de entrada y salida
archivo_entrada = f"{name}.csv"  # Cambia a tu archivo
archivo_salida = f"{name}_c.csv"


# Leer y transformar el archivo
with open(archivo_entrada, mode="r", encoding="utf-8") as entrada, \
     open(archivo_salida, mode="w", encoding="utf-8", newline="") as salida:
    
    lector = csv.DictReader(entrada)
    campos = lector.fieldnames  # Obtener los nombres de las columnas
    escritor = csv.DictWriter(salida, fieldnames=campos)
    
    escritor.writeheader()  # Escribir el encabezado
    
    for fila in lector:
        # Reemplazar comillas en la columna 'publicación'
        publicacion = fila["publicación"]
        fila["publicación"] = f'"""{publicacion}"""'
        escritor.writerow(fila)

print(f"El archivo ha sido modificado y guardado como {archivo_salida}")
