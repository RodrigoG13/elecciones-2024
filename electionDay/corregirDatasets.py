import csv

name = "15a16"
# Archivo de entrada y salida
archivo_entrada = f"{name}.txt"  # Cambia a tu archivo
archivo_salida = f"{name}.csv"

# Leer el archivo y procesarlo
with open(archivo_entrada, 'r', encoding='utf-8') as entrada:
    lineas = entrada.readlines()

# Separar encabezado
encabezado = lineas[0].strip().split(',')

# Procesar las líneas restantes
datos = []
for linea in lineas[1:]:
    partes = linea.strip().split(',', maxsplit=3)  # Separar hasta la columna "publicación"
    if len(partes) == 4:
        partes[3] = f'"{partes[3]}"'  # Agregar comillas dobles al contenido de la publicación
        datos.append(partes)

# Escribir el archivo corregido
with open(archivo_salida, 'w', encoding='utf-8', newline='') as salida:
    escritor = csv.writer(salida)
    escritor.writerow(encabezado)  # Escribir encabezado
    escritor.writerows(datos)  # Escribir filas procesadas

print(f"Archivo corregido guardado como {archivo_salida}")
