import csv

# Abrir el archivo CSV y procesar los datos
with open("horaHora_diaEleccion.csv", "r", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    hashtags = set()  # Usar un conjunto para almacenar palabras únicas

    # Recorrer cada fila y extraer palabras con #
    for row in reader:
        comentario = row["comentario"]
        palabras = comentario.split()
        hashtags.update([palabra for palabra in palabras if "#" in palabra])

# Escribir los hashtags únicos en aux.txt, separados por un "enter"
with open("aux.txt", "w", encoding="utf-8") as aux_file:
    aux_file.write(",,,,\n".join(sorted(hashtags)))  # Ordenar para mayor legibilidad

print("Se han escrito los hashtags únicos en el archivo aux.txt")
