import csv
import re

def parse_comments_from_file(input_file, output_file):
    # Extraer num_debate y canal del nombre del archivo
    num_debate, canal = input_file.split('_')
    
    # Asignar fecha basada en num_debate
    debate_dates = {
        '1': '07/04/2024',
        '2': '28/04/2024',
        '3': '19/05/2024'
    }
    debate_date = debate_dates.get(num_debate, 'Fecha desconocida')  # Usar una fecha genérica si no coincide ningún número
    
    # Leer el contenido del archivo de entrada
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    data = []
    # Mejorado para separar cada entrada por los usuarios y capturar los likes de manera más confiable
    entries = re.split(r'(?=\n@)', content)
    for entry in entries:
        if entry.strip():
            parts = entry.strip().split('\n')
            if len(parts) >= 3:
                username = parts[0].strip('@')
                # Utilizar la fecha asignada basada en num_debate
                fecha = debate_date
                # Verificar si el último elemento es numérico (likes)
                if parts[-1].isdigit():
                    num_likes = parts[-1]
                    comentario = ' '.join(parts[2:-1]).replace('\n', ' ').strip()
                else:
                    num_likes = "0"
                    comentario = ' '.join(parts[2:]).replace('\n', ' ').strip()
                data.append([num_debate, canal, username, fecha, comentario, num_likes])
    
    # Escribir los datos procesados en un archivo CSV
    with open(output_file + '.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['num_debate', 'canal', 'username', 'fecha', 'comentario', 'num_likes'])
        for row in data:
            writer.writerow(row)

# Uso de la función
file_path = "3_unotv"
parse_comments_from_file(file_path, file_path)
