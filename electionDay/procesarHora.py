import pandas as pd
import re
import sys
import random
from datetime import datetime, timedelta

def ajustar_hora_publicacion(row, hora_fin, hora_inicio):
    hora_publicacion = row['hora de publicación']
    
    # Asegurarse de que hora_publicacion es una cadena
    if pd.isnull(hora_publicacion):
        hora_publicacion = ''
    else:
        hora_publicacion = str(hora_publicacion).lower().strip()
    
    # Expresión regular para encontrar minutos y segundos
    match = re.match(r'(\d+)\s*(m|min|s)$', hora_publicacion)
    if match:
        cantidad = int(match.group(1))
        unidad = match.group(2)
        if unidad in ['m', 'min']:
            delta = timedelta(minutes=cantidad)
        elif unidad == 's':
            delta = timedelta(seconds=cantidad)
        nueva_hora = hora_fin - delta
    else:
        # Asignar hora aleatoria
        delta_total = (hora_fin - hora_inicio).total_seconds()
        if delta_total > 0:
            random_seconds = random.randint(0, int(delta_total))
            nueva_hora = hora_inicio + timedelta(seconds=random_seconds)
        else:
            # Si delta_total es 0 o negativo, retornar la hora de fin
            nueva_hora = hora_fin
    
    # Devolver solo la hora en formato HH:MM
    return nueva_hora.strftime('%H:%M')

def main(nombre):
    nombre_archivo_csv = f"{nombre}.csv"
    # Extraer horas del nombre del archivo
    match = re.search(r'(\d{1,2})a(\d{1,2})', nombre_archivo_csv)
    if not match:
        print("El nombre del archivo debe contener el intervalo de horas en formato 'XXaYY'")
        sys.exit(1)
    hora_inicio_str, hora_fin_str = match.groups()
    hora_inicio_int = int(hora_inicio_str)
    hora_fin_int = int(hora_fin_str)
    
    # Manejar casos donde hora_fin_int es menor que hora_inicio_int (cruce de medianoche)
    if hora_fin_int < hora_inicio_int:
        # Asumimos que hora_fin es al día siguiente
        fecha_inicio = datetime.now().date()
        fecha_fin = fecha_inicio + timedelta(days=1)
    else:
        fecha_inicio = fecha_fin = datetime.now().date()
    
    hora_inicio = datetime.combine(fecha_inicio, datetime.min.time()) + timedelta(hours=hora_inicio_int)
    hora_fin = datetime.combine(fecha_fin, datetime.min.time()) + timedelta(hours=hora_fin_int)
    
    # Leer el archivo CSV
    df = pd.read_csv(nombre_archivo_csv)
    
    # Añadir columnas de hora_inicio y hora_fin
    df['hora_inicio'] = hora_inicio
    df['hora_fin'] = hora_fin
    
    # Aplicar la función de ajuste
    df['hora de publicación'] = df.apply(ajustar_hora_publicacion, axis=1, hora_fin=hora_fin, hora_inicio=hora_inicio)
    
    # Eliminar columnas no necesarias
    df = df.drop(columns=['hora_inicio'])
    df = df.drop(columns=['hora_fin'])
    
    # Si deseas eliminar la columna original de 'hora de publicación ajustada', descomenta la siguiente línea
    # df = df.drop(columns=['hora de publicación original'])
    
    # Guardar el resultado en un nuevo archivo CSV
    df.to_csv(f'ajustado_{nombre}.csv', index=False)
    print(f"El archivo ajustado ha sido guardado como 'ajustado_{nombre}.csv'.")

if __name__ == '__main__':
    nombre = '22a23'  # Reemplaza con el nombre de tu archivo
    main(nombre)
