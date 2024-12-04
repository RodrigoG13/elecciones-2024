import pandas as pd
import Levenshtein
from difflib import get_close_matches
import os
import re

def cargar_dataset():
    print("Seleccione el dataset que desea normalizar:")
    print("1. comentarios_debates.csv")
    print("2. horaHora_diaEleccion.csv")
    opcion = input("Ingrese el número de su elección: ")
    
    if opcion == '1':
        nombre_dataset = 'comentarios_debates.csv'
        df = pd.read_csv(nombre_dataset, encoding='utf-8')
    elif opcion == '2':
        nombre_dataset = 'horaHora_diaEleccion.csv'
        df = pd.read_csv(nombre_dataset, encoding='utf-8', quotechar='"', skipinitialspace=True)
    else:
        print("Opción no válida. Inténtelo de nuevo.")
        return cargar_dataset()
    
    return df, nombre_dataset


def cargar_bolsa_palabras():
    bolsa_palabras = pd.read_csv('jergaN.csv')
    palabras_bolsa = bolsa_palabras['Palabra'].tolist()
    print(palabras_bolsa)
    return palabras_bolsa


def cargar_variantes():
    if os.path.exists('variantes.csv'):
        variantes = pd.read_csv('variantes.csv')
    else:
        # Crear un DataFrame vacío con las columnas necesarias
        variantes = pd.DataFrame(columns=['palabra', 'palabras_si_reemplazar', 'palabras_no_reemplazar'])
    return variantes


import pandas as pd
import Levenshtein
import os
import re

def analizar_comentarios(df, palabras_bolsa, variantes):
    # Convertir variantes a diccionarios para acceso rápido
    variantes_si = variantes.dropna(subset=['palabras_si_reemplazar'])
    dict_si = dict(zip(variantes_si['palabras_si_reemplazar'].str.lower(), variantes_si['palabra']))

    variantes_no = variantes.dropna(subset=['palabras_no_reemplazar'])
    set_no = set(variantes_no['palabras_no_reemplazar'].str.lower())

    # Iterar sobre cada comentario
    for idx, row in df.iterrows():
        comentario = str(row['comentario'])
        # Tokenizar el comentario, separando palabras y puntuación
        tokens = re.findall(r'\w+|[^\w\s]', comentario, re.UNICODE)
        comentario_modificado = []

        for token in tokens:
            # Si es puntuación, se agrega tal cual
            if re.match(r'[^\w]', token):
                comentario_modificado.append(token)
                continue

            palabra_original = token
            palabra_lower = palabra_original.lower()
            reemplazado = False

            # Verificar si la palabra ya ha sido evaluada
            if palabra_lower in dict_si:
                # Reemplazar con la palabra sugerida
                comentario_modificado.append(dict_si[palabra_lower])
                reemplazado = True
                continue
            elif palabra_lower in set_no:
                # No reemplazar
                comentario_modificado.append(palabra_original)
                reemplazado = True
                continue

            # Buscar coincidencias con distancia de Levenshtein
            posibles_matches = []
            distancia_min = None
            for palabra_bolsa in palabras_bolsa:
                distancia = Levenshtein.distance(palabra_lower, palabra_bolsa.lower())
                if distancia <= 2:
                    if distancia_min is None or distancia < distancia_min:
                        distancia_min = distancia
                        posibles_matches = [palabra_bolsa]
                    elif distancia == distancia_min:
                        posibles_matches.append(palabra_bolsa)
            if posibles_matches:
                # Preguntar al usuario si desea reemplazar
                print(f"\nPalabra encontrada: '{palabra_original}'")
                print("Posibles reemplazos:")
                for i, match in enumerate(posibles_matches):
                    print(f"{i+1}. {match}")
                print("0. No reemplazar")
                opcion = input("Seleccione una opción: ")
                try:
                    opcion = int(opcion)
                    if opcion > 0 and opcion <= len(posibles_matches):
                        palabra_sugerida = posibles_matches[opcion - 1]
                        comentario_modificado.append(palabra_sugerida)
                        # Registrar en variantes
                        nuevas_filas = pd.DataFrame({
                            'palabra': [palabra_sugerida],
                            'palabras_si_reemplazar': [palabra_original],
                            'palabras_no_reemplazar': [None]
                        })
                        variantes = pd.concat([variantes, nuevas_filas], ignore_index=True)
                        # Actualizar diccionario
                        dict_si[palabra_lower] = palabra_sugerida
                    else:
                        comentario_modificado.append(palabra_original)
                        # Registrar en variantes como no reemplazar
                        nuevas_filas = pd.DataFrame({
                            'palabra': [palabra_original],
                            'palabras_si_reemplazar': [None],
                            'palabras_no_reemplazar': [palabra_original]
                        })
                        variantes = pd.concat([variantes, nuevas_filas], ignore_index=True)
                        # Actualizar conjunto
                        set_no.add(palabra_lower)
                except ValueError:
                    comentario_modificado.append(palabra_original)
            else:
                comentario_modificado.append(palabra_original)

        # Reconstruir el comentario
        comentario_modificado = ''.join([' ' + token if not re.match(r'^\W$', token) and i > 0 else token for i, token in enumerate(comentario_modificado)]).strip()
        df.at[idx, 'comentario'] = comentario_modificado

    return df, variantes




def guardar_resultados(df, nombre_dataset, variantes):
    # Guardar el dataset modificado
    df.to_csv(f'{nombre_dataset}_normalizado.csv', index=False)
    # Guardar el archivo de variantes
    variantes.to_csv('variantes.csv', index=False)


def main():
    # Paso 1: Cargar el dataset seleccionado
    df, nombre_dataset = cargar_dataset()
    
    # Paso 2: Cargar la bolsa de palabras
    palabras_bolsa = cargar_bolsa_palabras()
    
    # Paso 3: Cargar o crear el archivo de variantes
    variantes = cargar_variantes()
    
    # Paso 4: Analizar comentarios y reemplazar palabras
    df, variantes_actualizadas = analizar_comentarios(df, palabras_bolsa, variantes)
    
    # Paso 5: Guardar los resultados
    guardar_resultados(df, nombre_dataset, variantes_actualizadas)
    
    print("Proceso completado. Los archivos actualizados han sido guardados.")

# Ejecutar la función principal
if __name__ == '__main__':
    main()
