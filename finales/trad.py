import tkinter as tk
import pandas as pd
import json
import re
import string
import csv  # Importar el módulo csv para manejar el quoting en el CSV

# Cargar los datasets
comentarios_df = pd.read_csv('comentarios_debates.csv')
jerga_df = pd.read_csv('jergaN.csv')

# Inicializar la columna para comentarios editados si no existe
if 'comentario_editado' not in comentarios_df.columns:
    comentarios_df['comentario_editado'] = ""

# Diccionario global para almacenar las decisiones del usuario
decisiones_usuario = {}
estado_actual = {"comentario_index": -1}  # Estado inicial (ningún comentario procesado)

# Función que calcula la distancia de Levenshtein entre dos cadenas
def distancia_levenshtein(s1, s2):
    if len(s1) < len(s2):
        return distancia_levenshtein(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

# Función para cargar el estado actual desde un archivo JSON
def cargar_estado_actual():
    global estado_actual
    try:
        with open("estado_actual.json", "r", encoding='utf-8') as f:
            estado_actual = json.load(f)
    except FileNotFoundError:
        estado_actual = {"comentario_index": -1}

# Función para guardar el estado actual en un archivo JSON
def guardar_estado_actual():
    with open("estado_actual.json", "w", encoding='utf-8') as f:
        json.dump(estado_actual, f, ensure_ascii=False)

# Cargar el estado actual al iniciar el programa
cargar_estado_actual()

# Función para cargar el historial de decisiones desde un archivo JSON
def cargar_decisiones():
    global decisiones_usuario
    try:
        with open("historial_decisiones.json", "r", encoding='utf-8') as f:
            decisiones_usuario = json.load(f)
    except FileNotFoundError:
        decisiones_usuario = {}

# Función para guardar el historial de decisiones en un archivo JSON
def guardar_decisiones():
    with open("historial_decisiones.json", "w", encoding='utf-8') as f:
        json.dump(decisiones_usuario, f, ensure_ascii=False)

# Cargar las decisiones al iniciar el programa
cargar_decisiones()

# Función para obtener el historial de decisiones de una palabra
def obtener_decision_usuario(palabra):
    if palabra in decisiones_usuario:
        return decisiones_usuario[palabra]
    return None

# Obtener el siguiente comentario
def obtener_comentario_siguiente():
    global estado_actual, palabras_pendientes, traduccion_completa, comentario_original_palabras
    estado_actual["comentario_index"] += 1
    if estado_actual["comentario_index"] < len(comentarios_df):
        comentario = comentarios_df['comentario'].dropna().iloc[estado_actual["comentario_index"]]
        # Guardar una versión de las palabras originales (incluyendo puntuación) para reconstruir el comentario editado
        comentario_original_palabras = re.findall(r'\b\w+\b|\S', comentario)
        # Extraer palabras sin puntuación y en minúsculas para procesar
        palabras_pendientes = re.findall(r'\b\w+\b', comentario.lower())
        traduccion_completa = []
        comentario_text.delete(1.0, tk.END)
        comentario_text.insert(tk.END, comentario)
        procesar_palabra()
    else:
        # Notificar al usuario que se han procesado todos los comentarios
        comentario_text.delete(1.0, tk.END)
        comentario_text.insert(tk.END, "Todos los comentarios han sido procesados.")
    guardar_estado_actual()  # Guardar el estado actual después de cambiar el comentario

# Función para procesar cada palabra
def procesar_palabra():
    global palabra_a_traducir, resultado_frase
    if not palabras_pendientes:
        # Reconstruir el comentario editado preservando la puntuación original
        comentario_editado = reconstruir_comentario_editado()
        comentarios_df.at[estado_actual["comentario_index"], 'comentario_editado'] = comentario_editado
        # Guarda el DataFrame actualizado en un nuevo archivo CSV con el manejo adecuado de comillas
        comentarios_df.to_csv("comentarios_debatesN.csv", index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
        obtener_comentario_siguiente()
        return

    palabra = palabras_pendientes.pop(0)
    palabra_a_traducir = palabra

    decision_anterior = obtener_decision_usuario(palabra)

    if decision_anterior:
        if decision_anterior["estado"] == "aceptada":
            traduccion_completa.append(decision_anterior["sugerencia"])
            procesar_palabra()
            return
        elif decision_anterior["estado"] == "rechazada":
            traduccion_completa.append(palabra)
            procesar_palabra()
            return

    sugerencias = obtener_sugerencias(palabra)

    if sugerencias:
        mostrar_sugerencias(sugerencias)
    else:
        traduccion_completa.append(palabra)
        procesar_palabra()

# Obtener las sugerencias para una palabra
def obtener_sugerencias(palabra):
    sugerencias = []
    for index, row in jerga_df.iterrows():
        palabra_jerga = row['Palabra'].lower()
        dist = distancia_levenshtein(palabra, palabra_jerga)
        if dist <= 1:
            sugerencias.append(palabra_jerga)
    return sugerencias

# Función para mostrar las sugerencias en la interfaz
def mostrar_sugerencias(sugerencias):
    sugerencia_frame.pack(pady=10)
    sugerencia_listbox.delete(0, tk.END)
    for palabra in sugerencias:
        sugerencia_listbox.insert(tk.END, palabra)
    resultado_palabra.set(f"Sugerencias para '{palabra_a_traducir}':")

# Aceptar una sugerencia
def aceptar_sugerencia():
    sugerencia_seleccionada = sugerencia_listbox.get(tk.ACTIVE)
    traduccion_completa.append(sugerencia_seleccionada)

    # Guardar la decisión en el historial
    decisiones_usuario[palabra_a_traducir] = {"estado": "aceptada", "sugerencia": sugerencia_seleccionada}
    guardar_decisiones()  # Guardar inmediatamente después de aceptar

    procesar_palabra()

# Rechazar sugerencia
def rechazar_sugerencia():
    traduccion_completa.append(palabra_a_traducir)

    # Guardar la decisión en el historial
    decisiones_usuario[palabra_a_traducir] = {"estado": "rechazada", "sugerencia": None}
    guardar_decisiones()  # Guardar inmediatamente después de rechazar

    procesar_palabra()

# Función para reconstruir el comentario editado preservando la puntuación original
def reconstruir_comentario_editado():
    resultado = []
    index_traduccion = 0
    for token in comentario_original_palabras:
        if re.match(r'\b\w+\b', token):
            if index_traduccion < len(traduccion_completa):
                resultado.append(traduccion_completa[index_traduccion])
                index_traduccion += 1
            else:
                resultado.append(token)
        else:
            resultado.append(token)
    # Unir los tokens considerando espacios y puntuación
    comentario_editado = ''
    for i, token in enumerate(resultado):
        if i > 0 and re.match(r'\w', token) and re.match(r'\w', resultado[i-1]):
            comentario_editado += ' ' + token
        else:
            comentario_editado += token
    return comentario_editado

# Crear la ventana raíz de la interfaz gráfica
root = tk.Tk()
root.geometry("800x600")
root.minsize(400, 300)
root.title("Sistema Evolutivo de Reescritura - Jerga Mexicana")

# Crear las variables de la interfaz
resultado_frase = tk.StringVar()
resultado_palabra = tk.StringVar()

frame_izquierdo = tk.Frame(root)
frame_izquierdo.pack(side=tk.LEFT, padx=10, pady=10)

frame_derecho = tk.Frame(root)
frame_derecho.pack(side=tk.RIGHT, padx=10, pady=10)

# Sección para mostrar el primer comentario
tk.Label(frame_izquierdo, text="Comentario para Traducir", font=('Arial', 14)).pack()

# Crear un Text widget con scrollbar para el comentario
comentario_frame = tk.Frame(frame_izquierdo)
comentario_text = tk.Text(comentario_frame, height=10, width=50, wrap=tk.WORD, font=('Arial', 14))
comentario_scrollbar = tk.Scrollbar(comentario_frame, orient="vertical", command=comentario_text.yview)
comentario_text.config(yscrollcommand=comentario_scrollbar.set)

comentario_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
comentario_scrollbar.pack(side=tk.RIGHT, fill="y")

comentario_frame.pack(pady=10)

# Sección auxiliar para palabras desconocidas
tk.Label(frame_derecho, text="Palabra Desconocida", font=('Arial', 14)).pack()
resultado_palabra_label = tk.Label(frame_derecho, textvariable=resultado_palabra, font=('Arial', 14))
resultado_palabra_label.pack(pady=10)

# Frame para sugerencias
sugerencia_frame = tk.Frame(frame_derecho)
sugerencia_label = tk.Label(sugerencia_frame, textvariable=resultado_palabra, font=('Arial', 14))
sugerencia_label.pack(side=tk.TOP, pady=5)

sugerencia_listbox = tk.Listbox(sugerencia_frame, font=('Arial', 14))
sugerencia_listbox.pack(side=tk.TOP, padx=5, pady=5)

aceptar_btn = tk.Button(sugerencia_frame, text="Aceptar", command=aceptar_sugerencia, font=('Arial', 14))
aceptar_btn.pack(side=tk.LEFT, padx=5, pady=5)

rechazar_btn = tk.Button(sugerencia_frame, text="Rechazar", command=rechazar_sugerencia, font=('Arial', 14))
rechazar_btn.pack(side=tk.LEFT, padx=5, pady=5)

# Botón para iniciar la traducción
iniciar_btn = tk.Button(frame_izquierdo, text="Iniciar Revisión de Jerga", command=obtener_comentario_siguiente, font=('Arial', 14))
iniciar_btn.pack(pady=10)

# Cerrar la ventana y guardar las decisiones y el estado actual
def on_closing():
    guardar_decisiones()
    guardar_estado_actual()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Iniciar la interfaz gráfica
root.mainloop()
