import tkinter as tk
from tkinter import messagebox
import pandas as pd
import json
import re
import csv
import os

# Cargar los datasets
def cargar_comentarios():
    if os.path.exists('comentarios_debatesN.csv'):
        comentarios = pd.read_csv('comentarios_debatesN.csv')
    else:
        comentarios = pd.read_csv('comentarios_debates.csv')
        if 'comentario_editado' not in comentarios.columns:
            comentarios['comentario_editado'] = ""
    return comentarios

comentarios_df = cargar_comentarios()
jerga_df = pd.read_csv('jergaN.csv')

# Cargar sugerencias del usuario
try:
    sugerencias_usuario_df = pd.read_csv('sugerencias_usuario.csv')
except FileNotFoundError:
    sugerencias_usuario_df = pd.DataFrame(columns=['Palabra', 'Sugerencia'])

# Diccionario global para almacenar las decisiones del usuario
decisiones_usuario = {}
estado_actual = {"comentario_index": -1}  # Estado inicial (ningún comentario procesado)
palabra_a_traducir = None  # Inicializar la variable global

# Función para guardar sugerencias del usuario
def guardar_sugerencias_usuario():
    sugerencias_usuario_df.to_csv("sugerencias_usuario.csv", index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')

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
        comentario_text.config(state=tk.NORMAL)
        comentario_text.delete(1.0, tk.END)
        comentario_text.insert(tk.END, comentario)
        comentario_text.config(state=tk.DISABLED)
        procesar_palabra()
    else:
        # Notificar al usuario que se han procesado todos los comentarios
        comentario_text.config(state=tk.NORMAL)
        comentario_text.delete(1.0, tk.END)
        comentario_text.insert(tk.END, "Todos los comentarios han sido procesados.")
        comentario_text.config(state=tk.DISABLED)
    guardar_estado_actual()  # Guardar el estado actual después de cambiar el comentario

# Función para mostrar las sugerencias en la interfaz
def mostrar_sugerencias(sugerencias):
    sugerencia_frame.pack(pady=10)
    sugerencia_listbox.delete(0, tk.END)
    for sugerencia in sugerencias:
        sugerencia_listbox.insert(tk.END, sugerencia)
    resultado_palabra.set(f"Sugerencias para '{palabra_a_traducir}':")

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

def procesar_palabra():
    global palabra_a_traducir, resultado_frase
    try:
        # Mientras haya palabras pendientes, procesarlas una por una
        while palabras_pendientes:
            palabra = palabras_pendientes.pop(0)
            palabra_a_traducir = palabra

            decision_anterior = obtener_decision_usuario(palabra)

            if decision_anterior:
                if decision_anterior["estado"] == "aceptada":
                    traduccion_completa.append(decision_anterior["sugerencia"])
                    resaltar_palabra_actual()
                    # Continúa con la siguiente palabra
                    continue
                elif decision_anterior["estado"] == "rechazada":
                    traduccion_completa.append(palabra)
                    resaltar_palabra_actual()
                    # Continúa con la siguiente palabra
                    continue

            sugerencias = obtener_sugerencias(palabra)

            if sugerencias:
                # Mostrar sugerencias y luego esperar a que el usuario elija
                # Aquí dejamos de avanzar en el ciclo, esperando a que el usuario
                # presione aceptar o rechazar. Una vez que el usuario elija,
                # se llamará de nuevo a 'procesar_palabra()' o se continuará de otra forma.
                mostrar_sugerencias(sugerencias)
                return
            else:
                traduccion_completa.append(palabra)
                resaltar_palabra_actual()
                # Continúa con la siguiente palabra

        # Si ya no hay palabras pendientes, se reconstruye el comentario
        comentario_editado = reconstruir_comentario_editado()
        comentarios_df.at[estado_actual["comentario_index"], 'comentario_editado'] = comentario_editado
        comentarios_df.to_csv("comentarios_debatesN.csv", index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
        obtener_comentario_siguiente()

    except Exception as e:
        print(f"Error al procesar la palabra '{palabra_a_traducir}': {e}")
        guardar_decisiones()
        guardar_estado_actual()
        comentarios_df.to_csv("comentarios_debatesN.csv", index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
        raise


# Obtener las sugerencias para una palabra
def obtener_sugerencias(palabra):
    sugerencias = []
    for index, row in jerga_df.iterrows():
        palabra_jerga = row['Palabra'].lower()
        dist = distancia_levenshtein(palabra, palabra_jerga)
        if dist <= 1:
            sugerencias.append(row['Palabra'])  # Asumiendo que la columna 'Sugerencia' existe
    # Añadir sugerencias del usuario
    for index, row in sugerencias_usuario_df.iterrows():
        palabra_usuario = row['Palabra'].lower()
        dist = distancia_levenshtein(palabra, palabra_usuario)
        if dist <= 1:
            sugerencias.append(row['Sugerencia'])
    return sugerencias

# Aceptar una sugerencia
def aceptar_sugerencia():
    sugerencia_seleccionada = sugerencia_listbox.get(tk.ACTIVE)
    if not sugerencia_seleccionada:
        messagebox.showwarning("Advertencia", "No has seleccionado ninguna sugerencia.")
        return  # No hay sugerencia seleccionada
    traduccion_completa.append(sugerencia_seleccionada)

    # Guardar la decisión en el historial
    decisiones_usuario[palabra_a_traducir] = {"estado": "aceptada", "sugerencia": sugerencia_seleccionada}
    guardar_decisiones()  # Guardar inmediatamente después de aceptar

    limpiar_resaltado()
    procesar_palabra()

# Rechazar sugerencia
def rechazar_sugerencia():
    traduccion_completa.append(palabra_a_traducir)

    # Guardar la decisión en el historial
    decisiones_usuario[palabra_a_traducir] = {"estado": "rechazada", "sugerencia": None}
    guardar_decisiones()  # Guardar inmediatamente después de rechazar

    limpiar_resaltado()
    procesar_palabra()

# Función para agregar una sugerencia personalizada
def agregar_sugerencia_personalizada():
    nueva_sugerencia = entrada_sugerencia.get().strip()
    if nueva_sugerencia:
        # Añadir la nueva sugerencia al DataFrame de sugerencias del usuario
        sugerencias_usuario_df.loc[len(sugerencias_usuario_df)] = [palabra_a_traducir, nueva_sugerencia]
        guardar_sugerencias_usuario()
        sugerencia_listbox.insert(tk.END, nueva_sugerencia)
        entrada_sugerencia.delete(0, tk.END)

# Función para resaltar la palabra actual en el comentario
def resaltar_palabra_actual():
    global palabra_a_traducir
    limpiar_resaltado()
    palabra = palabra_a_traducir
    if not palabra:
        return
    comentario_text.config(state=tk.NORMAL)
    contenido = comentario_text.get(1.0, tk.END)
    palabra_lower = palabra.lower()

    inicio = '1.0'
    while True:
        # Buscar palabra completa usando expresiones regulares
        pos = comentario_text.search(r'\b{}\b'.format(re.escape(palabra_lower)), inicio, stopindex=tk.END, nocase=1, regexp=1)
        if not pos:
            break
        # Calcular el final de la palabra
        fin = f"{pos}+{len(palabra)}c"
        # Aplicar la etiqueta
        comentario_text.tag_add("highlight", pos, fin)
        inicio = fin
    comentario_text.config(state=tk.DISABLED)

# Función para limpiar cualquier resaltado previo
def limpiar_resaltado():
    comentario_text.config(state=tk.NORMAL)
    comentario_text.tag_remove("highlight", '1.0', tk.END)
    comentario_text.config(state=tk.DISABLED)

# Configuración inicial de la interfaz gráfica
root = tk.Tk()
root.geometry("900x700")
root.minsize(600, 400)
root.title("Sistema Evolutivo de Reescritura - Jerga Mexicana")

# Estilos y Paletas de Colores
bg_color = "#f0f0f0"
frame_bg = "#ffffff"
button_bg = "#4CAF50"
button_fg = "#ffffff"
highlight_bg = "#ffff99"

root.configure(bg=bg_color)

# Crear las variables de la interfaz
resultado_frase = tk.StringVar()
resultado_palabra = tk.StringVar()

# Crear los frames principales
frame_izquierdo = tk.Frame(root, bg=bg_color)
frame_izquierdo.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH, expand=True)

frame_derecho = tk.Frame(root, bg=bg_color)
frame_derecho.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.Y)

# Sección para mostrar el comentario
tk.Label(frame_izquierdo, text="Comentario para Traducir", font=('Arial', 16, 'bold'), bg=bg_color).pack(anchor='w')

# Crear un Text widget con scrollbar para el comentario
comentario_frame = tk.Frame(frame_izquierdo, bg=frame_bg, bd=2, relief=tk.SUNKEN)
comentario_frame.pack(pady=10, fill=tk.BOTH, expand=True)

comentario_text = tk.Text(comentario_frame, height=20, wrap=tk.WORD, font=('Arial', 14), bg="#fafafa", fg="#333333")
comentario_scrollbar = tk.Scrollbar(comentario_frame, orient="vertical", command=comentario_text.yview)
comentario_text.config(yscrollcommand=comentario_scrollbar.set)

comentario_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
comentario_scrollbar.pack(side=tk.RIGHT, fill="y")

# Configurar la etiqueta de resaltado
comentario_text.tag_configure("highlight", background=highlight_bg)

# Desactivar el Text widget para evitar ediciones
comentario_text.config(state=tk.DISABLED)

# Sección auxiliar para palabras desconocidas
tk.Label(frame_derecho, text="Palabra Desconocida", font=('Arial', 16, 'bold'), bg=bg_color).pack(anchor='w')
resultado_palabra_label = tk.Label(frame_derecho, textvariable=resultado_palabra, font=('Arial', 14), bg=bg_color, fg="#d9534f")
resultado_palabra_label.pack(pady=10, anchor='w')

# Frame para sugerencias
sugerencia_frame = tk.Frame(frame_derecho, bg=frame_bg, bd=2, relief=tk.SUNKEN)
# No se packea aquí, se packa cuando hay sugerencias

sugerencia_label = tk.Label(sugerencia_frame, textvariable=resultado_palabra, font=('Arial', 14, 'bold'), bg=frame_bg)
sugerencia_label.pack(side=tk.TOP, pady=5)

sugerencia_listbox = tk.Listbox(sugerencia_frame, font=('Arial', 14), width=30, height=10, bg="#ffffff", fg="#333333")
sugerencia_listbox.pack(side=tk.TOP, padx=5, pady=5)

entrada_sugerencia = tk.Entry(frame_derecho, font=('Arial', 14), bg="#ffffff", fg="#333333")
entrada_sugerencia.pack(pady=5, fill=tk.X)

agregar_sugerencia_btn = tk.Button(frame_derecho, text="Agregar Sugerencia", command=agregar_sugerencia_personalizada, font=('Arial', 14), bg=button_bg, fg=button_fg, cursor="hand2")
agregar_sugerencia_btn.pack(pady=5, fill=tk.X)

aceptar_btn = tk.Button(sugerencia_frame, text="Aceptar", command=aceptar_sugerencia, font=('Arial', 14), bg="#5cb85c", fg='white', width=10, cursor="hand2")
aceptar_btn.pack(side=tk.LEFT, padx=10, pady=10)

rechazar_btn = tk.Button(sugerencia_frame, text="Rechazar", command=rechazar_sugerencia, font=('Arial', 14), bg="#d9534f", fg='white', width=10, cursor="hand2")
rechazar_btn.pack(side=tk.LEFT, padx=10, pady=10)

# Botón para iniciar la traducción
iniciar_btn = tk.Button(frame_izquierdo, text="Iniciar Traducción", command=obtener_comentario_siguiente, font=('Arial', 16, 'bold'), bg="#0275d8", fg='white', cursor="hand2")
iniciar_btn.pack(pady=20)

# Cerrar la ventana y guardar las decisiones y el estado actual
def on_closing():
    try:
        guardar_decisiones()
        guardar_estado_actual()
        comentarios_df.to_csv("comentarios_debatesN.csv", index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
    except Exception as e:
        messagebox.showerror("Error", f"Error al guardar los datos al cerrar: {e}")
    finally:
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Función para resaltar la palabra actual en el comentario
def resaltar_palabra_actual():
    global palabra_a_traducir
    limpiar_resaltado()
    palabra = palabra_a_traducir
    if not palabra:
        return
    comentario_text.config(state=tk.NORMAL)
    contenido = comentario_text.get(1.0, tk.END)
    palabra_lower = palabra.lower()

    inicio = '1.0'
    while True:
        # Buscar palabra completa usando expresiones regulares
        pos = comentario_text.search(r'\b{}\b'.format(re.escape(palabra_lower)), inicio, stopindex=tk.END, nocase=1, regexp=1)
        if not pos:
            break
        # Calcular el final de la palabra
        fin = f"{pos}+{len(palabra)}c"
        # Aplicar la etiqueta
        comentario_text.tag_add("highlight", pos, fin)
        inicio = fin
    comentario_text.config(state=tk.DISABLED)

# Función para limpiar cualquier resaltado previo
def limpiar_resaltado():
    comentario_text.config(state=tk.NORMAL)
    comentario_text.tag_remove("highlight", '1.0', tk.END)
    comentario_text.config(state=tk.DISABLED)

# Iniciar la interfaz gráfica
root.mainloop()
