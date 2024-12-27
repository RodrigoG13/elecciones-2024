import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import os

# Paleta de colores inspirada en Bootstrap
color_fondo = "#f8f9fa"      # color de fondo principal
color_primario = "#007bff"   # azul principal
color_texto = "#343a40"      # texto oscuro
color_borde = "#ced4da"      # gris suave para bordes
color_candidato_bg = "#ffffff"
color_marco_opciones = "#e9ecef"

# Colores para las opciones
color_negativa = "#dc3545"  # rojo
color_neutral = "#6c757d"   # gris neutro
color_positiva = "#28a745"  # verde

# Fuentes
fuente_titulo = ("Helvetica", 16, "bold")
fuente_subtitulo = ("Helvetica", 12, "bold")
fuente_texto = ("Helvetica", 12)
fuente_candidato = ("Helvetica", 14, "bold")
fuente_comentario = ("Helvetica", 12, "italic")
fuente_opcion = ("Helvetica", 12)

# lista de candidatos
CANDIDATOS = ["Xóchitl", "Claudia", "Maynez", "Ninguno"]

# Primero verifica si existe un dataset etiquetado previo
if os.path.exists("dataset_etiquetado3.csv"):
    df = pd.read_csv("dataset_etiquetado3.csv", sep=",", quotechar='"', encoding="utf-8", dtype=str)
else:
    # Carga dataset original
    df = pd.read_csv("horaHora_diaEleccionN.csv", sep=",", quotechar='"', encoding="utf-8", dtype=str)

# verificar existencia de columna comentario_editado
if "comentario_editado" not in df.columns:
    raise ValueError("La columna 'comentario_editado' no existe en el dataset.")

# crear columnas para etiquetas si no existen
for c in CANDIDATOS:
    if c not in df.columns:
        df[c] = None

# índice del comentario actual
if os.path.exists("progreso.txt"):
    with open("progreso.txt", "r") as file:
        idx = int(file.read().strip())
else:
    idx = 0

total = len(df)

def guardar_progreso():
    global idx
    with open("progreso.txt", "w") as file:
        file.write(str(idx))

def mostrar_comentario():
    comentario = df.loc[idx, "comentario_editado"]
    label_indicador.config(text=f"comentario {idx+1} de {total}")
    label_comentario.config(text=comentario)

    # Ajustar los radiobuttons según el valor guardado
    for cand in CANDIDATOS:
        valor = df.loc[idx, cand]
        if valor is None or pd.isna(valor):
            valor = "neutral"  # valor por defecto
        # si los valores no están normalizados, normalizar
        if str(valor) in ["-1", "-1.0"]:
            valor = "negativa"
        elif str(valor) in ["1", "1.0"]:
            valor = "positiva"
        elif str(valor) in ["0", "0.0"]:
            valor = "neutral"
        radio_vars[cand].set(valor)

def guardar_etiqueta_y_siguiente():
    global idx
    # guardar etiquetas usando mapping
    # -1 = negativa, 0 = neutral, 1 = positiva
    mapping = {"negativa": -1, "neutral": 0, "positiva": 1}
    for cand in CANDIDATOS:
        df.loc[idx, cand] = mapping[radio_vars[cand].get()]

    df.to_csv("dataset_etiquetado3.csv", index=False, quoting=1)
    guardar_progreso()

    label_feedback.config(text="etiqueta guardada. pasando al siguiente...", fg="#4a4a4a")
    root.after(800, limpiar_feedback)

    if idx < total - 1:
        idx += 1
        mostrar_comentario()
    else:
        messagebox.showinfo("fin", "has etiquetado todos los comentarios. presiona 'finalizar y guardar' para salir.")

def limpiar_feedback():
    label_feedback.config(text="")

def finalizar_y_guardar():
    df.to_csv("dataset_etiquetado3.csv", index=False, quoting=1)
    guardar_progreso()
    messagebox.showinfo("guardado", "se ha guardado tu etiquetado con éxito.")
    root.destroy()

root = tk.Tk()
root.title("Herramienta de etiquetado de comentarios")
root.configure(bg=color_fondo)

# Barra superior tipo "navbar"
frame_topbar = tk.Frame(root, bg=color_primario, padx=15, pady=10)
frame_topbar.pack(side="top", fill="x")

label_titulo = tk.Label(
    frame_topbar,
    text="Herramienta de etiquetado de comentarios",
    bg=color_primario,
    fg="#ffffff",
    font=fuente_titulo
)
label_titulo.pack(anchor="w")

# Instrucciones
frame_instrucciones = tk.Frame(root, bg=color_fondo, padx=10, pady=10)
frame_instrucciones.pack(fill="x")

label_instrucciones = tk.Label(
    frame_instrucciones,
    text="por favor, selecciona el tipo de opinión para cada candidato:\n\n"
         "❌ negativa: el comentario va en contra del candidato.\n"
         "➖ neutral: no expresa una postura clara.\n"
         "✅ positiva: apoya o favorece al candidato.",
    bg=color_fondo, fg=color_texto, font=fuente_texto, justify="left"
)
label_instrucciones.pack(anchor="w")

# Indicador de comentario actual
frame_indicador = tk.Frame(root, bg=color_fondo, padx=10, pady=5)
frame_indicador.pack(fill="x")

label_indicador = tk.Label(
    frame_indicador, text="", bg=color_fondo, fg=color_texto, font=fuente_subtitulo
)
label_indicador.pack(anchor="w")

# Marco para el comentario
frame_comentario = tk.Frame(root, bg="#ffffff", bd=2, relief="groove", padx=15, pady=15)
frame_comentario.pack(padx=10, pady=10, fill="both", expand=True)

label_comentario = tk.Label(
    frame_comentario, text="", wraplength=600, justify="left",
    bg="#ffffff", fg=color_texto, font=fuente_comentario
)
label_comentario.pack(pady=5)

# Marco para candidatos
frame_candidatos = tk.Frame(root, bg=color_fondo)
frame_candidatos.pack(padx=10, pady=10, fill="both", expand=True)

radio_vars = {}

# Opciones con sus colores y descripciones
opciones = [
    ("negativa", "❌ negativa", color_negativa),
    ("neutral", "➖ neutral", color_neutral),
    ("positiva", "✅ positiva", color_positiva)
]

# Crear estilos de sección candidato
for i, cand in enumerate(CANDIDATOS):
    lf = tk.LabelFrame(
        frame_candidatos,
        text=cand,
        bg=color_marco_opciones,
        fg=color_texto,
        font=fuente_candidato,
        padx=10,
        pady=10,
        bd=2,
        relief="groove"
    )
    lf.grid(row=0, column=i, padx=15, pady=10, sticky="nsew")

    var = tk.StringVar(value="neutral")
    radio_vars[cand] = var

    for val, val_str, color_icono in opciones:
        f_opcion = tk.Frame(lf, bg=color_marco_opciones)
        f_opcion.pack(anchor="w", pady=5)

        rb = tk.Radiobutton(
            f_opcion,
            text=val_str,
            variable=var,
            value=val,
            bg=color_marco_opciones,
            fg=color_icono,
            font=fuente_opcion,
            selectcolor="#ffffff",
            activebackground="#ffffff",
            anchor="w",
            padx=5,
            pady=2
        )
        rb.pack(side="left", anchor="w")

# Marco de acciones
frame_acciones = tk.Frame(root, bg=color_fondo, padx=10, pady=10)
frame_acciones.pack(fill="x")

btn_siguiente = tk.Button(
    frame_acciones,
    text="guardar etiqueta y siguiente",
    command=guardar_etiqueta_y_siguiente,
    bg=color_primario,
    fg="#ffffff",
    font=fuente_texto,
    padx=15,
    pady=8,
    relief="raised",
    bd=2,
    activebackground="#0056b3"
)
btn_siguiente.pack(side="left", padx=10)

btn_finalizar = tk.Button(
    frame_acciones,
    text="finalizar y guardar",
    command=finalizar_y_guardar,
    bg=color_primario,
    fg="#ffffff",
    font=fuente_texto,
    padx=15,
    pady=8,
    relief="raised",
    bd=2,
    activebackground="#0056b3"
)
btn_finalizar.pack(side="left", padx=10)

label_feedback = tk.Label(
    frame_acciones, text="", bg=color_fondo, fg="green", font=("Helvetica", 10, "italic")
)
label_feedback.pack(side="left", padx=10)

# Mostrar el primer comentario
mostrar_comentario()

root.mainloop()
