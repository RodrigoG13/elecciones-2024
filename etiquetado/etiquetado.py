import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np

# lista de candidatos
CANDIDATOS = ["Xóchitl", "Claudia", "Maynes", "Ninguno"]

# carga del dataset original
df = pd.read_csv("comentarios_debatesNP.csv", sep=",", quotechar='"', encoding="utf-8", dtype=str)

# verificar existencia de columna comentario_editado
if "comentario_editado" not in df.columns:
    raise ValueError("La columna 'comentario_editado' no existe en el dataset.")

# crear columnas para etiquetas si no existen
for c in CANDIDATOS:
    if c not in df.columns:
        df[c] = None

# índice del comentario actual
idx = 0
total = len(df)

def mostrar_comentario():
    comentario = df.loc[idx, "comentario_editado"]
    label_indicador.config(text=f"Estás etiquetando el comentario {idx+1} de {total}")
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

    # mensaje de retroalimentación inmediata
    label_feedback.config(text="Etiqueta guardada. Pasando al siguiente comentario...", fg="#333333")
    root.after(800, limpiar_feedback)

    # siguiente comentario
    if idx < total - 1:
        idx += 1
        mostrar_comentario()
    else:
        messagebox.showinfo("Fin", "Has llegado al último comentario. Si consideras que todo está correcto, presiona 'Finalizar y Guardar'.")

def limpiar_feedback():
    label_feedback.config(text="")

def finalizar_y_guardar():
    df.to_csv("dataset_etiquetado.csv", index=False, quoting=1)
    messagebox.showinfo("Guardado", "Se ha guardado tu etiquetado con éxito.")
    root.destroy()

# ventana principal
root = tk.Tk()
root.title("Herramienta de etiquetado de comentarios")

# colores
color_fondo = "#f7f7f7"
color_marco = "#e6e6e6"
color_botones = "#b3c6ff"
color_texto = "#333333"

root.configure(bg=color_fondo)

# fuentes
fuente_titulo = ("Arial", 14, "bold")
fuente_texto = ("Arial", 12)
fuente_candidato = ("Arial", 12, "bold")

# Instrucciones al inicio
frame_instrucciones = tk.Frame(root, bg=color_fondo)
frame_instrucciones.pack(padx=10, pady=10, fill="x")

label_instrucciones = tk.Label(
    frame_instrucciones,
    text="Por favor selecciona el tipo de opinión que expresa el comentario para cada candidato.\n"
         "Opciones:\n"
         "• Opinión negativa (❌): El comentario va en contra del candidato.\n"
         "• Neutral (➖): El comentario no expresa una postura clara.\n"
         "• Opinión positiva (✅): El comentario apoya o favorece al candidato.",
    bg=color_fondo, fg=color_texto, font=fuente_texto, justify="left"
)
label_instrucciones.pack(anchor="w")

# indicador de comentario actual
frame_indicador = tk.Frame(root, bg=color_fondo)
frame_indicador.pack(padx=10, pady=(0,10), fill="x")

label_indicador = tk.Label(frame_indicador, text="", bg=color_fondo, fg=color_texto, font=fuente_texto)
label_indicador.pack(anchor="w")

# marco para comentario (destacado visualmente)
frame_comentario = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
frame_comentario.pack(padx=10, pady=10, fill="both", expand=True)

label_comentario = tk.Label(frame_comentario, text="", wraplength=600, justify="left",
                            bg="#ffffff", fg=color_texto, font=("Arial", 12, "italic"), padx=10, pady=10)
label_comentario.pack(pady=10)

# marco para candidatos
frame_candidatos = tk.Frame(root, bg=color_fondo)
frame_candidatos.pack(padx=10, pady=10, fill="both", expand=True)

radio_vars = {}
opciones = [
    ("negativa", "❌ negativa"),
    ("neutral", "➖ neutral"),
    ("positiva", "✅ positiva")
]

for i, cand in enumerate(CANDIDATOS):
    f = tk.Frame(frame_candidatos, bg=color_marco, borderwidth=2, relief="groove", padx=5, pady=5)
    f.grid(row=0, column=i, padx=10, pady=10, sticky="nsew")
    tk.Label(f, text=cand, bg=color_marco, fg=color_texto, font=fuente_candidato).pack(pady=5)
    var = tk.StringVar()
    var.set("neutral") # valor inicial neutro
    radio_vars[cand] = var
    for val, val_str in opciones:
        tk.Radiobutton(
            f, text=val_str, variable=var, value=val, bg=color_marco, fg=color_texto, font=fuente_texto,
            selectcolor="#ffffff", activebackground="#ffffff", wraplength=120, justify="left", anchor="w"
        ).pack(anchor="w")

# marco para botones de navegación y feedback
frame_acciones = tk.Frame(root, bg=color_fondo)
frame_acciones.pack(padx=10, pady=10, fill="x")

btn_siguiente = tk.Button(frame_acciones, text="Guardar etiqueta y siguiente",
                          command=guardar_etiqueta_y_siguiente, bg=color_botones, fg=color_texto,
                          font=fuente_texto, padx=10, pady=5, relief="raised")
btn_siguiente.pack(side="left", padx=10)

btn_finalizar = tk.Button(frame_acciones, text="Finalizar y Guardar",
                          command=finalizar_y_guardar, bg=color_botones, fg=color_texto,
                          font=fuente_texto, padx=10, pady=5, relief="raised")
btn_finalizar.pack(side="left", padx=10)

label_feedback = tk.Label(frame_acciones, text="", bg=color_fondo, fg="green", font=("Arial", 10, "italic"))
label_feedback.pack(side="left", padx=10)

# mostrar el primer comentario
mostrar_comentario()

root.mainloop()
