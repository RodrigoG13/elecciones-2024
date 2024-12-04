import tkinter as tk
from tkinter import messagebox
import csv
import re

# Cargar jerga desde el archivo CSV
def cargar_jerga(archivo):
    jerga = []
    with open(archivo, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            jerga.append({
                'palabra': row['Palabra'].lower(),
                'intensidad': row['Nivel de Intensidad'],
                'sentimiento': row['Sentimiento Asociado'],
                'categoria': row['Categoría'],
                'comentarios': row['Comentarios']
            })
    return jerga

# Función para encontrar palabras sospechosas en un comentario
def encontrar_palabras_sospechosas(comentario, jerga):
    palabras_sospechosas = []
    palabras_comentario = comentario.split()

    # Comparar cada palabra del comentario con las palabras de la jerga
    for palabra in palabras_comentario:
        palabra_limpia = re.sub(r'[^\w\s]', '', palabra).lower()  # Eliminar signos de puntuación y pasar a minúscula
        for item in jerga:
            if palabra_limpia == item['palabra']:
                palabras_sospechosas.append({
                    'palabra_sospechosa': palabra,
                    'jerga': item
                })
    
    return palabras_sospechosas

# Mostrar la ventana de resultados
def mostrar_resultados(comentario, jerga):
    # Crear ventana
    ventana = tk.Tk()
    ventana.title("Análisis de Comentario")

    # Etiqueta para mostrar la frase analizada
    tk.Label(ventana, text="FRASE ANALIZADA:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w")
    tk.Label(ventana, text=comentario, wraplength=500, justify="left").grid(row=0, column=1, pady=10)

    # Buscar palabras sospechosas en el comentario
    palabras_sospechosas = encontrar_palabras_sospechosas(comentario, jerga)

    if palabras_sospechosas:
        for idx, item in enumerate(palabras_sospechosas):
            # Mostrar palabra sospechosa
            tk.Label(ventana, text="PALABRA SOSPECHOSA:", font=("Arial", 10, "bold")).grid(row=2 + idx*4, column=0, sticky="w")
            tk.Label(ventana, text=item['palabra_sospechosa'], wraplength=500, justify="left").grid(row=2 + idx*4, column=1)

            # Mostrar Listbox con jerga relacionada
            tk.Label(ventana, text="JERGA SOSPECHOSA:", font=("Arial", 10, "bold")).grid(row=3 + idx*4, column=0, sticky="w")
            listbox = tk.Listbox(ventana, height=4, width=50)
            listbox.grid(row=3 + idx*4, column=1)

            # Insertar coincidencias de jerga en el Listbox
            listbox.insert(tk.END, f"Palabra: {item['jerga']['palabra']}")
            listbox.insert(tk.END, f"Intensidad: {item['jerga']['intensidad']}")
            listbox.insert(tk.END, f"Sentimiento: {item['jerga']['sentimiento']}")
            listbox.insert(tk.END, f"Categoría: {item['jerga']['categoria']}")
            listbox.insert(tk.END, f"Comentarios: {item['jerga']['comentarios']}")
        
        # Botón para cerrar la ventana
        tk.Button(ventana, text="Cerrar", command=ventana.destroy).grid(row=4 + len(palabras_sospechosas)*4, column=0, columnspan=2)
    else:
        # Si no se encuentran palabras sospechosas
        messagebox.showinfo("Resultado", "No se encontraron palabras sospechosas en el comentario.")
        ventana.destroy()

    # Mostrar ventana
    ventana.mainloop()

# Cargar la jerga
jerga = cargar_jerga('jerga.csv')

# Comentario a analizar
comentario = "Si hace 6 años el debate fue un chiste. El de hoy fue peor, xd a quien irle."

# Mostrar resultados
mostrar_resultados(comentario, jerga)
