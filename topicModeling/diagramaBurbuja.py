import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns

# Configuración de estilo de seaborn
sns.set(style="whitegrid", palette="muted", color_codes=True)

# Cargar los datos
data = pd.read_csv("selected_topics.csv")

# Convertir la columna 'fecha' a formato de fecha
data['fecha'] = pd.to_datetime(data['fecha'])

# Crear los rangos para las fechas en el eje X
fechas_importantes = [
    "2024-04-07", "2024-04-28", "2024-05-19", "2024-06-02"
]
fechas_importantes = pd.to_datetime(fechas_importantes)

# Asignar valores X aleatorios dentro del rango de cada fecha
np.random.seed(42)  # Para consistencia
data['fecha_random'] = data['fecha'] + pd.to_timedelta(
    np.random.uniform(-3, 3, size=len(data)), unit='d'
)

# Normalizar el tamaño de las burbujas para mejorar la visualización
bubble_size = (data['Count'] / data['Count'].max()) * 3000  # Ajustado para mejor escala

# Crear el diagrama de burbuja
fig, ax = plt.subplots(figsize=(16, 9))

# Seleccionar una paleta de colores más sofisticada
unique_topics = data['Topic'].unique()
palette = sns.color_palette("hsv", len(unique_topics))
color_mapping = dict(zip(unique_topics, palette))
colors = data['Topic'].map(color_mapping)

scatter = ax.scatter(
    data['fecha_random'],
    data['silhouette_promedio'],
    s=bubble_size,  # Tamaño de las burbujas basado en 'Count'
    c=colors,        # Color basado en 'Topic'
    alpha=0.6,
    edgecolors="gray",
    linewidth=0.5
)

# Configuración del eje X (fechas importantes)
ax.set_xticks(fechas_importantes)
ax.set_xticklabels(fechas_importantes.strftime('%Y-%m-%d'), rotation=45, fontsize=10)

# Configuración de los ejes y título
ax.set_title("Diagrama de Burbuja Temporal de Tópicos", fontsize=20, weight='bold')
ax.set_xlabel("Fecha del Evento", fontsize=14)
ax.set_ylabel("Coeficiente de Silueta Promedio", fontsize=14)

# Agregar una leyenda para los tópicos
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=color_mapping[topic], markersize=10)
           for topic in unique_topics]
ax.legend(handles, unique_topics, title="Tópicos", bbox_to_anchor=(1.05, 1), loc='upper left')

# Agregar solo las líneas punteadas para fechas importantes
for fecha in fechas_importantes:
    ax.axvline(fecha, color='red', linestyle='--', alpha=0.7)
    # La línea de texto se elimina para mantener solo la línea punteada
    # ax.text(fecha, ax.get_ylim()[1], fecha.strftime('%Y-%m-%d'),
    #         rotation=90, verticalalignment='bottom', color='red', fontsize=10)

# Mejorar la estética general
plt.tight_layout()

# Guardar y mostrar el gráfico
plt.savefig("diagrama_burbuja.pdf", dpi=300)
plt.show()
