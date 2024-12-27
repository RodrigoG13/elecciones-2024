import pandas as pd
from bertopic import BERTopic
from nltk.corpus import stopwords
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP  # Importar UMAP para reducir dimensiones
import nltk
nltk.download('stopwords')

# Cargar stopwords
stop_words = set(stopwords.words("spanish"))

# Función para eliminar stopwords
def eliminar_stopwords(texto):
    palabras = texto.lower().split()  # Convertir a minúsculas y dividir en palabras
    palabras_limpias = [palabra for palabra in palabras if palabra.isalpha() and palabra not in stop_words]
    return " ".join(palabras_limpias)

# Diccionario con las fechas de los eventos
fechas_eventos = {
    "Debate 1": "2024-04-07",
    "Debate 2": "2024-04-28",
    "Debate 3": "2024-05-19",
    "Día de la elección": "2024-06-02"
}

# Cargar los datos
youtube_data = pd.read_csv("debates.csv")  # Cambia a la ruta de tu archivo
x_data = pd.read_csv("diaEleccion.csv")  # Cambia a la ruta de tu archivo

# Preprocesar los comentarios
youtube_data["comentario_editado"] = (
    youtube_data["comentario_editado"].astype(str).fillna("").apply(eliminar_stopwords)
)
x_data["comentario_editado"] = (
    x_data["comentario_editado"].astype(str).fillna("").apply(eliminar_stopwords)
)

# Configurar UMAP para 2 dimensiones
umap_model = UMAP(n_neighbors=15, n_components=2, metric='cosine', random_state=42)

# Instanciar BERTopic con UMAP configurado
topic_model = BERTopic(language="spanish", umap_model=umap_model)

# Lista para almacenar los resultados
resultados_topicos = []
topicos_seleccionados = []

# Procesar cada evento (debates y día de elección)
eventos = {
    "Debate 1": youtube_data[youtube_data["num_debate"] == 1]["comentario_editado"].tolist(),
    "Debate 2": youtube_data[youtube_data["num_debate"] == 2]["comentario_editado"].tolist(),
    "Debate 3": youtube_data[youtube_data["num_debate"] == 3]["comentario_editado"].tolist(),
    "Día de la elección": x_data["comentario_editado"].tolist()
}

for evento, comentarios in eventos.items():
    if not comentarios:
        print(f"{evento} no tiene comentarios válidos.")
        continue

    print(f"Procesando los temas de {evento}...")

    # Ajustar el modelo a los comentarios del evento
    topics, _ = topic_model.fit_transform(comentarios)

    # Reducir temas similares
    topic_model.reduce_topics(comentarios)

    # Guardar resultados del modelo
    topic_model.save(f"bertopic_{evento.replace(' ', '_').lower()}")

    # Generar embeddings
    try:
        embeddings = topic_model.embedding_model.embed_documents(comentarios, show_progress_bar=False)
    except TypeError:
        embeddings = topic_model.embedding_model.embed_documents(comentarios)

    # Reducir embeddings con UMAP
    reduced_embeddings = topic_model.umap_model.transform(embeddings)

    # Calcular la silueta promedio
    sil_score = silhouette_score(reduced_embeddings, topics)
    print(f"Coeficiente de Silueta para {evento}: {sil_score:.2f}")

    # Calcular las muestras de silueta para cada documento
    silhouette_vals = silhouette_samples(reduced_embeddings, topics)

    # Calcular el coeficiente de silueta promedio por tópico
    df_silueta = pd.DataFrame({
        'topic': topics,
        'silhouette': silhouette_vals
    })
    df_silueta = df_silueta[df_silueta['topic'] != -1]
    sil_por_topico = df_silueta.groupby('topic')['silhouette'].mean().reset_index()
    sil_por_topico.rename(columns={'silhouette': 'silhouette_promedio'}, inplace=True)

    # Información de los tópicos
    topics_info = topic_model.get_topic_info()
    topics_info = topics_info.merge(sil_por_topico, how='left', left_on='Topic', right_on='topic')
    topics_info.drop(columns=['topic'], inplace=True)
    topics_info['evento'] = evento
    topics_info['fecha'] = fechas_eventos[evento]  # Añadir la fecha del evento
    topics_info = topics_info[['evento', 'fecha', 'Topic', 'Count', 'Name', 'silhouette_promedio']]
    resultados_topicos.append(topics_info)

    # Seleccionar los 5 mejores tópicos
    topicos_mejores = topics_info.sort_values(
        by=['silhouette_promedio', 'Count'], ascending=[False, False]
    ).head(5)
    topicos_seleccionados.append(topicos_mejores)

    # --- Gráfica de silueta ---
    fig, ax = plt.subplots(figsize=(12, 10))
    y_lower = 10

    unique_topics = sil_por_topico['topic'].unique()
    for i in unique_topics:
        cluster_silhouette_vals = silhouette_vals[np.array(topics) == i]
        cluster_silhouette_vals.sort()
        y_upper = y_lower + len(cluster_silhouette_vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, alpha=0.7)
        if i % 5 == 0:
            ax.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_vals), str(i), fontsize=8, rotation=45)
        y_lower = y_upper + 10

    ax.set_title(f"Gráfica de Silueta para {evento}", fontsize=14)
    ax.set_xlabel("Coeficiente de Silueta", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)
    #plt.axvline(sil_score, color="red", linestyle="--", label=f"Promedio: {sil_score:.2f}")
    plt.legend()
    plt.savefig(f"silhouette_{evento.replace(' ', '_').lower()}.pdf")
    plt.close()

    # --- Gráfica 2D de clusters ---
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        reduced_embeddings[:, 0], 
        reduced_embeddings[:, 1], 
        c=topics, cmap="tab10", alpha=0.6
    )
    ax.set_title(f"Visualización de Clusters en 2D para {evento}")
    ax.set_xlabel("UMAP Dimensión 1")
    ax.set_ylabel("UMAP Dimensión 2")
    plt.colorbar(scatter, label="Clusters")
    plt.savefig(f"clusters_{evento.replace(' ', '_').lower()}.pdf")
    plt.close()

# Guardar los resultados en un archivo CSV
df_resultados = pd.concat(resultados_topicos, ignore_index=True)
df_resultados.to_csv("topicos.csv", index=False)

# Guardar los mejores tópicos en otro archivo CSV
df_topicos_seleccionados = pd.concat(topicos_seleccionados, ignore_index=True)
df_topicos_seleccionados.to_csv("selected_topics.csv", index=False)

print("\nLas visualizaciones se han guardado en PDFs separados.")
print("Los resultados generales se han guardado en 'topicos.csv'.")
print("Los mejores tópicos por evento se han guardado en 'selected_topics.csv'.")
