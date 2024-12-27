from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer, BertModel
import torch
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from umap import UMAP
from hdbscan import HDBSCAN
import re

# Descargar stopwords de NLTK
nltk.download("stopwords")

# Cargar palabras vacías en español desde NLTK
spanish_stopwords = stopwords.words("spanish")

# Función de preprocesamiento
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar caracteres especiales
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)
    # Eliminar múltiples espacios
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Cargar los datasets
debates = pd.read_csv("debates.csv", quotechar='"', encoding="utf-8")
dia_eleccion = pd.read_csv("diaEleccion.csv", quotechar='"', encoding="utf-8")

# Convertir y limpiar comentarios
debates["comentario"] = debates["comentario"].astype(str).fillna("").apply(preprocess_text)
dia_eleccion["comentario"] = dia_eleccion["comentario"].astype(str).fillna("").apply(preprocess_text)

# Verificar que no hay valores no string
for debate_num in debates["num_debate"].unique():
    debate_comments = debates[debates["num_debate"] == debate_num]["comentario"]
    non_string = debate_comments[~debate_comments.apply(lambda x: isinstance(x, str))]
    if not non_string.empty:
        print(f"Debate {debate_num} tiene comentarios no string:")
        print(non_string)

eleccion_comments = dia_eleccion["comentario"]
non_string_elec = eleccion_comments[~eleccion_comments.apply(lambda x: isinstance(x, str))]
if not non_string_elec.empty:
    print("Comentarios del día de la elección tienen valores no string:")
    print(non_string_elec)

# Fechas asociadas a cada debate
debate_dates = {
    1: "2024-04-07",  # Primer debate
    2: "2024-04-28",  # Segundo debate
    3: "2024-05-19"   # Tercer debate
}
election_date = "2024-06-02"  # Día de la elección

# Función para graficar siluetas
def plot_silhouette(embeddings, topics, title, filename):
    from sklearn.metrics import silhouette_score, silhouette_samples

    silhouette_avg = silhouette_score(embeddings, topics)
    sample_silhouette_values = silhouette_samples(embeddings, topics)
    print(f"Puntaje promedio de silueta: {silhouette_avg}")

    plt.figure(figsize=(10, 7))
    y_lower = 10
    for i in sorted(set(topics)):
        # Obtener siluetas para este cluster
        cluster_silhouette = sample_silhouette_values[np.array(topics) == i]

        # Continuar solo si el cluster no está vacío
        if cluster_silhouette.size > 0:
            cluster_silhouette.sort()
            size_cluster = len(cluster_silhouette)
            y_upper = y_lower + size_cluster

            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette, alpha=0.7)
            plt.text(-0.05, y_lower + 0.5 * size_cluster, str(i))
            y_lower = y_upper + 10

    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.title(title)
    plt.xlabel("Puntaje de silueta")
    plt.ylabel("Clusters")
    plt.savefig(filename)  # Guardar cada gráfica en su PDF
    plt.close()

# Inicializar un diccionario para guardar modelos y resultados por debate
results_by_debate = {}
topicos_texto = []

# Ajustar el tamaño mínimo de los tópicos
min_topic_size = 20  # Aumentado para mejorar la coherencia

# Configurar UMAP y HDBSCAN
umap_model = UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size, metric='euclidean', cluster_selection_method='eom')

# Inicializar el tokenizador y el modelo BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")

# Función para generar embeddings usando BERT
def get_bert_embeddings(texts):
    # Tokenizar las entradas
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = bert_model(**encoded_input)
    # Obtener las últimas capas ocultas
    embeddings = model_output.last_hidden_state
    # Hacer mean pooling
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).numpy()

# Procesar cada debate por separado
for debate_num in debates["num_debate"].unique():
    print(f"Procesando debate {debate_num}...")
    debate_comments = debates[debates["num_debate"] == debate_num]["comentario"].tolist()
    debate_dates_list = [debate_dates[debate_num]] * len(debate_comments)  # Fecha fija para todos los comentarios del debate

    # Generar embeddings con BERT
    embeddings = get_bert_embeddings(debate_comments)

    # Verificar la forma de los embeddings
    print(f"Embeddings shape (Debate {debate_num}): {embeddings.shape}")

    # Entrenar BERTopic para este debate
    model = BERTopic(
        vectorizer_model=CountVectorizer(stop_words=spanish_stopwords),
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=min_topic_size
    )
    try:
        topics, probs = model.fit_transform(debate_comments, embeddings)
    except Exception as e:
        print(f"Error al procesar el debate {debate_num}: {e}")
        continue

    # Calcular y graficar el puntaje de silueta
    silhouette_filename = f"silueta_debate_{debate_num}.pdf"
    plot_silhouette(embeddings, topics, f"Silueta del Debate {debate_num}", silhouette_filename)

    # Guardar el modelo y resultados
    results_by_debate[debate_num] = {
        "model": model,
        "topics": topics,
        "comments": debate_comments,
        "dates": debate_dates_list,
    }

    # Guardar tópicos principales en texto
    topicos_texto.append(f"Debate {debate_num}:\n")
    topicos_texto.append(model.get_topic_info().to_string(index=False) + "\n")

    # Guardar gráficas de tópicos en HTML
    bubbles_filename = f"burbujas_debate_{debate_num}.html"  # Cambiado a HTML
    bars_filename = f"barras_debate_{debate_num}.html"      # Cambiado a HTML
    model.visualize_topics().write_html(bubbles_filename)
    model.visualize_barchart().write_html(bars_filename)

# Procesar los comentarios del día de la elección
print("Procesando comentarios del día de la elección...")
eleccion_comments = dia_eleccion["comentario"].tolist()
eleccion_dates_list = [election_date] * len(eleccion_comments)  # Fecha fija para el día de la elección
eleccion_embeddings = get_bert_embeddings(eleccion_comments)

# Verificar la forma de los embeddings
print(f"Embeddings shape (Día de la elección): {eleccion_embeddings.shape}")

# Entrenar BERTopic para el día de la elección
eleccion_model = BERTopic(
    vectorizer_model=CountVectorizer(stop_words=spanish_stopwords),
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    min_topic_size=min_topic_size
)
try:
    eleccion_topics, eleccion_probs = eleccion_model.fit_transform(eleccion_comments, eleccion_embeddings)
except Exception as e:
    print(f"Error al procesar el día de la elección: {e}")

# Calcular y graficar el puntaje de silueta para el día de la elección
silhouette_filename = "silueta_dia_eleccion.pdf"
plot_silhouette(eleccion_embeddings, eleccion_topics, "Silueta del Día de la Elección", silhouette_filename)

# Guardar gráficas de tópicos del día de la elección en HTML
bubbles_filename = "burbujas_dia_eleccion.html"  # Cambiado a HTML
bars_filename = "barras_dia_eleccion.html"      # Cambiado a HTML
eleccion_model.visualize_topics().write_html(bubbles_filename)
#eleccion_model.visualize_barchart().write_html(barras_filename)

# Guardar tópicos principales del día de la elección en texto
topicos_texto.append("Día de la Elección:\n")
topicos_texto.append(eleccion_model.get_topic_info().to_string(index=False) + "\n")

# Consolidar todos los resultados para análisis temporal
print("Consolidando resultados...")
all_comments = []
all_dates = []
all_topics = []
for debate_num, result in results_by_debate.items():
    all_comments.extend(result["comments"])
    all_dates.extend(result["dates"])
    all_topics.extend(result["topics"])

# Agregar resultados del día de la elección
all_comments.extend(eleccion_comments)
all_dates.extend(eleccion_dates_list)
all_topics.extend(eleccion_topics)

# Crear un DataFrame para análisis temporal
df = pd.DataFrame({"comment": all_comments, "date": pd.to_datetime(all_dates), "topic": all_topics})

# Guardar resultados finales en archivo de texto
with open("topicos_principales.txt", "w", encoding="utf-8") as f:
    f.writelines(topicos_texto)

print("Análisis completado. Gráficas y archivos guardados.")
