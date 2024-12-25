import os
import pandas as pd
import torch
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    TrainingArguments, 
    Trainer, 
    TrainerCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

##################################
# 1) Cargar y combinar datasets #
##################################

def sentiment_to_label(x):
    """
    Transforma las etiquetas de sentimiento:
    -1 -> 0, 0 -> 1, 1 -> 2
    Otros valores o NaN -> 1 (neutral)
    """
    if pd.isna(x):
        return 1  # Valor por defecto para NaN
    elif x == -1:
        return 0
    elif x == 0:
        return 1
    elif x == 1:
        return 2
    else:
        print(f"Valor de sentimiento inesperado: {x}")
        return 1  # Valor por defecto

# Rutas de tus archivos CSV
debates_csv = "datasets/debates.csv"              # Dataset de debates
election_day_csv = "datasets/election_day.csv"    # Dataset del día de la elección

# Verifica que los archivos existen
if not os.path.isfile(debates_csv):
    raise FileNotFoundError(f"No se encontró el archivo: {debates_csv}")
if not os.path.isfile(election_day_csv):
    raise FileNotFoundError(f"No se encontró el archivo: {election_day_csv}")

# Carga de CSVs
debates_df = pd.read_csv(debates_csv)
election_day_df = pd.read_csv(election_day_csv)

# Verifica que las columnas necesarias existan en ambos datasets
required_columns = ["comentario_editado", "Xóchitl", "Claudia", "Maynez", "Ninguno"]

for df, name in zip([debates_df, election_day_df], ["debates.csv", "election_day.csv"]):
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"El dataset {name} le falta(n) la(s) columna(s): {missing_cols}")

# Selecciona únicamente las columnas relevantes
debates_df = debates_df[required_columns]
election_day_df = election_day_df[required_columns]

##########################################
# 1.1) Diagnosticar Tipos de Datos y Valores Únicos #
##########################################

def diagnosticar_tipos_y_valores(df, dataset_name, columnas):
    """
    Diagnostica y reporta los tipos de datos y valores únicos en las columnas especificadas.
    """
    for col in columnas:
        unique_values = df[col].unique()
        data_type = df[col].dtype
        print(f"Dataset '{dataset_name}', Columna '{col}':")
        print(f"  Tipo de dato: {data_type}")
        print(f"  Valores únicos: {unique_values}\n")

print("=== Diagnóstico de Tipos de Datos y Valores Únicos ===")
diagnosticar_tipos_y_valores(debates_df, "debates.csv", ["Xóchitl", "Claudia", "Maynez", "Ninguno"])
diagnosticar_tipos_y_valores(election_day_df, "election_day.csv", ["Xóchitl", "Claudia", "Maynez", "Ninguno"])
print("======================================================\n")

# Convertir las columnas de sentimiento a int si no lo están, excepto 'Xóchitl' en election_day.csv
for col in ["Claudia", "Maynez", "Ninguno"]:
    # Convertir en debates_df
    if debates_df[col].dtype != 'int64':
        try:
            debates_df[col] = debates_df[col].astype(int)
            print(f"Convertido '{col}' a int en debates.csv.")
        except ValueError:
            print(f"Error al convertir '{col}' a int en debates.csv. Revisar los datos.")
    # Convertir en election_day_df
    if election_day_df[col].dtype != 'int64':
        try:
            election_day_df[col] = election_day_df[col].astype(int)
            print(f"Convertido '{col}' a int en election_day.csv.")
        except ValueError:
            print(f"Error al convertir '{col}' a int en election_day.csv. Revisar los datos.")

# Limpieza específica para 'Xóchitl' en election_day.csv
print("=== Limpieza de la columna 'Xóchitl' en election_day.csv ===")
# Utilizar .loc para evitar SettingWithCopyWarning
election_day_df.loc[:, 'Xóchitl'] = election_day_df['Xóchitl'].astype(str).str.replace("'", "").str.strip()

# Verificar si la limpieza fue exitosa
unique_vals_after_clean = election_day_df['Xóchitl'].unique()
print(f"Valores únicos en 'Xóchitl' después de limpieza: {unique_vals_after_clean}\n")

# Intentar convertir 'Xóchitl' a int, manejando errores
def convertir_a_int(x):
    try:
        return int(x)
    except ValueError:
        print(f"Error al convertir '{x}' a int en 'Xóchitl'. Asignando valor por defecto 1.")
        return 1  # Valor por defecto para valores inválidos

election_day_df.loc[:, 'Xóchitl'] = election_day_df['Xóchitl'].apply(convertir_a_int)

# Re-diagnosticar tipos de datos y valores únicos después de la limpieza
print("=== Re-Diagnóstico de Tipos de Datos y Valores Únicos ===")
diagnosticar_tipos_y_valores(election_day_df, "election_day.csv", ["Xóchitl", "Claudia", "Maynez", "Ninguno"])
print("==========================================================\n")

##########################################
# 1.2) Diagnosticar NaN y Valores Inválidos #
##########################################

def diagnosticar_datos_problema(df, dataset_name, columnas):
    """
    Diagnostica y reporta filas con NaN o valores inesperados en las columnas especificadas.
    """
    valores_esperados = {-1, 0, 1}
    for col in columnas:
        # Detectar NaN
        nan_filas = df[df[col].isna()]
        if not nan_filas.empty:
            print(f"\n[ADVERTENCIA] Dataset '{dataset_name}', Columna '{col}': Encontrados {len(nan_filas)} valores NaN.")
            print(nan_filas[[col]].head())  # Muestra las primeras 5 filas con NaN
        # Detectar valores inesperados
        invalid_filas = df[~df[col].isin(valores_esperados)]
        if not invalid_filas.empty:
            print(f"\n[ADVERTENCIA] Dataset '{dataset_name}', Columna '{col}': Encontrados {len(invalid_filas)} valores inesperados.")
            print(invalid_filas[[col]].head())  # Muestra las primeras 5 filas con valores inesperados

# Lista de columnas de sentimiento
sentiment_columns = ["Xóchitl", "Claudia", "Maynez", "Ninguno"]

print("=== Diagnóstico de Datos ===")
diagnosticar_datos_problema(debates_df, "debates.csv", sentiment_columns)
diagnosticar_datos_problema(election_day_df, "election_day.csv", sentiment_columns)
print("============================\n")

# Manejar Datos Problema: Eliminar filas con NaN o valores inesperados
valores_esperados = {-1, 0, 1}
for col in sentiment_columns:
    # Eliminar filas con NaN en la columna
    debates_df = debates_df.dropna(subset=[col])
    election_day_df = election_day_df.dropna(subset=[col])
    
    # Eliminar filas con valores inesperados en la columna
    debates_df = debates_df[debates_df[col].isin(valores_esperados)]
    election_day_df = election_day_df[election_day_df[col].isin(valores_esperados)]

print("=== Datos después de limpieza ===")
print(f"Debates.csv: {len(debates_df)} filas")
print(f"Election_day.csv: {len(election_day_df)} filas")
print("===============================\n")


print("----------------")
print("PUERCO")
# Lista de columnas a analizar
columns_to_check = ["Xóchitl", "Claudia", "Maynez", "Ninguno"]

# Diccionario para almacenar los resultados
counts = {col: election_day_df[col].value_counts() for col in columns_to_check}

# Imprimir los resultados
for col in columns_to_check:
    print(f"Resultados para la columna '{col}':")
    print(f"  Total de 1: {counts[col].get(1, 0)}")
    print(f"  Total de -1: {counts[col].get(-1, 0)}")
    print(f"  Total de 0: {counts[col].get(0, 0)}\n")

input()

# Aplica la transformación de etiquetas usando .loc para evitar SettingWithCopyWarning
for col in sentiment_columns:
    debates_df.loc[:, col] = debates_df[col].apply(sentiment_to_label)
    election_day_df.loc[:, col] = election_day_df[col].apply(sentiment_to_label)

# Combina ambos datasets
combined_df = pd.concat([debates_df, election_day_df], ignore_index=True)
# Verificar los tipos de datos en 'comentario_editado'
print(combined_df['comentario_editado'].apply(type).value_counts())
# Mostrar el número de filas antes de la limpieza
print(f"Número de filas antes de eliminar valores no string: {len(combined_df)}")

# Eliminar filas donde 'comentario_editado' no es una cadena de texto
combined_df = combined_df[combined_df['comentario_editado'].apply(lambda x: isinstance(x, str))].reset_index(drop=True)

# Mostrar el número de filas después de la limpieza
print(f"Número de filas después de eliminar valores no string: {len(combined_df)}")


# Realiza un muestreo aleatorio y separa 80% para entrenamiento y 20% para validación
train_df, valid_df = train_test_split(combined_df, test_size=0.2, random_state=42, shuffle=True)

print(f"Total de muestras después de limpieza: {len(combined_df)}")
print(f"Muestras de entrenamiento: {len(train_df)}")
print(f"Muestras de validación: {len(valid_df)}")

# Crea datasets de Huggingface
train_dataset_hf = Dataset.from_pandas(train_df)
valid_dataset_hf = Dataset.from_pandas(valid_df)

###################################
# 2) Preparar tokenizador (BETO) #
###################################
# Define la ruta absoluta a la carpeta del modelo local
model_path = os.path.expanduser("~/bert-base-spanish-wwm-cased")  # Ajusta si es necesario

# Verifica si la ruta existe
if not os.path.isdir(model_path):
    raise ValueError(f"La ruta del modelo local no existe: {model_path}")

# Carga el tokenizador desde la carpeta local
tokenizer = AutoTokenizer.from_pretrained(model_path)

def tokenize_function(examples):
    return tokenizer(
        text=examples["comentario_editado"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

# Tokeniza los datasets con batched=True para mayor eficiencia
train_dataset_hf = train_dataset_hf.map(tokenize_function, batched=True)
valid_dataset_hf = valid_dataset_hf.map(tokenize_function, batched=True)

# Función para formatear etiquetas
def format_labels(example):
    return {
        "input_ids": example["input_ids"],
        "attention_mask": example["attention_mask"],
        "labels": [
            example["Xóchitl"],
            example["Claudia"],
            example["Maynez"],
            example["Ninguno"]
        ]
    }

# Aplica el formateo de etiquetas
train_dataset_hf = train_dataset_hf.map(format_labels, batched=False)
valid_dataset_hf = valid_dataset_hf.map(format_labels, batched=False)

# Elimina columnas que sobran
keep_cols = ["input_ids", "attention_mask", "labels"]
train_dataset_hf = train_dataset_hf.remove_columns(
    [c for c in train_dataset_hf.column_names if c not in keep_cols]
)
valid_dataset_hf = valid_dataset_hf.remove_columns(
    [c for c in valid_dataset_hf.column_names if c not in keep_cols]
)

# Convierte a tensores
train_dataset_hf = train_dataset_hf.with_format("torch")
valid_dataset_hf = valid_dataset_hf.with_format("torch")

####################################
# 3) Definir el modelo multi-salida #
####################################
class BetoMultiOutput(nn.Module):
    def __init__(self, model_name, num_labels_per_candidate=3, num_candidates=4):
        super().__init__()
        self.num_labels_per_candidate = num_labels_per_candidate
        self.num_candidates = num_candidates
        
        # Cargar el modelo base
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Capa final: num_candidates * num_labels_per_candidate logits
        self.classifier = nn.Linear(
            hidden_size, 
            num_labels_per_candidate * num_candidates
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Tomar la salida del token [CLS]
        pooled_output = outputs.last_hidden_state[:, 0]
        
        logits = self.classifier(pooled_output)  # [batch_size, num_candidates * num_labels_per_candidate]
        reshaped_logits = logits.view(-1, self.num_candidates, self.num_labels_per_candidate)
        # reshaped_logits: [batch_size, num_candidates, num_labels_per_candidate]

        loss = None
        if labels is not None:
            # labels: [batch_size, num_candidates]
            loss_fct = nn.CrossEntropyLoss()
            losses = []
            for c in range(self.num_candidates):
                losses.append(loss_fct(reshaped_logits[:, c, :], labels[:, c]))
            loss = torch.stack(losses).mean()  # Puedes usar sum() si lo prefieres

        return {"loss": loss, "logits": reshaped_logits}

#####################################################
# 4) Definir métricas (accuracy, matriz de confusión, etc.)
#####################################################
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)  # [batch_size, num_candidates]

    metrics = {}
    candidate_names = ["Xóchitl", "Claudia", "Maynez", "Ninguno"]
    
    # Inicializar listas para almacenar métricas globales
    accuracies = []
    
    for i, candidate_name in enumerate(candidate_names):
        y_true = labels[:, i]
        y_pred = predictions[:, i]
        
        # Calcular exactitud
        accuracy = (y_true == y_pred).mean()
        metrics[f"accuracy_{candidate_name}"] = accuracy
        accuracies.append(accuracy)
        
        # Calcular y guardar la matriz de confusión
        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
        np.save(f"conf_matrix_{candidate_name}.npy", cm)
    
    # Calcular exactitud global
    metrics["accuracy_global"] = np.mean(accuracies)
    return metrics



##########################################
# 5) Instanciar el modelo y configurar el Trainer #
##########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# Instanciar el modelo
model = BetoMultiOutput(model_name=model_path)
model.to(device)

# Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="beto-multi",           # Carpeta temporal para checkpoints
    eval_strategy="epoch",             # Reemplazado de 'evaluation_strategy'
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy_global",
    logging_dir="logs"                  # Carpeta para logs
)

# Instanciar el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_hf,
    eval_dataset=valid_dataset_hf,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

###############################################
# 6) Definir y añadir un callback para métricas por época #
###############################################
from transformers import TrainerCallback

class MyMetricCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs.get('trainer', None)
        
        if trainer is None:
            print("Advertencia: 'trainer' no se encontró en kwargs. No se pueden calcular métricas.")
            return control
        
        # Evaluar solo sobre el set de validación
        val_metrics = trainer.evaluate(
            eval_dataset=trainer.eval_dataset,
            metric_key_prefix="val"
        )
        print(f"Epoch {state.epoch} - Métricas (validación):")
        print(val_metrics)
        
        return control




# Instanciar el callback
metric_callback = MyMetricCallback()
# Añadir el callback al Trainer
trainer.add_callback(metric_callback)


###################################
# 7) Entrenar el modelo #
###################################
trainer.train()

###################################
# 8) Guardar el modelo y el tokenizador fine-tuned #
###################################
# Definir la carpeta donde guardar el modelo ajustado
"""save_path = "mis_modelos/mi_beto_finetuned"

# Crear la carpeta si no existe
os.makedirs(save_path, exist_ok=True)

# Guardar el modelo y el tokenizador
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print(f"Modelo y tokenizador guardados en la carpeta: {save_path}")"""

# Evaluación final en validación (opcional)
eval_results = trainer.evaluate()
candidate_names = ["Xóchitl", "Claudia", "Maynez", "Ninguno"]
for cand in candidate_names:
    # Cargar la matriz de confusión desde el archivo
    cm = np.load(f"conf_matrix_{cand}.npy")
    print(f"=== {cand} ===")
    print(cm)
    print("Exactitud:", eval_results[f"accuracy_{cand}"])
    print()

print("Exactitud global:", eval_results["accuracy_global"])
