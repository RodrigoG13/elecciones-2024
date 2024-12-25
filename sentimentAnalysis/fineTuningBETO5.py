import os
import pandas as pd
import torch
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    TrainingArguments, 
    Trainer, 
    TrainerCallback,
    DataCollatorWithPadding  # Para manejar la deprecación del tokenizer
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

##################################
# 1) Definición de Funciones Auxiliares #
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

def diagnosticar_datos_problema(df, dataset_name, columnas, valores_esperados):
    """
    Diagnostica y reporta filas con NaN o valores inesperados en las columnas especificadas.
    """
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

##################################
# 2) Carga y Preprocesamiento de Datos #
##################################

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

##################################
# 3) Diagnóstico y Limpieza de Datos #
##################################

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

##################################
# 4) Diagnóstico de NaN y Valores Inválidos #
##################################

# Lista de columnas de sentimiento
sentiment_columns = ["Xóchitl", "Claudia", "Maynez", "Ninguno"]
valores_esperados = {-1, 0, 1}

print("=== Diagnóstico de Datos ===")
diagnosticar_datos_problema(debates_df, "debates.csv", sentiment_columns, valores_esperados)
diagnosticar_datos_problema(election_day_df, "election_day.csv", sentiment_columns, valores_esperados)
print("============================\n")

# Manejar Datos Problema: Eliminar filas con NaN o valores inesperados
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

# Conteo de valores por columna
print("----------------")
print("CONTEO DE VALORES POR COLUMNA")
print("----------------")
for col in sentiment_columns:
    counts = election_day_df[col].value_counts()
    print(f"Resultados para la columna '{col}':")
    print(f"  Total de 1: {counts.get(1, 0)}")
    print(f"  Total de -1: {counts.get(-1, 0)}")
    print(f"  Total de 0: {counts.get(0, 0)}\n")

# Pausa para inspección (puedes comentar o eliminar esta línea si no la necesitas)
# input()


##################################
# 5) Transformación de Etiquetas y Combinación de Datasets #
##################################

# Aplica la transformación de etiquetas usando .loc para evitar SettingWithCopyWarning
for col in sentiment_columns:
    debates_df.loc[:, col] = debates_df[col].apply(sentiment_to_label)
    election_day_df.loc[:, col] = election_day_df[col].apply(sentiment_to_label)

# Combina ambos datasets
combined_df = pd.concat([debates_df, election_day_df], ignore_index=True)

# Verificar los tipos de datos en 'comentario_editado'
print("Tipos de datos en 'comentario_editado':")
print(combined_df['comentario_editado'].apply(type).value_counts())

# Mostrar el número de filas antes de la limpieza
print(f"Número de filas antes de eliminar valores no string: {len(combined_df)}")

# Eliminar filas donde 'comentario_editado' no es una cadena de texto
combined_df = combined_df[combined_df['comentario_editado'].apply(lambda x: isinstance(x, str))].reset_index(drop=True)




"""import pandas as pd
from sklearn.utils import resample

# Filtrar los datos que cumplen con la condición específica
subset_condition = combined_df[
    (combined_df["Xóchitl"] == 1) & 
    (combined_df["Claudia"] == 1) & 
    (combined_df["Maynez"] == 1)
]

# Realizar undersampling del subset
# Define el número de muestras deseadas después del undersampling
n_samples = 100  # Ajusta según lo que desees
subset_downsampled = resample(
    subset_condition,
    replace=False,  # No duplicar muestras
    n_samples=n_samples,
    random_state=42  # Para reproducibilidad
)

# Filtrar el resto de los datos (que no cumplen la condición)
other_data = combined_df.drop(subset_condition.index)

# Recombinar los datos
balanced_df = pd.concat([other_data, subset_downsampled], ignore_index=True)

# Verificar la distribución después del undersampling
print("Distribución después del undersampling:")
print(balanced_df[["Xóchitl", "Claudia", "Maynez"]].value_counts())


combined_df = balanced_df"""


# Mostrar el número de filas después de la limpieza
print(f"Número de filas después de eliminar valores no string: {len(combined_df)}")

# Realiza un muestreo aleatorio y separa 80% para entrenamiento y 20% para validación
train_df, valid_df = train_test_split(combined_df, test_size=0.2, random_state=42, shuffle=True)

print(f"Total de muestras después de limpieza: {len(combined_df)}")
print(f"Muestras de entrenamiento: {len(train_df)}")
print(f"Muestras de validación: {len(valid_df)}")



from sklearn.utils import resample

# Define el número de registros que deseas mantener
n_to_keep = 300  # Ajusta según tus necesidades

# Identificar las filas que cumplen con la condición
train_condition = train_df[
    (train_df["Xóchitl"] == 1) & 
    (train_df["Claudia"] == 1) & 
    (train_df["Maynez"] == 1)
]

# Realizar undersampling para mantener solo 'n_to_keep' registros
train_condition_downsampled = resample(
    train_condition,
    replace=False,  # No duplicar muestras
    n_samples=n_to_keep,  # Número de registros a mantener
    random_state=42       # Para reproducibilidad
)

# Eliminar todas las filas originales que cumplen la condición
train_df = train_df.drop(train_condition.index)

# Reagregar únicamente los registros submuestreados
train_df = pd.concat([train_df, train_condition_downsampled], ignore_index=True)

print("=== Actualización del conjunto de entrenamiento ===")
print(f"Total de muestras después de ajustar: {len(train_df)}")
print(f"Registros específicos con la condición: {len(train_condition_downsampled)}")
print(f"Total de muestras en validación (sin cambios): {len(valid_df)}")




from sklearn.utils import resample

import pandas as pd
from sklearn.utils import resample

def oversample_multi_labeled(
    train_df, 
    sentiment_columns, 
    oversample_factor=2, 
    random_state=42
):
    """
    Duplica (oversampling) únicamente los registros que tienen 
    sentimientos distintos de 0 para más de un candidato.
    
    Parámetros:
    -----------
    train_df : pd.DataFrame
        Conjunto de entrenamiento que ya contiene las columnas de sentimiento.
    sentiment_columns : list
        Lista de columnas que representan cada candidato. 
        Ejemplo: ["Xóchitl", "Claudia", "Maynez", "Ninguno"].
    oversample_factor : int
        Veces que se quiere duplicar el subconjunto multi-labeled.
        Si oversample_factor=2, duplicas ese subconjunto, 
        resultando en el doble de registros para esas filas.
    random_state : int
        Semilla para reproducibilidad.
        
    Retorna:
    --------
    pd.DataFrame
        Nuevo DataFrame con las filas multi-labeled aumentadas por oversampling.
    """

    # 1. Identificar registros "multi-labeled":
    #    Aquellos que tienen más de un sentimiento != 0 en las columnas especificadas.
    #    Puedes ajustar la condición si consideras 1 y 2 como “sentimientos asociados”.
    multi_label_condition = train_df[sentiment_columns].apply(
        lambda row: sum(row != 0), axis=1
    ) > 1

    # Subconjunto multi-labeled
    subset_multi = train_df[multi_label_condition]

    # Subconjunto que NO es multi-labeled (se deja tal cual)
    non_multi = train_df[~multi_label_condition]

    # 2. Calcular cuántas muestras quieres en total para el subset multi-labeled.
    #    Con oversample_factor=2, duplicas su tamaño, 
    #    con factor=3 lo triplicas, etc.
    n_samples = oversample_factor * len(subset_multi)

    # 3. Realizar oversampling (duplicado con reemplazo) sólo para multi-labeled
    subset_multi_oversampled = resample(
        subset_multi,
        replace=True,        # Con reemplazo
        n_samples=n_samples,
        random_state=random_state
    )

    # 4. Combinar todo: el conjunto no multi-labeled, 
    #    el subset multi-labeled original y las copias extras
    result_df = pd.concat([non_multi, subset_multi, subset_multi_oversampled], ignore_index=True)

    # 5. Retornar el DataFrame ya modificado
    return result_df



sentiment_columns = ["Xóchitl", "Claudia", "Maynez", "Ninguno"]
train_df = oversample_multi_labeled(train_df, sentiment_columns, oversample_factor=2)


print("Tamaño final de train_df:", len(train_df))
# Checar cuántas filas hay con sum(row != 0) > 1
multi_label_count = sum(
    train_df[sentiment_columns].apply(lambda row: sum(row != 0), axis=1) > 1
)
print(f"Filas multi-labeled después de oversampling: {multi_label_count}")

input()

# Crea datasets de Huggingface
train_dataset_hf = Dataset.from_pandas(train_df)
valid_dataset_hf = Dataset.from_pandas(valid_df)

###################################
# 6) Preparar Tokenizador (BETO) #
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
        max_length=256
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

###################################
# 7) Cálculo de Pesos de Clase #
###################################

# Definir el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# Definir las clases
classes = [0, 1, 2]

# Calcular los pesos de clase para cada candidato y moverlos al dispositivo
class_weights = {}
for col in sentiment_columns:
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=combined_df[col]
    )
    class_weights[col] = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"Pesos de clase para {col}: {class_weights[col]}")

###################################
# 8) Definición del Modelo con Pesos de Clase y Focal Loss #
###################################

class FocalLoss(nn.Module):
    def __init__(self, alpha=4, gamma=9, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits de tamaño [batch_size, num_classes]
            targets: etiquetas verdaderas de tamaño [batch_size]
        """
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BetoMultiOutput(nn.Module):
    def __init__(self, model_name, class_weights=None, use_focal_loss=True, num_labels_per_candidate=3, num_candidates=4, alpha=1, gamma=2):
        super().__init__()
        self.num_labels_per_candidate = num_labels_per_candidate
        self.num_candidates = num_candidates
        self.use_focal_loss = use_focal_loss
        self.class_weights = class_weights  # Diccionario de pesos por candidato si no se usa Focal Loss

        # Cargar el modelo base
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Capa final: num_candidates * num_labels_per_candidate logits
        self.classifier = nn.Linear(
            hidden_size, 
            num_labels_per_candidate * num_candidates
        )

        if self.use_focal_loss:
            self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            self.loss_fcts = {}
            if self.class_weights:
                for c in range(num_candidates):
                    self.loss_fcts[c] = nn.CrossEntropyLoss(weight=self.class_weights[sentiment_columns[c]])
            else:
                for c in range(num_candidates):
                    self.loss_fcts[c] = nn.CrossEntropyLoss()

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
            losses = []
            for c in range(self.num_candidates):
                if self.use_focal_loss:
                    loss_c = self.focal_loss(reshaped_logits[:, c, :], labels[:, c])
                else:
                    loss_c = self.loss_fcts[c](reshaped_logits[:, c, :], labels[:, c])
                losses.append(loss_c)
            loss = torch.stack(losses).mean()  # Promedio de las pérdidas por candidato

        return {"loss": loss, "logits": reshaped_logits}

###################################
# 9) Definición de Métricas de Evaluación #
###################################

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)  # [batch_size, num_candidates]

    metrics = {}
    candidate_names = ["Xóchitl", "Claudia", "Maynez", "Ninguno"]
    
    # Inicializar listas para almacenar métricas globales
    accuracies = []
    f1_scores = []
    
    for i, candidate_name in enumerate(candidate_names):
        y_true = labels[:, i]
        y_pred = predictions[:, i]
        
        # Calcular exactitud
        accuracy = (y_true == y_pred).mean()
        metrics[f"accuracy_{candidate_name}"] = accuracy
        accuracies.append(accuracy)
        
        # Calcular F1-Score
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics[f"f1_{candidate_name}"] = f1
        f1_scores.append(f1)
        
        # Calcular y guardar la matriz de confusión
        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
        np.save(f"conf_matrix_{candidate_name}.npy", cm)
    
    # Calcular exactitud global y F1 global
    metrics["accuracy_global"] = np.mean(accuracies)
    metrics["f1_global"] = np.mean(f1_scores)
    return metrics

###################################
# 10) Configuración del Trainer de Huggingface #
###################################

# Definir si usar Focal Loss o Pesos de Clase
use_focal_loss = False  # Cambia a True para usar Focal Loss

# Instanciar el modelo
model = BetoMultiOutput(
    model_name=model_path, 
    class_weights=class_weights if not use_focal_loss else None, 
    use_focal_loss=use_focal_loss,
    num_labels_per_candidate=3, 
    num_candidates=4,
    alpha=2,  # Solo si usas Focal Loss
    gamma=4   # Solo si usas Focal Loss
)
model.to(device)

# Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="beto-multi",           # Carpeta temporal para checkpoints
    eval_strategy="epoch",             # Evaluación al final de cada época
    save_strategy="epoch",             # Guardar al final de cada época
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy_global",
    logging_dir="logs"                  # Carpeta para logs
)

# Definir el data_collator para manejar la deprecación del tokenizer
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Instanciar el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_hf,
    eval_dataset=valid_dataset_hf,
    tokenizer=tokenizer,  # Aunque está deprecado, se mantiene por compatibilidad
    data_collator=data_collator,  # Usar data_collator en lugar de tokenizer si es necesario
    compute_metrics=compute_metrics
)

###############################################
# 11) Definición y Adición de Callbacks #
###############################################

"""class MyMetricCallback(TrainerCallback):
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
trainer.add_callback(metric_callback)"""

###################################
# 12) Entrenamiento del Modelo #
###################################

trainer.train()

###################################
# 13) Guardado del Modelo y Tokenizador Fine-Tuned #
###################################

# Definir la carpeta donde guardar el modelo ajustado
save_path = "mis_modelos/mi_beto_finetuned"

# Crear la carpeta si no existe
os.makedirs(save_path, exist_ok=True)

# Guardar el modelo y el tokenizador
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print(f"Modelo y tokenizador guardados en la carpeta: {save_path}")

###################################
# 14) Evaluación Final #
###################################

eval_results = trainer.evaluate()
candidate_names = ["Xóchitl", "Claudia", "Maynez", "Ninguno"]

for cand in candidate_names:
    # Cargar la matriz de confusión desde el archivo
    cm = np.load(f"conf_matrix_{cand}.npy")
    print(f"=== {cand} ===")
    print("Matriz de Confusión:")
    print(cm)
    
    # Acceder a las métricas con el prefijo 'eval_'
    accuracy_key = f"eval_accuracy_{cand}"
    f1_key = f"eval_f1_{cand}"
    
    # Verificar si las claves existen antes de imprimir
    if accuracy_key in eval_results:
        print("Exactitud:", eval_results[accuracy_key])
    else:
        print(f"Advertencia: {accuracy_key} no se encontró en eval_results.")
    
    if f1_key in eval_results:
        print("F1-Score:", eval_results[f1_key])
    else:
        print(f"Advertencia: {f1_key} no se encontró en eval_results.")
    
    print()

# Acceder a las métricas globales con el prefijo 'eval_'
print("Exactitud global:", eval_results.get("eval_accuracy_global", "No disponible"))
print("F1-Score global:", eval_results.get("eval_f1_global", "No disponible"))


###################################
# Visualización de Curvas de Pérdida #
###################################

# Extrae los valores de pérdida desde el historial de entrenamiento
train_loss = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
val_loss = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]

# Genera la curva de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Entrenamiento', marker='o')
plt.plot(val_loss, label='Validación', marker='o')
plt.xlabel('Iteraciones' if len(train_loss) > len(val_loss) else 'Épocas')
plt.ylabel('Pérdida')
plt.title('Curvas de Aprendizaje')
plt.legend()
plt.grid()
plt.show()
