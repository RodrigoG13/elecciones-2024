import os
import csv
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from safetensors.torch import load_file

########################################
# 1. Definición de Clases y Funciones #
########################################

class BetoMultiOutput(nn.Module):
    def __init__(
        self,
        model_name,
        num_labels_per_candidate=3,
        num_candidates=4,
        use_focal_loss=False,
        class_weights=None,
        alpha=1,
        gamma=2
    ):
        super().__init__()
        self.num_labels_per_candidate = num_labels_per_candidate
        self.num_candidates = num_candidates
        self.use_focal_loss = use_focal_loss
        self.class_weights = class_weights

        # Cargar el modelo base (BETO/BERT)
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Definir el clasificador
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_labels_per_candidate * num_candidates)
        )

        # Definir funciones de pérdida (solo se usan con labels, en entrenamiento/evaluación)
        if self.use_focal_loss:
            # Si tuvieras Focal Loss, la definirías aquí. Se omite para simplificar.
            pass
        else:
            self.loss_fcts = {}
            for c in range(num_candidates):
                if self.class_weights:
                    self.loss_fcts[c] = nn.CrossEntropyLoss(weight=self.class_weights[c])
                else:
                    self.loss_fcts[c] = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)  # [batch_size, num_candidates * num_labels_per_candidate]
        reshaped_logits = logits.view(-1, self.num_candidates, self.num_labels_per_candidate)

        loss = None
        if labels is not None:
            # Solo se calcularía la pérdida si pasas 'labels'
            losses = []
            for c in range(self.num_candidates):
                if self.use_focal_loss:
                    pass  # Aquí iría FocalLoss
                else:
                    loss_c = self.loss_fcts[c](reshaped_logits[:, c, :], labels[:, c])
                losses.append(loss_c)
            loss = torch.stack(losses).mean()

        return {"loss": loss, "logits": reshaped_logits}


def recodificar(valor):
    """
    Recodifica la salida de 0,1,2 → -1,0,1.
    """
    if valor == 0:
        return -1
    elif valor == 1:
        return 0
    elif valor == 2:
        return 1
    # Si, por alguna razón, es otro valor, devuélvelo tal cual o maneja un caso especial
    return 0


def inferir_y_guardar(csv_entrada, csv_salida, model, tokenizer, device):
    """
    Carga un CSV, realiza inferencia usando el modelo y tokenizador dados, 
    recodifica las predicciones y guarda el resultado a otro CSV con comillas.
    """

    # 1) Cargar CSV
    df = pd.read_csv(csv_entrada)

    # Verificar que exista la columna "comentario_editado"
    if "comentario_editado" not in df.columns:
        raise ValueError(f"El dataset {csv_entrada} no tiene la columna 'comentario_editado'.")

    # Convertir a string y asegurar no tener NaN
    df["comentario_editado"] = df["comentario_editado"].astype(str)
    df["comentario_editado"] = df["comentario_editado"].fillna("")

    # 2) Convertir a Dataset HuggingFace y tokenizar
    dataset_hf = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(
            text=examples["comentario_editado"], 
            truncation=True,
            padding="max_length",
            max_length=256
        )

    tokenized_dataset = dataset_hf.map(tokenize_function, batched=True)

    # Dejar solo 'input_ids' y 'attention_mask'
    keep_cols = ["input_ids", "attention_mask"]
    tokenized_dataset = tokenized_dataset.remove_columns(
        [col for col in tokenized_dataset.column_names if col not in keep_cols]
    )

    # 3) Preparar DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=16,
        collate_fn=data_collator
    )

    # 4) Inferencia
    preds_xochitl = []
    preds_claudia = []
    preds_maynez  = []
    preds_ninguno = []

    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inferencia sobre {csv_entrada}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]  # [batch_size, 4, 3] si hay 4 candidatos y 3 clases
            predicted = torch.argmax(logits, dim=-1).cpu().numpy()  # [batch_size, 4]

            preds_xochitl.extend(predicted[:, 0])
            preds_claudia.extend(predicted[:, 1])
            preds_maynez.extend(predicted[:, 2])
            preds_ninguno.extend(predicted[:, 3])

    # 5) Agregar columnas recodificadas
    assert len(df) == len(preds_xochitl), "Inconsistencia en cantidad de filas vs. predicciones"

    df["Xóchitl"] = [recodificar(v) for v in preds_xochitl]
    df["Claudia"] = [recodificar(v) for v in preds_claudia]
    df["Maynez"]  = [recodificar(v) for v in preds_maynez]
    df["Ninguno"] = [recodificar(v) for v in preds_ninguno]

    # 6) Guardar con comillas
    df.to_csv(csv_salida, index=False, quoting=csv.QUOTE_ALL)
    print(f"Guardado: {csv_salida} con {len(df)} filas.")


########################################
# 2. Cargar Modelo y Tokenizador
########################################

save_path = "mis_modelos/mi_beto_finetuned9"
if not os.path.isdir(save_path):
    raise ValueError(f"La ruta del modelo no existe: {save_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# Pesos de clase (opcional, para entrenamiento)
class_weights_dict = {
    0: torch.tensor([0.9370, 0.6096, 3.4209], device=device),   
    1: torch.tensor([2.0877, 0.5883, 1.2178], device=device),   
    2: torch.tensor([4.1136, 0.3905, 5.0948], device=device),   
    3: torch.tensor([3.0401, 0.3879, 10.7484], device=device)
}

tokenizer = AutoTokenizer.from_pretrained(save_path)

model = BetoMultiOutput(
    model_name=save_path,
    num_labels_per_candidate=3,
    num_candidates=4,
    use_focal_loss=False,
    class_weights=class_weights_dict
)

# Cargar pesos (model.safetensors) usando safetensors
model_weights = load_file(os.path.join(save_path, "model.safetensors"))
model.load_state_dict(model_weights)
model.to(device)
model.eval()

########################################
# 3. Inferir y Guardar cada CSV
########################################

debates_in  = "datasets/debatesSinEtiquetas.csv"
debates_out = "datasets/debatesEtiquetado.csv"

dia_in  = "datasets/diaEleccionSinEtiquetas.csv"
dia_out = "datasets/diaEleccionEtiquetado.csv"

# Procesamos cada CSV de forma independiente
inferir_y_guardar(debates_in, debates_out, model, tokenizer, device)
inferir_y_guardar(dia_in, dia_out, model, tokenizer, device)

print("¡Proceso de inferencia completado para ambos archivos!")
