import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def guardar_matrices_confusion_pdf(candidate_names=None):
    """
    Carga las matrices de confusión .npy de la carpeta actual,
    y las guarda en PDF utilizando seaborn para la visualización.
    """
    if candidate_names is None:
        candidate_names = ["Xóchitl", "Claudia", "Maynez", "Ninguno"]
    
    for cand in candidate_names:
        # Ruta del archivo que contiene la matriz de confusión
        cm_file = f"conf_matrix_{cand}.npy"
        
        # Carga la matriz de confusión
        cm = np.load(cm_file)
        
        # Crear la figura
        plt.figure(figsize=(6, 6))
        
        # Crear el mapa de calor
        # Ajusta las etiquetas xticklabels e yticklabels según tus clases
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=[0, 1, 2],  # O la secuencia de clases que uses
            yticklabels=[0, 1, 2]
        )
        
        # Etiquetas de los ejes y título
        plt.xlabel("Predicción")
        plt.ylabel("Etiqueta real")
        plt.title(f"Matriz de confusión: {cand}")
        
        # Ajustar el layout para que no se sobrepongan
        plt.tight_layout()
        
        # Guardar en PDF
        pdf_file = f"conf_matrix_{cand}.pdf"
        plt.savefig(pdf_file)
        
        # Cerrar la figura para no sobrecargar memoria
        plt.close()
        
        print(f"Matriz de confusión para {cand} guardada en {pdf_file}")

# Ejecución de la función
if __name__ == "__main__":
    guardar_matrices_confusion_pdf()
