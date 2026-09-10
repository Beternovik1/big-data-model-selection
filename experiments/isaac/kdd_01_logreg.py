import pandas as pd
import numpy as np
import gc
import time
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, 
    roc_auc_score, balanced_accuracy_score
)

os.makedirs("02_RESULTS/KDD/isaac", exist_ok=True)

print("Cargando datos KDD")
df_train = pd.read_csv("01_DATA/KDD/train_500000.csv", header=None)
df_val = pd.read_csv("01_DATA/KDD/validation.csv", header=None)

# Separación (0 a 40 características, 41 target)
X_train = df_train.iloc[:, :-1]
y_train_raw = df_train.iloc[:, -1]

X_val = df_val.iloc[:, :-1]
y_val_raw = df_val.iloc[:, -1]

del df_train, df_val
gc.collect()

print("Mapeando target (normal. -> 0, attack -> 1)")
y_train = np.where(y_train_raw == 'normal.', 0, 1).astype(np.int8)
y_val = np.where(y_val_raw == 'normal.', 0, 1).astype(np.int8)

print("Construyendo pipeline de preprocesamiento híbrido")
cat_cols = [1, 2, 3] 

num_cols = [0] + list(range(4, 41))

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', StandardScaler(), num_cols)
    ],
    remainder='passthrough'
)

print("Aplicando transformaciones (fit_transform en train, transform en val)")
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

del X_train, X_val
gc.collect()

print("Entrenando LogisticRegression")
model = LogisticRegression(
    max_iter=1000, 
    n_jobs=-1, 
    random_state=42, 
    class_weight='balanced' 
)

start_time = time.time()
model.fit(X_train_processed, y_train)
training_time = time.time() - start_time

print("Evaluando modelo")
y_pred = model.predict(X_val_processed)
y_proba = model.predict_proba(X_val_processed)[:, 1]

print("\n--- MÉTRICAS KDD PARA GOOGLE SHEETS ---")
print(f"Modelo: Logistic Regression (Balanced)")
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(f"Recall (Attack): {recall_score(y_val, y_pred):.4f}")
print(f"F1 (Attack): {f1_score(y_val, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_proba):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_val, y_pred):.4f}")
print(f"Tiempo de entrenamiento: {training_time:.2f} s")
print("Notas: ColumnTransformer (OHE+StandardScaler), LogisticRegression(class_weight=balanced)")

print("\nGenerando predicciones preliminares")
manifest_val = pd.read_csv("01_DATA/KDD/validation.manifest.csv")
pd.DataFrame({
    'observation_id': manifest_val['observation_id'],
    'prediction': y_pred
}).to_csv("02_RESULTS/KDD/isaac/validation_predictions_logreg.csv", index=False)
print("Ejecución finalizada.")