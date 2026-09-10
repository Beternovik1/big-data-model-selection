import pandas as pd
import numpy as np
import gc
import time
import os
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Directorios de salida
os.makedirs("02_RESULTS/HIGGS/isaac", exist_ok=True)

print("Cargando datos")
df_train = pd.read_csv("01_DATA/HIGGS/train_1000000.csv", header=None, dtype=np.float32)
df_val = pd.read_csv("01_DATA/HIGGS/validation.csv", header=None, dtype=np.float32)

X_train = df_train.iloc[:, 1:].values
y_train = df_train.iloc[:, 0].astype(np.int8).values
X_val = df_val.iloc[:, 1:].values
y_val = df_val.iloc[:, 0].astype(np.int8).values

del df_train, df_val
gc.collect()

print("Aplicando StandardScaler")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

del X_train, X_val
gc.collect()

print("Entrenando XGBoost")
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    tree_method='hist',
    n_jobs=-1, 
    random_state=42
)

start_time = time.time()
model.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

print("Evaluando modelo")
y_pred = model.predict(X_val_scaled)

print("\n--- RESULTADOS ---")
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(f"Recall: {recall_score(y_val, y_pred):.4f}")
print(f"F1: {f1_score(y_val, y_pred):.4f}")
print(f"Tiempo: {training_time:.2f} s")

print("\nGenerando predicciones")
manifest_val = pd.read_csv("01_DATA/HIGGS/validation.manifest.csv")
pd.DataFrame({
    'observation_id': manifest_val['observation_id'],
    'prediction': y_pred
}).to_csv("02_RESULTS/HIGGS/isaac/validation_predictions_xgb.csv", index=False)
print("Completado.")