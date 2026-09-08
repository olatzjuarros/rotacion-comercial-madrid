import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib

# Definimos las variables exactamente igual que en el hito 6
HISTORIA = ["antiguedad_negocio", "antiguedad_local", "rotaciones_previas", "antiguedad_censurada"]
GEOGRAFIA = ["n_locales_150m", "n_competidores_200m", "pct_vacantes_150m", "diversidad_150m"]
CATEGORICAS = ["desc_tipo_acceso_local", "id_division", "desc_distrito_local"]
COLS = HISTORIA + GEOGRAFIA + CATEGORICAS

def objective(trial, X_train, y_train, X_val, y_val):
    # Replicamos el procesador del hito 6
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1), CATEGORICAS)],
        remainder="passthrough")
    
    idx_cat = list(range(len(CATEGORICAS)))
    
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 400, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 150),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-3, 10.0, log=True),
        'categorical_features': idx_cat,
        'early_stopping': True,
        'validation_fraction': 0.15,
        'random_state': 42
    }
    
    modelo = Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(**param))
    ])
    
    modelo.fit(X_train, y_train)
    preds = modelo.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="datos")
    args = parser.parse_args()

    print("[1/4] Cargando datos y aplicando particiones del hito 6...")
    datos = pd.read_pickle(Path(args.dir) / "dataset_modelado.pkl")
    
    train = datos[datos["uso"] == "train"]
    val = datos[datos["uso"] == "val"]
    test = datos[datos["uso"] == "test"]
    
    X_train, y_train = train[COLS], train["target"]
    X_val, y_val = val[COLS], val["target"]
    X_test, y_test = test[COLS], test["target"]

    print(f"[2/4] Iniciando optimización con Optuna (30 iteraciones)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Reducimos el ruido en consola
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=30)

    print("\n==========================================================")
    print("RESULTADOS DE LA OPTIMIZACIÓN")
    print("==========================================================")
    print(f"Mejor AUC en Validación: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\n[3/4] Entrenando modelo final campeon...")
    best_params = study.best_params
    best_params['categorical_features'] = list(range(len(CATEGORICAS)))
    best_params['early_stopping'] = True
    best_params['validation_fraction'] = 0.15
    best_params['random_state'] = 42

    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1), CATEGORICAS)],
        remainder="passthrough")
        
    final_model = Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(**best_params))
    ])
    final_model.fit(X_train, y_train)

    print("[4/4] Evaluando en conjunto de TEST (Cohorte 2021)...")
    preds_test = final_model.predict_proba(X_test)[:, 1]
    final_auc_test = roc_auc_score(y_test, preds_test)
    
    print(f"\n---> AUC FINAL EN TEST: {final_auc_test:.4f} <---")
    
    joblib.dump(final_model, Path(args.dir) / 'modelo_optimo_optuna.joblib')
    print("Modelo guardado en datos/modelo_optimo_optuna.joblib")