"""
Experimento reproduzível: Árvore de Decisão x Random Forest x XGBoost
para detecção/classificação de tráfego de rede usando o dataset UNSW-NB15.
"""

from __future__ import annotations

import time
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def load_unsw_nb15() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Carrega a base UNSW-NB15 que já vem dividida em treino e teste.
    Certifique-se de que os arquivos estejam na pasta 'data/'.
    """
    train_path = DATA_DIR / "UNSW_NB15_training-set.csv"
    test_path = DATA_DIR / "UNSW_NB15_testing-set.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Coloque os arquivos UNSW_NB15_training-set.csv e UNSW_NB15_testing-set.csv na pasta {DATA_DIR}"
        )

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Separando o alvo (label = 0 normal, 1 ataque)
    y_train = df_train.pop("label")
    y_test = df_test.pop("label")

    # Removendo colunas que não ajudam ou causam vazamento de dados
    colunas_remover = ['id', 'attack_cat']
    df_train = df_train.drop(columns=colunas_remover, errors='ignore')
    df_test = df_test.drop(columns=colunas_remover, errors='ignore')

    return df_train, df_test, y_train, y_test

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])

def main() -> None:
    # 1. Carrega os dados já separados
    X_train, X_test, y_train, y_test = load_unsw_nb15()

    # 2. Definição dos Modelos (com configurações honestas para o experimento)
    models = {
        "Árvore de Decisão": DecisionTreeClassifier(
            max_depth=10, # Aumentei um pouco a profundidade para pegar mais detalhes
            criterion="gini",
            random_state=SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, # Reduzido para 100 para rodar mais rápido
            random_state=SEED,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=SEED,
            n_jobs=-1,
            eval_metric="logloss",
        ),
    }

    rows = []
    matrices = {}

    for name, estimator in models.items():
        # Pipeline aplica o pré-processamento (imputer + onehot) e depois o modelo
        pipe = Pipeline([
            ("prep", build_preprocessor(X_train)),
            ("model", estimator),
        ])

        # Medindo tempo de treino
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        # Medindo tempo de predição
        t1 = time.perf_counter()
        pred = pipe.predict(X_test)
        predict_time = time.perf_counter() - t1

        # Salvando as métricas
        rows.append({
            "modelo": name,
            "precision": precision_score(y_test, pred, average="binary", zero_division=0),
            "recall": recall_score(y_test, pred, average="binary", zero_division=0),
            "f1": f1_score(y_test, pred, average="binary", zero_division=0),
            "tempo_treino_s": train_time,
            "tempo_predicao_s": predict_time,
        })

        matrices[name] = confusion_matrix(y_test, pred).tolist()

        print(f"\n=== {name} ===")
        print(classification_report(y_test, pred, zero_division=0))
        print("Matriz de confusão:")
        print(confusion_matrix(y_test, pred))
        print(f"Treino: {train_time:.6f}s")
        print(f"Predição: {predict_time:.6f}s")

    # 3. Exportação dos Resultados
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "metrics.csv", index=False)

    with open(RESULTS_DIR / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(matrices, f, ensure_ascii=False, indent=2)

    print("\nResultados salvos com sucesso na pasta:", RESULTS_DIR)
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()
