"""
Experimento reproduzível: Árvore de Decisão x Random Forest x XGBoost
para detecção/classificação de tráfego de rede.

IMPORTANTE:
- Este script é um esqueleto executável.
- Ele NÃO inventa resultados.
- Coloque a base pública escolhida em data/ e adapte LOAD_DATASET().
- Nunca use o conjunto de teste para escolher hiperparâmetros.
- Fixe a semente para permitir reprodução.

Uso:
    python experiment/experiment.py
"""

from __future__ import annotations

import time
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
from sklearn.model_selection import train_test_split
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


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    ADAPTE ESTA FUNÇÃO À BASE ESCOLHIDA PELO GRUPO.

    Exemplo esperado:
        df = pd.read_csv(DATA_DIR / "sua_base.csv")
        y = df.pop("label")
        return df, y

    Para NSL-KDD, será necessário mapear o arquivo e a coluna alvo
    de acordo com a versão baixada e documentar essa decisão no README.
    """
    path = DATA_DIR / "dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Coloque a base pública em {path} e adapte load_dataset()."
        )

    df = pd.read_csv(path)
    target = "label"
    if target not in df.columns:
        raise ValueError(
            f"A coluna alvo '{target}' não existe. Ajuste TARGET/load_dataset()."
        )

    y = df.pop(target)
    return df, y


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
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )

    # O ajuste de hiperparâmetros deve ocorrer somente em treino/validação.
    # Os valores abaixo são um ponto de partida comparável, não um "melhor ajuste".
    models = {
        "Árvore de Decisão": DecisionTreeClassifier(
            max_depth=5,
            criterion="gini",
            random_state=SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=SEED,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            eval_metric="logloss",
        ),
    }

    rows = []
    matrices = {}

    for name, estimator in models.items():
        pipe = Pipeline([
            ("prep", build_preprocessor(X_train)),
            ("model", estimator),
        ])

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        pred = pipe.predict(X_test)
        predict_time = time.perf_counter() - t1

        rows.append({
            "modelo": name,
            "precision": precision_score(
                y_test, pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_test, pred, average="weighted", zero_division=0
            ),
            "f1": f1_score(
                y_test, pred, average="weighted", zero_division=0
            ),
            "tempo_treino_s": train_time,
            "tempo_predicao_s": predict_time,
        })

        matrices[name] = confusion_matrix(y_test, pred).tolist()

        print("\n===", name, "===")
        print(classification_report(y_test, pred, zero_division=0))
        print("Matriz de confusão:")
        print(confusion_matrix(y_test, pred))
        print(f"Treino: {train_time:.6f}s")
        print(f"Predição: {predict_time:.6f}s")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "metrics.csv", index=False)

    # As matrizes são salvas separadamente para facilitar a publicação.
    import json
    with open(RESULTS_DIR / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(matrices, f, ensure_ascii=False, indent=2)

    print("\nResultados salvos em:", RESULTS_DIR)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
