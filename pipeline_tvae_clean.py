# pipeline_tvae_clean.py
import time
import logging
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from ctgan import TVAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def convert_and_impute(data: pd.DataFrame) -> pd.DataFrame:
    def _try_numeric(col: pd.Series) -> pd.Series:
        if col.dtype in ("object", "category"):
            has_alpha = col.dropna().astype(str).str.contains(r"[A-Za-z]", regex=True).any()
            if not has_alpha:
                return pd.to_numeric(col, errors="coerce")
        return col

    data = data.apply(_try_numeric)
    medians = data.select_dtypes(include="number").median()
    return data.fillna(medians)


def encode_columns(data: pd.DataFrame) -> pd.DataFrame:
    for col in data.columns:
        if 'AJCC Stage' in str(col) or 'AJCC_Stage' in str(col) or col == 'Stage':
            data[col] = data[col].map({'I': 1, 'II': 2, 'III': 3, 'NA': 0}).fillna(0)
        if 'Sex' in str(col):
            data[col] = data[col].map({'Male': 0, 'Female': 1}).fillna(0)
    return data


def _fit_tvae_per_class(X_class: np.ndarray, feature_cols: list[str], epochs: int, batch_size: int) -> TVAE:
    df_class = pd.DataFrame(X_class, columns=feature_cols)
    tvae = TVAE(epochs=epochs, batch_size=batch_size, verbose=False)
    tvae.fit(df_class, discrete_columns=[])
    return tvae


def _generate_synthetic(tvae: TVAE, n_samples: int, feature_cols: list[str], clip: bool = True) -> np.ndarray:
    synthetic = tvae.sample(n_samples)
    synthetic = synthetic.reindex(columns=feature_cols)
    arr = synthetic.values.astype(np.float32)
    if clip:
        arr = np.clip(arr, 0.0, 1.0)
    return arr


def tvae_augmentation(X: np.ndarray, y: np.ndarray, feature_names: list, epochs: int = 150, batch_size: int = 64) -> \
tuple[np.ndarray, np.ndarray]:
    class_counts = Counter(y)
    classes = sorted(class_counts.keys())
    target = max(class_counts.values())

    log.info("[TVAEAUG] Distribuzione originale del Train Puro: %s", dict(class_counts))
    X_parts, y_parts = [X], [y]
    total_start = time.time()

    for cls in classes:
        mask = y == cls
        X_cls = X[mask].astype(np.float32)
        current_count = len(X_cls)
        needed = target - current_count

        if needed <= 0:
            continue

        log.info("[TVAE] Classe '%s': genero %d campioni sintetici...", cls, needed)
        try:
            tvae = _fit_tvae_per_class(X_cls, feature_names, epochs=epochs, batch_size=min(batch_size, current_count))
            X_syn = _generate_synthetic(tvae, needed, feature_names, clip=True)
        except Exception as exc:
            log.error("[TVAE] Fallita per classe '%s': %s", cls, exc)
            continue

        y_syn = np.array([cls] * needed, dtype=y.dtype)
        X_parts.append(X_syn)
        y_parts.append(y_syn)

    X_aug = np.vstack(X_parts)
    y_aug = np.concatenate(y_parts)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_aug))

    log.info("[TVAE] Tempo totale generazione: %.1f s", time.time() - total_start)
    return X_aug[idx], y_aug[idx]


def prepare_data_tvae(excel_path="Data.xlsx", epochs=150):
    """Pipeline TVAE protetta anti-leakage in PyTorch Puro ed allineata."""
    df = pd.read_excel(excel_path)
    df.replace("?", pd.NA, inplace=True)

    # 🛡️ ESCLUSIONE TOTALE DATA LEAKAGE CLINICO ED ANAGRAFICO
    lista_nera_clinica = [
        'Patient ID', 'Sample ID', 'Patient ID #', 'Sample ID #',
        'Tumor type', 'CANCER_TYPE', 'AJCC Stage', 'AJCC_Stage', 'Stage',
        'Age', 'Sex', 'Race', 'Class', 'CancerSEEK Test Result',
        'CancerSEEK Logistic Regression Score', 'Omega Score'
    ]

    colonna_target_tumore = None
    colonna_target_stadio = None

    for c in df.columns:
        if 'Tumor type' in str(c) or 'CANCER_TYPE' in str(c):
            colonna_target_tumore = c
        if 'AJCC Stage' in str(c) or 'AJCC_Stage' in str(c) or c == 'Stage':
            colonna_target_stadio = c

    if colonna_target_tumore is None:
        raise ValueError("[!] Errore: Colonna del tipo di tumore mancante nell'Excel!")

    # Estrarre solo i biomarcatori molecolari reali
    feature_cols = [c for c in df.columns if
                    c not in lista_nera_clinica and not any(x in str(c) for x in ['Logistic', 'Regr'])]
    print(f"-> Pipeline TVAE configurata con {len(feature_cols)} biomarcatori proteici puri.")

    df = df.rename(columns={colonna_target_tumore: "CANCER_TYPE"})
    if colonna_target_stadio:
        df = df.rename(columns={colonna_target_stadio: "AJCC_Stage"})

    df = convert_and_impute(df)
    df = encode_columns(df)

    X = df[feature_cols].values
    y = df["CANCER_TYPE"].values

    # 1. ISOLAMENTO TEST SET (20%) REALE E PURO
    X_train_full, X_test_scaled, y_train_full, y_test_enc = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 2. ISOLAMENTO VALIDATION SET (10% del rimanente) REALE E PURO
    X_train_puro, X_val_puro, y_train_puro, y_val_enc = train_test_split(
        X_train_full, y_train_full, test_size=0.10, random_state=42, stratify=y_train_full
    )

    # 3. SCALING BASATO ESCLUSIVAMENTE SUL TRAIN PURO
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_puro).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_puro).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_scaled).astype(np.float32)

    # 4. AUGMENTAZIONE PROBABILISTICA TVAE (Solo ed esclusivamente sul Train Scaled Puro)
    X_train_final, y_train_aug = tvae_augmentation(X_train_scaled, y_train_puro, feature_cols, epochs=epochs)

    le = LabelEncoder()
    le.fit(df["CANCER_TYPE"])

    y_train_enc = le.transform(y_train_aug)
    y_val_enc = le.transform(y_val_enc)
    y_test_enc = le.transform(y_test_enc)

    # 5. COSTRUZIONE DATAFRAME DI TEST MAPPATO CON I NOMI PROTEICI REALI
    x_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    x_test_df["CANCER_TYPE"] = le.inverse_transform(y_test_enc)

    if "AJCC_Stage" in df.columns:
        _, ajcc_test = train_test_split(df["AJCC_Stage"].values, test_size=0.20, random_state=42, stratify=y)
        x_test_df["AJCC_Stage"] = ajcc_test

    n_classes = len(le.classes_)
    genes_len = X_train_final.shape[1]

    # Esportazione del dataset bilanciato per tracciabilità
    df_export = pd.DataFrame(X_train_final, columns=feature_cols)
    df_export["CANCER_TYPE"] = le.inverse_transform(y_train_enc)
    df_export.to_excel("Data_TVAE.xlsx", index=False)
    print(f"\n[FILE] Dataset probabilistico salvato come: Data_TVAE.xlsx")

    return (
        X_train_final, X_val_scaled, X_test_scaled,
        y_train_enc, y_val_enc, y_test_enc,
        x_test_df, le, n_classes, genes_len
    )