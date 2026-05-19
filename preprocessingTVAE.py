import os
import time
import logging
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
 
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from ctgan import TVAE
 
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Pulizia / Encoding
# ---------------------------------------------------------------------------
 
def convert_and_impute(data: pd.DataFrame) -> pd.DataFrame:
    """
    Converti le colonne numericamente convertibili in float
    e imputa i NaN con la mediana della colonna.
    """
    def _try_numeric(col: pd.Series) -> pd.Series:
        if col.dtype in ("object", "category"):
            has_alpha = col.dropna().astype(str).str.contains(r"[A-Za-z]", regex=True).any()
            if not has_alpha:
                return pd.to_numeric(col, errors="coerce")
        return col
 
    data = data.apply(_try_numeric)
    medians = data.select_dtypes(include="number").median()
    return data.fillna(medians)          # niente inplace — compatibile con pandas >= 2.x
 
 
def encode_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding ordinale per AJCC Stage e binario per Sex.
    """
    if "AJCC Stage" in data.columns:
        stage_map = {"I": 1, "II": 2, "III": 3, "NA": 0}
        data["AJCC Stage"] = data["AJCC Stage"].map(stage_map).fillna(0)
 
    if "Sex" in data.columns:
        data["Sex"] = data["Sex"].map({"Male": 0, "Female": 1})
 
    return data
 
 
# ---------------------------------------------------------------------------
# Augmentazione con TVAE — versione migliorata
# ---------------------------------------------------------------------------
 
def _fit_tvae_per_class(
    X_class: np.ndarray,
    feature_cols: list[str],
    epochs: int,
    batch_size: int,
) -> TVAE:
    """
    Addestra un'istanza TVAE su un singolo sottoinsieme di classe.
    Restituisce il modello addestrato.
    """
    df_class = pd.DataFrame(X_class, columns=feature_cols)
    tvae = TVAE(epochs=epochs, batch_size=batch_size, verbose=False)
    tvae.fit(df_class, discrete_columns=[])   # tutte continue
    return tvae
 
 
def _generate_synthetic(
    tvae: TVAE,
    n_samples: int,
    feature_cols: list[str],
    clip: bool = True,
) -> np.ndarray:
    """
    Genera `n_samples` campioni sintetici dal TVAE fornito.
    Se `clip=True` i valori vengono riportati in [0, 1]
    (coerente con MinMaxScaler applicato a monte).
    """
    synthetic = tvae.sample(n_samples)
    # Manteniamo solo le colonne feature nell'ordine corretto
    synthetic = synthetic.reindex(columns=feature_cols)
    arr = synthetic.values.astype(np.float32)
    if clip:
        arr = np.clip(arr, 0.0, 1.0)
    return arr
 
 
def tvae_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 150,
    batch_size: int = 64,
    strategy: str = "balance",          # "balance" | "fixed"
    target_per_class: int | None = None, # usato solo con strategy="fixed"
    clip_synthetic: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augmentazione tabellare condizionata alla classe tramite TVAE.
 
    Un modello TVAE separato viene addestrato per ogni classe, eliminando
    il problema di TVAE.sample() che non rispetta la label assegnata
    in post-hoc.
 
    Parametri
    ----------
    X               : feature matrix (già scalata, float)
    y               : vettore delle etichette (string o int)
    epochs          : epoche di training TVAE per classe
    batch_size      : batch size TVAE
    strategy        : "balance"  → porta ogni classe al conteggio della classe maggioritaria
                      "fixed"    → porta ogni classe a `target_per_class`
    target_per_class: valore target per strategy="fixed"
    clip_synthetic  : se True, clippa i valori sintetici in [0, 1]
    random_state    : seed per la shuffle finale
 
    Ritorna
    -------
    X_aug, y_aug  : array aumentati e mescolati
    """
    class_counts = Counter(y)
    classes = sorted(class_counts.keys())
    feature_cols = [f"F{i}" for i in range(X.shape[1])]
 
    if strategy == "balance":
        target = max(class_counts.values())
    elif strategy == "fixed":
        if target_per_class is None:
            raise ValueError("target_per_class richiesto con strategy='fixed'")
        target = target_per_class
    else:
        raise ValueError(f"strategy non riconosciuta: {strategy!r}")
 
    log.info("Distribuzione originale: %s", dict(class_counts))
    log.info("Target campioni per classe: %d  (strategy=%s)", target, strategy)
 
    X_parts = [X]
    y_parts = [y]
 
    total_start = time.time()
 
    for cls in classes:
        mask = y == cls
        X_cls = X[mask].astype(np.float32)
        current_count = len(X_cls)
        needed = target - current_count
 
        if needed <= 0:
            log.info("Classe '%s': %d campioni → nessuna generazione necessaria.", cls, current_count)
            continue
 
        log.info("Classe '%s': %d campioni → genero %d campioni sintetici …", cls, current_count, needed)
 
        # Verifica che ci siano abbastanza campioni per addestrare TVAE
        if current_count < 10:
            log.warning(
                "Classe '%s' ha solo %d campioni — TVAE potrebbe produrre output di bassa qualità.",
                cls, current_count,
            )
 
        t0 = time.time()
        try:
            tvae = _fit_tvae_per_class(X_cls, feature_cols, epochs=epochs, batch_size=min(batch_size, current_count))
            X_syn = _generate_synthetic(tvae, needed, feature_cols, clip=clip_synthetic)
        except Exception as exc:
            log.error("TVAE fallita per classe '%s': %s — la classe viene saltata.", cls, exc)
            continue
 
        y_syn = np.array([cls] * needed, dtype=y.dtype)
        X_parts.append(X_syn)
        y_parts.append(y_syn)
 
        log.info("  → generazione completata in %.1f s", time.time() - t0)
 
    X_aug = np.vstack(X_parts)
    y_aug = np.concatenate(y_parts)
 
    # Shuffle riproducibile
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(X_aug))
    X_aug, y_aug = X_aug[idx], y_aug[idx]
 
    log.info("Tempo totale augmentazione: %.1f s", time.time() - total_start)
    log.info("Distribuzione finale:       %s", dict(Counter(y_aug)))
    log.info("Shape X finale:             %s", X_aug.shape)
 
    return X_aug, y_aug
 
 
# ---------------------------------------------------------------------------
# Funzione di pipeline unificata (rimpiazza cose_ai_dati)
# ---------------------------------------------------------------------------
 
def scale_and_augment(
    X: np.ndarray,
    y: np.ndarray,
    tvae_epochs: int = 150,
    tvae_batch_size: int = 64,
    augmentation_strategy: str = "balance",
    target_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Scala X con MinMaxScaler e applica l'augmentazione TVAE per classe.
 
    Ritorna
    -------
    X_res    : feature augmentate (scalate)
    y_res    : etichette augmentate
    scaler   : scaler fittato sui dati originali (da usare su test set)
    """
    log.info("=== Scaling MinMax ===")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
 
    log.info("=== Augmentazione TVAE per classe ===")
    X_res, y_res = tvae_augmentation(
        X_scaled, y,
        epochs=tvae_epochs,
        batch_size=tvae_batch_size,
        strategy=augmentation_strategy,
        target_per_class=target_per_class,
    )
 
    return X_res, y_res, scaler
 
 
# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
 
def reshape_for_cnn(X: np.ndarray) -> np.ndarray:
    """Reshape in (n, features, 1) per CNN 1-D."""
    return X.reshape(X.shape[0], X.shape[1], 1)
 
 
def salva_e_stampa(
    X_res: np.ndarray,
    y_res: np.ndarray,
    output_file: str = "Data_TVAE.xlsx",
) -> pd.DataFrame:
    df = pd.DataFrame(X_res)
    df["Class"] = y_res
    log.info("Dataset aumentato — prime righe:\n%s", df.head())
    df.to_excel(output_file, index=False)
    log.info("Salvato: %s", output_file)
    return df
 
 
# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------
 
def prepare_data_for_models():
    # 1. Caricamento
    df = pd.read_excel("Data.xlsx")
 
    # 2. Pulizia
    df.replace("?", pd.NA, inplace=True)
    df = df.drop(columns=[
        "Patient ID #", "Sample ID #",
        "CancerSEEK Logistic Regression Score",
        "CancerSEEK Test Result", "Class", "Race",
    ])
    df = convert_and_impute(df)
    df = encode_columns(df)
 
    df.rename(columns={"Tumor type": "CANCER_TYPE", "AJCC Stage": "AJCC_Stage"}, inplace=True)
 
    # 3. Split feature / target
    feature_cols = [c for c in df.columns if c not in ("CANCER_TYPE", "AJCC_Stage")]
    X = df[feature_cols].values
    y = df["CANCER_TYPE"].values
 
    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
 
    # 5. Scaling + augmentazione TVAE (solo sul train)
    X_train_res, y_train_res, scaler = scale_and_augment(X_train, y_train)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
 
    # 6. Split train / validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_res, y_train_res, test_size=0.10, random_state=42
    )
 
    # 7. Reshape per CNN
    X_train_resh = reshape_for_cnn(X_train_final)
    X_val_resh   = reshape_for_cnn(X_val)
    X_test_resh  = reshape_for_cnn(X_test_scaled)
 
    # 8. Label encoding + OHE
    le = LabelEncoder()
    le.fit(df["CANCER_TYPE"])
 
    salva_e_stampa(X_train_res, y_train_res, output_file="Data_TVAE.xlsx")
 
    y_train_enc = le.transform(y_train_final)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)
 
    n_classes = len(le.classes_)
    y_train_ohe = tf.keras.utils.to_categorical(y_train_enc, num_classes=n_classes)
    y_val_ohe   = tf.keras.utils.to_categorical(y_val_enc,   num_classes=n_classes)
    y_test_ohe  = tf.keras.utils.to_categorical(y_test_enc,  num_classes=n_classes)
 
    # 9. DataFrame per rete Siamese
    def _make_df(X_flat, enc_labels):
        d = pd.DataFrame(X_flat)
        d["CANCER_TYPE"] = le.inverse_transform(enc_labels)
        return d
 
    x_train_df = _make_df(X_train_final, y_train_enc)
    x_val_df   = _make_df(X_val,         y_val_enc)
    x_test_df  = _make_df(X_test_scaled, y_test_enc)
 
    # AJCC_Stage nel test set (per one-shot testing)
    if "AJCC_Stage" in df.columns:
        _, ajcc_test = train_test_split(
            df["AJCC_Stage"].values, test_size=0.20, random_state=42, stratify=y
        )
        x_test_df["AJCC_Stage"] = ajcc_test
 
    genes_len = X_train_resh.shape[1]
 
    return (
        x_train_df, x_val_df, x_test_df,
        y_test, y_train_enc, y_val_enc, y_test_enc,
        X_train, X_val, X_test,
        X_train_resh, X_val_resh, X_test_resh,
        y_train_ohe, y_val_ohe, y_test_ohe,
        le, n_classes, genes_len,
    )