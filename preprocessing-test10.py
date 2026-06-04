import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# LETTURA E PULIZIA
# ---------------------------------------------------------------------------

def read_and_clean_data(excel_path: str) -> pd.DataFrame:
    data = generate_mixup_excel(excel_path, n_samples_per_group=1, alpha=0.4, random_state=42)

    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.replace('*', '', regex=False)
            data[col] = data[col].str.replace('**', '', regex=False)

    cols_to_remove = [
        'Patient ID #', 'Sample ID #', 'CancerSEEK Logistic Regression Score',
        'CancerSEEK Test Result', 'Class', 'Race'
    ]
    data.drop(columns=cols_to_remove, inplace=True, errors='ignore')
    return data


# ---------------------------------------------------------------------------
# ENCODING
# ---------------------------------------------------------------------------

def encode_columns(data: pd.DataFrame) -> pd.DataFrame:
    if 'AJCC Stage' in data.columns:
        data['AJCC Stage'] = data['AJCC Stage'].map({
            'I': 1, 'II': 2, 'III': 3, 'NA': 0
        }).fillna(0)

    if 'Sex' in data.columns:
        data['Sex'] = data['Sex'].map({'Male': 0, 'Female': 1})

    return data


# ---------------------------------------------------------------------------
# CONVERSIONE E IMPUTAZIONE
# ---------------------------------------------------------------------------

def convert_and_impute(data: pd.DataFrame) -> pd.DataFrame:
    def convert_to_numeric(column):
        if column.dtype in ['object', 'category']:
            if not any(isinstance(val, str) and any(c.isalpha() for c in val) for val in column):
                return pd.to_numeric(column, errors='coerce')
        return column

    data = data.apply(convert_to_numeric)
    numeric_columns = data.select_dtypes(include='number')
    median_values = numeric_columns.median()
    data.fillna(median_values, inplace=True)
    return data


# ---------------------------------------------------------------------------
# SMOTE + SCALING
# ---------------------------------------------------------------------------

def standardize_and_smote(X, y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)

    def aggiungi_noise(X, noise_level):
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise

    X_res_noisy = aggiungi_noise(X_res, noise_level=0.01)


    return X_res_noisy, y_res, scaler


# ---------------------------------------------------------------------------
# RESHAPE PER CNN / SIAMESE
# ---------------------------------------------------------------------------

def reshape_for_cnn(X):
    return X.reshape(X.shape[0], X.shape[1], 1)


# ---------------------------------------------------------------------------
# MIXUP TABULARE
# ---------------------------------------------------------------------------

def tabular_mixup(
    df: pd.DataFrame,
    group_cols: list,
    numeric_cols: list,
    categorical_cols: list,
    n_samples_per_group: int = 1,
    alpha: float = 0.4,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Genera campioni sintetici tramite MixUp interpolando coppie di righe
    che appartengono allo stesso gruppo (stessa combinazione di group_cols).
    Solo le colonne numeriche vengono interpolate; le categoriali restano
    identiche a quelle del primo campione scelto (coerenza di gruppo garantita).
    """
    np.random.seed(random_state)
    synthetic_rows = []

    for _, group_df in df.groupby(group_cols, sort=False):
        if len(group_df) < 2:
            continue

        group_df = group_df.reset_index(drop=True)

        for _ in range(n_samples_per_group * len(group_df)):
            i, j = np.random.choice(len(group_df), size=2, replace=False)
            row_i = group_df.iloc[i]
            row_j = group_df.iloc[j]

            lam = np.random.beta(alpha, alpha)

            synthetic = {}
            for col in numeric_cols:
                synthetic[col] = lam * row_i[col] + (1 - lam) * row_j[col]
            for col in categorical_cols:
                synthetic[col] = row_i[col]

            synthetic_rows.append(synthetic)

    return pd.DataFrame(synthetic_rows)


def generate_mixup_excel(
    excel_path: str,
    n_samples_per_group: int = 1,
    alpha: float = 0.4,
    random_state: int = 42
) -> str:
    """
    Legge il dataset grezzo (pre-preprocessing), applica il MixUp raggruppando
    per (Tumor type, AJCC Stage, CancerSEEK Test Result), concatena i campioni
    sintetici agli originali e salva il risultato in un file Excel.

    Deve essere chiamata PRIMA di prepare_data_for_models, passando poi
    output_path come excel_path alla pipeline di addestramento.

    Returns:
        output_path: percorso del file Excel generato.
    """
    df = pd.read_excel(excel_path)

    # Pulizia minimale dei simboli spurii
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('*', '', regex=False)

    # Colonne di raggruppamento specifiche del dataset CancerSEEK
    group_candidates = ["Tumor type", "AJCC Stage", "CancerSEEK Test Result"]
    group_cols = [c for c in group_candidates if c in df.columns]

    if not group_cols:
        raise ValueError(
            "Nessuna delle colonne di raggruppamento trovata nel file. "
            f"Attese: {group_candidates}"
        )

    # Colonne escluse dalla miscelazione numerica
    id_like = {'Patient ID #', 'Sample ID #', 'Class', 'Race',
               'CancerSEEK Logistic Regression Score'}
    exclude = id_like | set(group_cols)

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude and 'id' not in c.lower()
    ]

    # Tutto il resto è categoriale (mantenuto identico per coerenza)
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    df_mix = tabular_mixup(
        df=df,
        group_cols=group_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        n_samples_per_group=n_samples_per_group,
        alpha=alpha,
        random_state=random_state,
    )

    df_final = pd.concat([df, df_mix], ignore_index=True)
    return df_final


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPALE
# ---------------------------------------------------------------------------

def prepare_data_for_models(excel_path: str):
    """
    Esegue l'intera pipeline: caricamento, pulizia, encoding, split,
    standardizzazione, SMOTE, reshape e encoding del target.

    Per usare il MixUp, chiama prima generate_mixup_excel() e passa
    l'output_path risultante come excel_path a questa funzione.
    """
    
    data = read_and_clean_data(excel_path)
    data = convert_and_impute(data)
    data = encode_columns(data)

    if "Tumor type" in data.columns:
        data.rename(columns={"Tumor type": "CANCER_TYPE"}, inplace=True)
    if "AJCC Stage" in data.columns:
        data.rename(columns={"AJCC Stage": "AJCC_Stage"}, inplace=True)

    feature_cols = [col for col in data.columns if col not in ["CANCER_TYPE", "AJCC_Stage"]]
    X = data[feature_cols].values
    y = data["CANCER_TYPE"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train_res, y_train_res, scaler = standardize_and_smote(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_res, y_train_res, test_size=0.10, random_state=42
    )

    X_train_resh = reshape_for_cnn(X_train_final)
    X_val_resh   = reshape_for_cnn(X_val)
    X_test_resh  = reshape_for_cnn(X_test_scaled)

    le = LabelEncoder()
    le.fit(data["CANCER_TYPE"])
    y_train_enc = le.transform(y_train_final)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    n_classes = len(le.classes_)
    y_train_ohe = tf.keras.utils.to_categorical(y_train_enc, num_classes=n_classes)
    y_val_ohe   = tf.keras.utils.to_categorical(y_val_enc,   num_classes=n_classes)
    y_test_ohe  = tf.keras.utils.to_categorical(y_test_enc,  num_classes=n_classes)

    x_train_df = pd.DataFrame(X_train_final)
    x_train_df["CANCER_TYPE"] = le.inverse_transform(y_train_enc)

    x_val_df = pd.DataFrame(X_val)
    x_val_df["CANCER_TYPE"] = le.inverse_transform(y_val_enc)

    x_test_df = pd.DataFrame(X_test_scaled)
    x_test_df["CANCER_TYPE"] = le.inverse_transform(y_test_enc)

    if "AJCC_Stage" in data.columns:
        ajcc_stages = data["AJCC_Stage"].values
        _, ajcc_test = train_test_split(
            ajcc_stages, test_size=0.20, random_state=42, stratify=y
        )
        x_test_df["AJCC_Stage"] = ajcc_test

    genes_len = X_train_resh.shape[1]

    return (
        x_train_df, x_val_df, x_test_df,
        y_test, y_train_enc, y_val_enc, y_test_enc,
        X_train, X_val, X_test,
        X_train_resh, X_val_resh, X_test_resh,
        y_train_ohe, y_val_ohe, y_test_ohe,
        le, n_classes, genes_len
    )
