
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

def mixup_data(X, y, alpha=0.4):

    X_synthetic = []
    y_synthetic = []

    unique_classes = np.unique(y)

    for cls in unique_classes:

        idx = np.where(y == cls)[0]

        if len(idx) < 2:
            continue

        X_cls = X[idx]

        # permutation interna alla classe
        perm = np.random.permutation(len(X_cls))

        for i in range(len(X_cls)):

            lam = np.random.beta(alpha, alpha)

            mixed = lam * X_cls[i] + (1 - lam) * X_cls[perm[i]]

            X_synthetic.append(mixed)
            y_synthetic.append(cls)

    X_synthetic = np.vstack(X_synthetic)
    y_synthetic = np.array(y_synthetic)

    # concat originale + sintetico
    X_final = np.vstack([X, X_synthetic])
    y_final = np.concatenate([y, y_synthetic])

    return X_final, y_final


def convert_and_impute(data: pd.DataFrame) -> pd.DataFrame:
    """
    Conversione di tutte le colonne possibili in float e imputazione dei valori NaN con la mediana.
    """
    def convert_to_numeric(column):
        if column.dtype in ['object', 'category']:
            # Se non contiene stringhe alfanumeriche, prova la conversione
            if not any(isinstance(val, str) and any(c.isalpha() for c in val) for val in column):
                return pd.to_numeric(column, errors='coerce')
        return column

    data = data.apply(convert_to_numeric)
    numeric_columns = data.select_dtypes(include='number')
    median_values = numeric_columns.median()
    data.fillna(median_values, inplace=True)
    return data

def encode_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applicazione dell'encoding alle variabili categoriali (AJCC Stage, Sex e Race).
    """
    if 'AJCC Stage' in data.columns:
        data['AJCC Stage'] = data['AJCC Stage'].map({
            'I': 1, 'II': 2, 'III': 3, 'NA': 0
        }).fillna(0)

    if 'Sex' in data.columns:
        data['Sex'] = data['Sex'].map({'Male': 0, 'Female': 1})
    """
    if 'Race' in data.columns:
        data['Race'] = data['Race'].map({
            'Unknown': 0, 'Caucasian': 1, 'Black': 2, 'Asian': 3,
            'Hispanic': 4, 'Black/Hispanic': 5, 'Caucasian/Hispanic': 6,
            'Other': 7
        })
    """
    return data

def cose_ai_dati(X, y):
    """
    Scaling + SMOTE + noise (finale).
    """

    # 1. Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. MIXUP
    X_mix, y_mix = mixup_data(X_scaled, y)

    # 3. SMOTE (prima del noise)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_mix, y_mix)
    
    # 4. Noise (dopo SMOTE)
    def aggiungi_noise(X, noise_level):
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise

    X_res_noisy = aggiungi_noise(X_res, noise_level=0.01)

    print(f"\nFeature originali: {X.shape[1]}")
    print(f"Dopo MixUp: {X_mix.shape[0]}")
    print(f"Dopo SMOTE: {X_res.shape[0]}")
    print(f"Noise aggiunto: sì")

    return X_res_noisy, y_res, scaler

def reshape_for_cnn(X):
    """
    Reshape in (n_campioni, n_feature, 1) per reti 1D o Siamese.
    """
    return X.reshape(X.shape[0], X.shape[1], 1)

def salva_e_stampa_smote(X_res, y_res, output_file="Data_SMOTE.xlsx"):
    df_resampled = pd.DataFrame(X_res)

    df_resampled["Class"] = y_res

    print("\n=== DATASET DOPO SMOTE ===")
    print(df_resampled.head())

    df_resampled.to_excel(output_file, index=False)
    print(f"\nFile salvato come: {output_file}")

    return df_resampled

def prepare_data_for_models (excel_path: str):
    # 1. Caricamento
    df = pd.read_excel(excel_path)
    # 2. Pulizia
    df.replace("?", pd.NA, inplace=True)
    df = df.drop(columns=['Patient ID #', 'Sample ID #', 'CancerSEEK Logistic Regression Score',
        'CancerSEEK Test Result', 'Class', 'Race'])

    df = convert_and_impute(df)
    df = encode_columns(df)

    if "Tumor type" in df.columns:
        df.rename(columns={"Tumor type": "CANCER_TYPE"}, inplace=True)
    if "AJCC Stage" in df.columns:
        df.rename(columns={"AJCC Stage": "AJCC_Stage"}, inplace=True)

    # Seleziono le feature numeriche
    feature_cols = [col for col in df.columns if col not in ["CANCER_TYPE", "AJCC_Stage"]]
    X = df[feature_cols].values
    y = df["CANCER_TYPE"].values

    # Suddivisione in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Passo pazzo in cui attuiamo le cose ai dati
    X_train_res, y_train_res, scaler = cose_ai_dati(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)


    # Split train/val
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_res, y_train_res, test_size=0.10, random_state=42
    )

    # Reshape per CNN/Siamese
    X_train_resh = reshape_for_cnn(X_train_final)
    X_val_resh = reshape_for_cnn(X_val)
    X_test_resh = reshape_for_cnn(X_test_scaled)

    # Encoding del target
    le = LabelEncoder()
    le.fit(df["CANCER_TYPE"])

    salva_e_stampa_smote(X_train_res, y_train_res, output_file="Data_SMOTE.xlsx")

    y_train_enc = le.transform(y_train_final)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    n_classes = len(le.classes_)
    y_train_ohe = tf.keras.utils.to_categorical(y_train_enc, num_classes=n_classes)
    y_val_ohe = tf.keras.utils.to_categorical(y_val_enc, num_classes=n_classes)
    y_test_ohe = tf.keras.utils.to_categorical(y_test_enc, num_classes=n_classes)

    # Creazione DataFrame per la rete Siamese
    x_train_df = pd.DataFrame(X_train_final)
    x_train_df["CANCER_TYPE"] = [le.inverse_transform([label])[0] for label in y_train_enc]

    x_val_df = pd.DataFrame(X_val)
    x_val_df["CANCER_TYPE"] = [le.inverse_transform([label])[0] for label in y_val_enc]

    x_test_df = pd.DataFrame(X_test_scaled)
    x_test_df["CANCER_TYPE"] = [le.inverse_transform([label])[0] for label in y_test_enc]

    # Aggiungo AJCC_Stage solo al test set per test one-shot
    if "AJCC_Stage" in df.columns:
        ajcc_stages = df["AJCC_Stage"].values
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

