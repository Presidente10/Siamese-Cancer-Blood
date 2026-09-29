# preprocessing.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch


def read_and_clean_data(excel_path: str) -> pd.DataFrame:
    """Lettura dei dati da un file Excel e rimozione delle colonne non necessarie."""
    data = pd.read_excel(excel_path)

    # Rimozione simboli indesiderati dalle stringhe (es. gli asterischi)
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.replace('*', '', regex=False)
            data[col] = data[col].str.replace('**', '', regex=False)

    # Rimozione delle colonne di tracciamento o non utili alla classificazione
    cols_to_remove = [
        'Patient ID #', 'Sample ID #', 'CancerSEEK Logistic Regression Score',
        'CancerSEEK Test Result', 'Class', 'Race'
    ]
    data.drop(columns=cols_to_remove, inplace=True, errors='ignore')
    return data


def encode_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Applicazione dell'encoding alle variabili categoriali."""
    if 'AJCC Stage' in data.columns:
        data['AJCC Stage'] = data['AJCC Stage'].map({
            'I': 1, 'II': 2, 'III': 3, 'NA': 0
        }).fillna(0)

    if 'Sex' in data.columns:
        data['Sex'] = data['Sex'].map({'Male': 0, 'Female': 1})
    return data


def convert_and_impute(data: pd.DataFrame) -> pd.DataFrame:
    """Conversione in float e imputazione dei valori mancanti con la mediana."""

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


def standardize_and_smote(X, y):
    """Normalizzazione MinMaxScaler e bilanciamento delle classi con SMOTE."""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # SMOTE risolve lo sbilanciamento creando campioni sintetici per i tumori rari
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)
    return X_res, y_res, scaler


def reshape_for_cnn(X):
    """Reshape in (n_campioni, n_feature, 1) necessario per la logica del main."""
    return X.reshape(X.shape[0], X.shape[1], 1)


def prepare_data_for_models(excel_path: str):
    """Pipeline completa di preparazione dati."""
    # 1. Caricamento e pulizia
    data = read_and_clean_data(excel_path)
    data = convert_and_impute(data)
    data = encode_columns(data)

    # Uniformiamo i nomi delle colonne come richiesto dal codice
    if "Tumor type" in data.columns:
        data.rename(columns={"Tumor type": "CANCER_TYPE"}, inplace=True)
    if "AJCC Stage" in data.columns:
        data.rename(columns={"AJCC Stage": "AJCC_Stage"}, inplace=True)

    # Separazione feature (X) e target (y)
    feature_cols = [col for col in data.columns if col not in ["CANCER_TYPE", "AJCC_Stage"]]
    X = data[feature_cols].values
    y = data["CANCER_TYPE"].values

    # 2. Split Train/Test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Bilanciamento delle classi solo sul Train set
    X_train_res, y_train_res, scaler = standardize_and_smote(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Split Train/Validation (90% / 10%) dai dati bilanciati
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_res, y_train_res, test_size=0.10, random_state=42
    )

    # 5. Reshape e Label Encoding delle stringhe in ID numerici
    X_train_resh = reshape_for_cnn(X_train_final)
    X_val_resh = reshape_for_cnn(X_val)
    X_test_resh = reshape_for_cnn(X_test_scaled)

    le = LabelEncoder()
    le.fit(data["CANCER_TYPE"])
    y_train_enc = le.transform(y_train_final)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    n_classes = len(le.classes_)

    # One-Hot Encoding finto/inutile per PyTorch (ma mantenuto per compatibilità di output delle funzioni)
    y_train_ohe = np.eye(n_classes)[y_train_enc]
    y_val_ohe = np.eye(n_classes)[y_val_enc]
    y_test_ohe = np.eye(n_classes)[y_test_enc]

    # 6. Creazione dei DataFrame specchio usati nei calcoli K-Shot
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