import os
import time
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
from ctgan import CTGAN


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
    Scaling + CTGAN augmentation.
    """

    # =========================
    # 1. Scaling
    # =========================
    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    # =========================
    # 2. DataFrame CTGAN
    # =========================
    feature_cols = [f"F{i}" for i in range(X.shape[1])]

    df_train = pd.DataFrame(X_scaled, columns=feature_cols)
    df_train["CANCER_TYPE"] = y

    print("\nDistribuzione originale:")
    print(Counter(y))

    print("\n===================================")
    print(" Inizializzazione CTGAN")
    print("===================================")

    ctgan = CTGAN(
        epochs=150,
        batch_size=64,
        pac=1,
        verbose=True
    )

    print("\nCTGAN creata correttamente.")

    # =========================
    # FIT CTGAN
    # =========================
    print("\n===================================")
    print(" Avvio training CTGAN...")
    print("===================================")

    start_time = time.time()

    ctgan.fit(
        df_train,
        discrete_columns=["CANCER_TYPE"]
    )

    end_time = time.time()

    print("\n===================================")
    print(" Training CTGAN completato!")
    print("===================================")

    print(f"\nTempo totale: {(end_time - start_time):.2f} secondi")

    # =========================
    # 4. Bilanciamento classi
    # =========================
    class_counts = Counter(y)

    max_class = max(class_counts.values())

    synthetic_samples = []

    for cancer_type, count in class_counts.items():

        needed = max_class - count

        if needed > 0:

            print(f"\nGenero {needed} campioni per: {cancer_type}")

            sampled = ctgan.sample(
                needed,
                condition_column="CANCER_TYPE",
                condition_value=cancer_type
            )

            synthetic_samples.append(sampled)

    # =========================
    # 5. Merge dati reali + sintetici
    # =========================
    if len(synthetic_samples) > 0:
        synthetic_df = pd.concat(synthetic_samples, ignore_index=True)

        final_df = pd.concat(
            [df_train, synthetic_df],
            ignore_index=True
        )
    else:
        final_df = df_train.copy()

    # =========================
    # 6. Shuffle
    # =========================
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # =========================
    # 7. Split X/y
    # =========================
    y_res = final_df["CANCER_TYPE"].values

    X_res = final_df.drop(columns=["CANCER_TYPE"]).values

    print("\nDistribuzione finale:")
    print(Counter(y_res))

    print(f"\nCampioni originali: {len(df_train)}")
    print(f"Campioni finali: {len(final_df)}")

    return X_res, y_res, scaler

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

def prepare_data_for_models ():
    # 1. Caricamento
    df = pd.read_excel("Data.xlsx")
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

    salva_e_stampa_smote(X_train_res, y_train_res, output_file="Data_CTGAN.xlsx")

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