import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from ctgan import CTGAN


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


def encode_columns(data: pd.DataFrame) -> pd.DataFrame:
    # Cerchiamo le colonne cliniche in modo flessibile per la codifica provvisoria
    for col in data.columns:
        if 'AJCC Stage' in str(col) or 'AJCC_Stage' in str(col) or col == 'Stage':
            data[col] = data[col].map({'I': 1, 'II': 2, 'III': 3, 'NA': 0}).fillna(0)
        if 'Sex' in str(col):
            data[col] = data[col].map({'Male': 0, 'Female': 1}).fillna(0)
    return data


def prepare_data_ctgan(excel_path="Data.xlsx", epochs=150):
    """
    Pipeline di Data Augmentation Generativa tramite CTGAN (Versione Purificata Anti-Leakage).
    Garantisce che l'addestramento e il test avvengano SOLO su biomarcatori molecolari puri.
    """
    # 1. Caricamento del dataset originario
    df = pd.read_excel(excel_path)
    df.replace("?", pd.NA, inplace=True)

    # ==============================================================================
    # 🛡️ LISTA NERA ASSOLUTA: IDENTIFICAZIONE ED ESCLUSIONE DI QUALSIASI DATA LEAKAGE
    # ==============================================================================
    lista_nera_clinica = [
        'Patient ID', 'Sample ID', 'Patient ID #', 'Sample ID #',
        'Tumor type', 'CANCER_TYPE', 'AJCC Stage', 'AJCC_Stage', 'Stage',
        'Age', 'Sex', 'Race', 'Class', 'CancerSEEK Test Result',
        'CancerSEEK Logistic Regression Score', 'Omega Score'
    ]

    # Isoliaramo i target primari prima della cancellazione
    colonna_target_tumore = None
    colonna_target_stadio = None

    for c in df.columns:
        if 'Tumor type' in str(c) or 'CANCER_TYPE' in str(c):
            colonna_target_tumore = c
        if 'AJCC Stage' in str(c) or 'AJCC_Stage' in str(c) or c == 'Stage':
            colonna_target_stadio = c

    if colonna_target_tumore is None:
        raise ValueError("[!] Errore: Impossibile trovare la colonna del tipo di tumore nell'Excel!")

    # Creiamo la lista delle SOLE feature proteiche biologiche pure
    feature_cols = [c for c in df.columns if
                    c not in lista_nera_clinica and not any(x in str(c) for x in ['Logistic', 'Regr'])]

    print(f"-> Pipeline CTGAN configurata con {len(feature_cols)} biomarcatori proteici puri.")
    # ==============================================================================

    # Rinominaiamo nel DataFrame principale per mantenere compatibilità con il resto del codice
    df = df.rename(columns={colonna_target_tumore: "CANCER_TYPE"})
    if colonna_target_stadio:
        df = df.rename(columns={colonna_target_stadio: "AJCC_Stage"})

    df = convert_and_impute(df)
    df = encode_columns(df)

    # Splitting 80/20 stratificato sul tumore prima della generazione sintetica
    df_train, df_test = train_test_split(df, test_size=0.20, random_state=42, stratify=df["CANCER_TYPE"])

    print("\n[CTGAN] Distribuzione originale del Train (Solo Proteine):")
    print(Counter(df_train["CANCER_TYPE"]))

    # Prepariamo la tabella da dare in pasto alla GAN (Feature Pure + Target)
    colonne_gan = feature_cols + ["CANCER_TYPE"]
    dati_per_gan = df_train[colonne_gan].copy()

    print(f"\n[CTGAN] Inizializzazione e Avvio FIT per {epochs} epoche...")
    ctgan = CTGAN(epochs=epochs)
    ctgan.fit(dati_per_gan, discrete_columns=["CANCER_TYPE"])
    print("[CTGAN] Training completato con successo!")

    # Processo di bilanciamento controllato
    target_count = dati_per_gan["CANCER_TYPE"].value_counts().max()
    dati_bilanciati = dati_per_gan.copy()

    for classe, count in dati_per_gan["CANCER_TYPE"].value_counts().items():
        if count < target_count:
            n_da_generare = target_count - count
            print(f"[CTGAN] Genero campioni sintetici filtrati per la classe: {classe} (Target: {n_da_generare})")

            campioni_validi_classe = []
            accumulo_campioni = 0

            while accumulo_campioni < n_da_generare:
                blocco_grezzo = ctgan.sample(
                    n=max(n_da_generare * 2, 300),
                    condition_column="CANCER_TYPE",
                    condition_value=classe
                )
                blocco_filtrato = blocco_grezzo[blocco_grezzo["CANCER_TYPE"] == classe]

                if not blocco_filtrato.empty:
                    campioni_validi_classe.append(blocco_filtrato)
                    accumulo_campioni += len(blocco_filtrato)

            df_classe_pronta = pd.concat(campioni_validi_classe, ignore_index=True).head(n_da_generare)
            dati_bilanciati = pd.concat([dati_bilanciati, df_classe_pronta], ignore_index=True)

    print("\n[CTGAN] Distribuzione finale bilanciata:")
    print(Counter(dati_bilanciati["CANCER_TYPE"]))

    # Estrazione delle matrici NUMERICHE pure (Niente ID, Niente Stadi)
    X_train_res = dati_bilanciati[feature_cols].values
    y_train_res = dati_bilanciati["CANCER_TYPE"].values

    X_test = df_test[feature_cols].values
    y_test = df_test["CANCER_TYPE"].values

    # Scaling MinMax post-generazione
    scaler = MinMaxScaler()
    X_train_res_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # Splitting finale 90/10 per validazione interna
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_res_scaled, y_train_res, test_size=0.10, random_state=42
    )

    le = LabelEncoder()
    le.fit(df["CANCER_TYPE"])

    dati_bilanciati.to_excel("Data_CTGAN.xlsx", index=False)
    print("\n[FILE] Dataset generato salvato come: Data_CTGAN.xlsx")

    y_train_enc = le.transform(y_train_final)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    n_classes = len(le.classes_)
    y_train_ohe = tf.keras.utils.to_categorical(y_train_enc, num_classes=n_classes)
    y_val_ohe = tf.keras.utils.to_categorical(y_val_enc, num_classes=n_classes)
    y_test_ohe = tf.keras.utils.to_categorical(y_test_enc, num_classes=n_classes)

    # Creazione dei DataFrame finali per i Notebook (Risolto l'allineamento degli indici)
    x_train_df = pd.DataFrame(X_train_final, columns=feature_cols)
    x_train_df["CANCER_TYPE"] = le.inverse_transform(y_train_enc)

    x_val_df = pd.DataFrame(X_val, columns=feature_cols)
    x_val_df["CANCER_TYPE"] = le.inverse_transform(y_val_enc)

    x_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    x_test_df["CANCER_TYPE"] = le.inverse_transform(y_test_enc)

    if "AJCC_Stage" in df.columns:
        x_test_df["AJCC_Stage"] = df_test["AJCC_Stage"].values

    genes_len = X_train_final.shape[1]

    return (
        X_train_final, X_val, X_test_scaled,  # Matrici Numpy pure allineate (Fase 1 e 2)
        y_train_enc, y_val_enc, y_test_enc,  # Label numerici
        x_test_df, le, n_classes, genes_len  # Interfacce umane per SHAP e metriche
    )