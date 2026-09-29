# pipeline_filtrato_clean.py
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def tabular_mixup(df, group_cols, numeric_cols, categorical_cols):
    """Genera dati sintetici applicando il Mixup solo all'interno dello stesso gruppo clinico."""
    np.random.seed(42)
    synthetic_rows = []
    for _, group_df in df.groupby(group_cols, sort=False):
        if len(group_df) < 2:
            continue
        group_df = group_df.reset_index(drop=True)
        for _ in range(len(group_df)):
            i, j = np.random.choice(len(group_df), size=2, replace=False)
            lam = np.random.beta(0.4, 0.4)
            synthetic = {}
            for col in numeric_cols:
                synthetic[col] = lam * group_df.iloc[i][col] + (1 - lam) * group_df.iloc[j][col]
            for col in categorical_cols:
                synthetic[col] = group_df.iloc[i][col]
            synthetic_rows.append(synthetic)
    return pd.DataFrame(synthetic_rows)


def prepare_data_filtrato(excel_path: str):
    """Pipeline Filtrata Protetta Anti-Leakage con 39 biomarcatori molecolari puri."""
    data = pd.read_excel(excel_path)

    # 1. 🛡️ PULIZIA IMMEDIATA STRINGHE E CARATTERI SCONOSCIUTI (Anti-Crash)
    data.replace("?", pd.NA, inplace=True)

    # Standardizzazione nomi colonne target
    if 'Tumor type' in data.columns: data.rename(columns={'Tumor type': 'CANCER_TYPE'}, inplace=True)
    if 'AJCC Stage' in data.columns: data.rename(columns={'AJCC Stage': 'AJCC_Stage'}, inplace=True)

    data = data[data['CANCER_TYPE'].notna()].copy()
    data['CANCER_TYPE'] = data['CANCER_TYPE'].astype(str).str.strip()

    # Identificazione precisa delle colonne cliniche nell'Excel corrente
    colonna_target_stadio = None
    for c in data.columns:
        if 'AJCC Stage' in str(c) or 'AJCC_Stage' in str(c) or c == 'Stage':
            colonna_target_stadio = c

    # 2. ESCLUSIONE TOTALE DATA LEAKAGE CLINICO ED ANAGRAFICO (Allineamento Simmetrico)
    lista_nera_clinica = [
        'Patient ID', 'Sample ID', 'Patient ID #', 'Sample ID #',
        'Tumor type', 'CANCER_TYPE', 'AJCC Stage', 'AJCC_Stage', 'Stage',
        'Age', 'Sex', 'Race', 'Class', 'CancerSEEK Test Result',
        'CancerSEEK Logistic Regression Score', 'Omega Score'
    ]

    # Estrarre l'elenco dei soli biomarcatori molecolari reali (Le 39 proteine pure)
    feature_cols = [c for c in data.columns if
                    c not in lista_nera_clinica and not any(x in str(c) for x in ['Logistic', 'Regr'])]

    # 3. CONVERSIONE FORZATA DI TUTTI I BIOMARCATORI IN FLOAT
    for col in feature_cols:
        if data[col].dtype == 'object':
            data[col] = data[col].astype(str).str.replace('*', '', regex=False).str.replace('**', '', regex=False)
        data[col] = pd.to_numeric(data[col], errors='coerce')

    log.info("[FILTRATO] Configurato con %d biomarcatori proteici puri.", len(feature_cols))

    # Mappatura e pulizia colonne cliniche d'appoggio prima degli split fisici
    if colonna_target_stadio:
        data[colonna_target_stadio] = data[colonna_target_stadio].map({'I': 1, 'II': 2, 'III': 3, 'NA': 0}).fillna(0)

    # 4. ISOLAMENTO IMMUTABILE DEL TEST SET (20% dei pazienti reali)
    df_train_raw, df_test = train_test_split(data, test_size=0.20, random_state=42, stratify=data["CANCER_TYPE"])

    # 5. ISOLAMENTO IMMEDIATO DEL VALIDATION SET PURO (10% dei pazienti reali rimasti)
    df_train, df_val = train_test_split(df_train_raw, test_size=0.10, random_state=42,
                                        stratify=df_train_raw["CANCER_TYPE"])

    # 6. CONFIGURAZIONE GRUPPI MIX-UP ED AUGMENTATION SUL SOLO TRAIN
    group_cols = ["CANCER_TYPE"]
    if colonna_target_stadio:
        group_cols.append(colonna_target_stadio)

    categorical_features = ["CANCER_TYPE"]
    if colonna_target_stadio:
        categorical_features.append(colonna_target_stadio)

    df_mix = tabular_mixup(df_train, group_cols, feature_cols, categorical_features)
    df_train_augmented = pd.concat([df_train, df_mix], ignore_index=True)

    # 7. IMPUTAZIONE CALCOLATA SULLE MEDIANE DEL TRAIN GENERATO
    medians = df_train_augmented[feature_cols].median()

    X_train = df_train_augmented[feature_cols].fillna(medians).values.astype(np.float32)
    y_train = df_train_augmented["CANCER_TYPE"].values

    X_val = df_val[feature_cols].fillna(medians).values.astype(np.float32)
    y_val = df_val["CANCER_TYPE"].values

    X_test = df_test[feature_cols].fillna(medians).values.astype(np.float32)
    y_test = df_test["CANCER_TYPE"].values

    # 8. SCALING COERENTE SENZA LEAKAGE
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # 9. APPLICAZIONE DELLO SMOTE + RUMORE SUL TRAIN SCALATO
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
    X_train_final = (X_train_res + np.random.normal(0, 0.01, X_train_res.shape)).astype(np.float32)
    y_train_final = y_train_res

    # 10. ENCODING DELLE ETICHETTE CLINICHE
    le = LabelEncoder().fit(data["CANCER_TYPE"])

    # CREAZIONE DEL DATAFRAME DI TEST ALLINEATO E PROTETTO PER SHAP
    x_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    x_test_df["CANCER_TYPE"] = y_test
    if colonna_target_stadio:
        x_test_df["AJCC_Stage"] = df_test[colonna_target_stadio].values

    n_classes = len(le.classes_)
    genes_len = X_train_final.shape[1]

    # Esportazione di tracciabilità del dataset
    df_export = pd.DataFrame(X_train_final, columns=feature_cols)
    df_export["CANCER_TYPE"] = y_train_final
    df_export.to_excel("Data_FILTRATO.xlsx", index=False)
    log.info("[FILE] Dataset aumentato salvato come: Data_FILTRATO.xlsx")

    return (
        X_train_final, X_val_scaled, X_test_scaled,
        le.transform(y_train_final), le.transform(y_val), le.transform(y_test),
        x_test_df, le, n_classes, genes_len
    )