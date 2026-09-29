# pipeline_smote_clean.py
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def prepare_data_smote(excel_path: str):
    """Pipeline SMOTE protetta anti-leakage con 39 biomarcatori molecolari puri."""
    data = pd.read_excel(excel_path)

    # 1. 🛡️ PULIZIA IMMEDIATA STRINGHE E CARATTERI SCONOSCIUTI (Anti-Crash)
    data.replace("?", pd.NA, inplace=True)

    # Standardizzazione nomi colonne target
    if 'Tumor type' in data.columns: data.rename(columns={'Tumor type': 'CANCER_TYPE'}, inplace=True)
    if 'AJCC Stage' in data.columns: data.rename(columns={'AJCC Stage': 'AJCC_Stage'}, inplace=True)

    data = data[data['CANCER_TYPE'].notna()]
    data['CANCER_TYPE'] = data['CANCER_TYPE'].astype(str).str.strip()

    # Identificazione precisa delle colonne cliniche nell'Excel corrente
    colonna_target_stadio = None
    for c in data.columns:
        if 'AJCC Stage' in str(c) or 'AJCC_Stage' in str(c) or c == 'Stage':
            colonna_target_stadio = c

    # 2. ESCLUSIONE TOTALE DATA LEAKAGE CLINICO ED ANAGRAFICO
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

    # 4. IMPUTAZIONE CALCOLATA SULLE MEDIANE EFFETTIVE DRLLE PROTEINE
    for col in feature_cols:
        col_median = data[col].median()
        data[col] = data[col].fillna(col_median if pd.notna(col_median) else 0.0)

    log.info("[SMOTE] Configurato con %d biomarcatori proteici puri.", len(feature_cols))

    # Mappatura e pulizia colonne cliniche d'appoggio prima dell'estrazione matrici
    if colonna_target_stadio:
        data[colonna_target_stadio] = data[colonna_target_stadio].map({'I': 1, 'II': 2, 'III': 3, 'NA': 0}).fillna(0)

    # Estrazione finale sicura in float32 nativo
    X = data[feature_cols].values.astype(np.float32)
    y = data["CANCER_TYPE"].values

    # 5. ISOLAMENTO TEST SET (20%) REALE E PURO
    X_train_full, X_test_raw, y_train_full, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # 6. ISOLAMENTO DEL VALIDATION SET REALE (10% del rimanente)
    X_train_raw, X_val_raw, y_train_raw, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.10, random_state=42, stratify=y_train_full
    )

    # 7. SCALATURA COERENTE SENZA LEAKAGE
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)

    # 8. Lo SMOTE interviene sul Train Scaled molecolare Puro
    sm = SMOTE(random_state=42)
    X_train_final, y_train_final = sm.fit_resample(X_train_scaled, y_train_raw)
    X_train_final = X_train_final.astype(np.float32)

    # 9. CODIFICA DELLE ETICHETTE CLINICHE
    le = LabelEncoder()
    le.fit(data["CANCER_TYPE"])

    # 10. COSTRUZIONE DATAFRAME DI TEST MAPPATO CON I NOMI PROTEICI REALI PER SHAP
    x_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    x_test_df["CANCER_TYPE"] = le.inverse_transform(le.transform(y_test))

    if colonna_target_stadio:
        df_orig_test = data.iloc[train_test_split(data.index, test_size=0.20, random_state=42, stratify=y)[1]]
        x_test_df["AJCC_Stage"] = df_orig_test[colonna_target_stadio].values

    n_classes = len(le.classes_)
    genes_len = X_train_final.shape[1]

    # Salvataggio del dataset bilanciato tramite SMOTE per tracciabilità
    df_export = pd.DataFrame(X_train_final, columns=feature_cols)
    df_export["CANCER_TYPE"] = le.inverse_transform(le.transform(y_train_final))
    df_export.to_excel("Data_SMOTE.xlsx", index=False)
    log.info("[FILE] Dataset bilanciato salvato come: Data_SMOTE.xlsx")

    return (
        X_train_final, X_val_scaled, X_test_scaled,
        le.transform(y_train_final), le.transform(y_val), le.transform(y_test),
        x_test_df, le, n_classes, genes_len
    )