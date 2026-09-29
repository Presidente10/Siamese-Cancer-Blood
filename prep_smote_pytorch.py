import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def convert_and_impute(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

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
    data = data.copy()
    if 'AJCC Stage' in data.columns:
        data['AJCC Stage'] = data['AJCC Stage'].map({'I': 1, 'II': 2, 'III': 3, 'NA': 0}).fillna(0)
    if 'Sex' in data.columns:
        data['Sex'] = data['Sex'].map({'Male': 0, 'Female': 1})
    return data


def main():
    df = pd.read_excel("C:/Users/danid/PycharmProjects/Siamese-Cancer-Blood/Data.xlsx")
    df.replace("?", pd.NA, inplace=True)
    df.drop(columns=['Patient ID #', 'Sample ID #', 'CancerSEEK Logistic Regression Score',
                     'CancerSEEK Test Result', 'Class', 'Race'], errors='ignore', inplace=True)

    # 1) Calcolo mediana globale e pulizia identica alla sua
    df = convert_and_impute(df)
    df = encode_columns(df)

    if "Tumor type" in df.columns:
        df.rename(columns={"Tumor type": "CANCER_TYPE"}, inplace=True)
    if "AJCC Stage" in df.columns:
        df.rename(columns={"AJCC Stage": "AJCC_Stage"}, inplace=True)

    # 42 Feature totali
    feature_cols = [col for col in df.columns if col not in ["CANCER_TYPE", "AJCC_Stage"]]
    X = df[feature_cols].values
    y = df["CANCER_TYPE"].values

    # 2) Primo Split Onesto 80/20 Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Normalizzazione MinMaxScaler fit solo sul Train
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3) SMOTE applicato SOLO sul blocco di Train
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    # 4) Secondo Split 90/10 per la Validazione (Inquinata dai cloni SMOTE)
    X_train_final, X_val_scaled, y_train_final, y_val = train_test_split(
        X_train_res, y_train_res, test_size=0.10, random_state=42
    )

    # Label Encoder per i target
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_final)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    # Reshape tridimensionale per PyTorch (CNN 1D)
    X_train_pt = X_train_final.reshape(X_train_final.shape[0], 1, X_train_final.shape[1])
    X_val_pt = X_val_scaled.reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])
    X_test_pt = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

    print(f"📊 [REPLICA 100%] Train shape finale (Solo SMOTE): {X_train_pt.shape}")
    print(f"📊 [REPLICA 100%] Validation shape: {X_val_pt.shape}")
    print(f"🛡️ [REPLICA 100%] Test Set Reale shape: {X_test_pt.shape}")

    return X_train_pt, X_val_pt, X_test_pt, y_train_enc, y_val_enc, y_test_enc, le


if __name__ == "__main__":
    main()