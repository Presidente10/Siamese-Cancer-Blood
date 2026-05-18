import numpy as np
import tensorflow as tf
# Importiamo il main dal file preprocessing (SMOTE) dei tuoi colleghi
from preprocessing import main as load_data_smote
from siamese_model_core import build_siamese_model, contrastive_loss

print("=== FASE 1: Caricamento dati con SMOTE ===")
(
    x_train_df, x_val_df, x_test_df,
    y_test_raw, y_train_enc, y_val_enc, y_test_enc,
    _, _, _,
    X_train_resh, X_val_resh, X_test_resh,
    _, _, _,
    le, n_classes, genes_len
) = load_data_smote()


def create_pairs(x_df, labels_enc):
    pairs = []
    labels = []
    # Escludiamo le colonne di target e stadio per tenere solo i biomarcatori
    x_values = x_df.drop(columns=["CANCER_TYPE", "AJCC_Stage"], errors="ignore").values

    unique_classes = np.unique(labels_enc)
    digit_indices = [np.where(labels_enc == i)[0] for i in unique_classes]

    for idx1 in range(len(x_values)):
        x1 = x_values[idx1]
        label1 = labels_enc[idx1]

        # Coppia POSITIVA (stesso tumore -> etichetta 1)
        idx2 = np.random.choice(digit_indices[label1])
        x2 = x_values[idx2]
        pairs += [[x1, x2]]
        labels += [1]

        # Coppia NEGATIVA (tumore diverso -> etichetta 0)
        label2 = np.random.choice([l for l in unique_classes if l != label1])
        idx3 = np.random.choice(digit_indices[label2])
        x3 = x_values[idx3]
        pairs += [[x1, x3]]
        labels += [0]

    return np.array(pairs), np.array(labels)


print("\n=== FASE 2: Generazione Coppie per la Rete Siamese ===")
pairs_train, labels_train = create_pairs(x_train_df, y_train_enc)
pairs_val, labels_val = create_pairs(x_val_df, y_val_enc)

print(f"Coppie di addestramento create: {len(pairs_train)}")

# Inizializziamo il modello con il numero corretto di biomarcatori
input_shape = (genes_len,)
model = build_siamese_model(input_shape)
model.compile(loss=contrastive_loss, optimizer='adam', metrics=['accuracy'])

print("\n=== FASE 3: Avvio Allenamento Siamese ===")
model.fit(
    [pairs_train[:, 0], pairs_train[:, 1]], labels_train,
    batch_size=32,
    epochs=10,
    validation_data=([pairs_val[:, 0], pairs_val[:, 1]], labels_val)
)

print("\nAllenamento completato con successo!")