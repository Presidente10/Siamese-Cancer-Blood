# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Importiamo le funzioni dai file creati e da preprocessing
from preprocessing import prepare_data_for_models
from models import SiameseBranch, SiameseNetwork
from utils import create_siamese_pairs_torch, test_kshot_torch

if __name__ == "__main__":
    excel_path = "Data.xlsx"

    # 1. Caricamento dati (usando lo script che hai già)
    (x_train_df, x_val_df, x_test_df,
     y_test, y_train_enc, y_val_enc, y_test_enc,
     X_train, X_val, X_test,
     X_train_resh, X_val_resh, X_test_resh,
     y_train_ohe, y_val_ohe, y_test_ohe,
     le, n_classes, genes_len) = prepare_data_for_models(excel_path)

    # Adattiamo lo shape per PyTorch convoluzionale 1D: da [B, 43, 1] a [B, 1, 43]
    X_train_pt = torch.tensor(X_train_resh, dtype=torch.float32).transpose(1, 2)
    X_val_pt = torch.tensor(X_val_resh, dtype=torch.float32).transpose(1, 2)
    y_train_pt = torch.tensor(y_train_enc, dtype=torch.long)
    y_val_pt = torch.tensor(y_val_enc, dtype=torch.long)

    # 2. Inizializzazione Branch Model e Addestramento Classico
    branch_model = SiameseBranch(input_len=genes_len, n_classes=n_classes)
    criterion_cls = nn.CrossEntropyLoss()
    optimizer_branch = optim.Adam(branch_model.parameters(), lr=0.0001, weight_decay=1e-5)

    train_loader = DataLoader(TensorDataset(X_train_pt, y_train_pt), batch_size=64, shuffle=True)

    print("--- Training del Branch Model (Fase 1) ---")
    branch_model.train()
    for epoch in range(20):  # Puoi aumentare le epoche
        for batch_x, batch_y in train_loader:
            optimizer_branch.zero_grad()
            outputs = branch_model(batch_x)
            loss = criterion_cls(outputs, batch_y)
            loss.backward()
            optimizer_branch.step()

    # 3. Creazione del Modello Siamese e Congelamento del Branch
    siamese_model = SiameseNetwork(branch_model)
    criterion_siamese = nn.BCELoss()
    optimizer_siamese = optim.Adam(filter(lambda p: p.requires_grad, siamese_model.parameters()), lr=0.001)

    # Generazione delle coppie per la rete siamese
    pairs_A, pairs_B, pair_labels = create_siamese_pairs_torch(X_train_resh, y_train_enc)
    # Adattiamo lo shape delle coppie per PyTorch [B, 1, 43]
    pairs_A = pairs_A.transpose(1, 2)
    pairs_B = pairs_B.transpose(1, 2)

    siamese_loader = DataLoader(TensorDataset(pairs_A, pairs_B, pair_labels), batch_size=64, shuffle=True)

    print("\n--- Training della testa della Rete Siamese (Fase 2) ---")
    siamese_model.train()
    for epoch in range(15):
        for b_a, b_b, b_l in siamese_loader:
            optimizer_siamese.zero_grad()
            predictions = siamese_model(b_a, b_b)
            loss = criterion_siamese(predictions, b_l)
            loss.backward()
            optimizer_siamese.step()

    # 4. Valutazione K-Shot (L'esperimento per calcolare l'81%)
    print("\n--- Valutazione K-Shot in corso... ---")
    accuracy_finale = test_kshot_torch(
        branch_extractor=branch_model,
        x_test_df=x_test_df,
        X_test_resh=X_test_resh,
        k_shot=5,
        label_column='CANCER_TYPE'
    )

    print(f"\n[One-shot] Accuratezza K-Shot (Siamese in PyTorch): {accuracy_finale:.2f}%")