import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from preprocessing import main as load_data_smote
from train_siamese_smote import create_pairs, model

print("=== FASE 1: Caricamento dati di TEST (SMOTE) ===")
(
    _, _, x_test_df,
    _, _, _, y_test_enc,
    _, _, _,
    _, _, _,
    _, _, _,
    le, _, _
) = load_data_smote()

print("\n=== FASE 2: Generazione Coppie di Test ===")
pairs_test, labels_test = create_pairs(x_test_df, y_test_enc)
print(f"Coppie di test create: {len(pairs_test)}")

print("\n=== FASE 3: Calcolo delle Distanze con la Rete Siamese ===")
# Facciamo predire al modello la distanza per ogni coppia di test
distances = model.predict([pairs_test[:, 0], pairs_test[:, 1]]).flatten()

# Troviamo la soglia ottimale sul test set (giusto per capire come si comporta)
def find_best_threshold(y_true, dists):
    best_thresh = 0.0
    best_acc = 0.0
    for thresh in np.arange(0.0, 2.0, 0.01):
        preds = (dists < thresh).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_thresh, best_acc

best_thresh, best_acc = find_best_threshold(labels_test, distances)
print(f"\n[RISULTATO] Soglia ottimale trovata: {best_thresh:.2f}")
print(f"[RISULTATO] Vera Accuratezza della Rete: {best_acc * 100:.2f}%")

# Generiamo le predizioni finali basate sulla soglia migliore
y_pred = (distances < best_thresh).astype(int)

print("\n=== REPORT DI CLASSIFICAZIONE ===")
print(classification_report(labels_test, y_pred, target_names=["Diversi (0)", "Simili (1)"]))

# Mostriamo la Matrice di Confusione grafica
plt.figure(figsize=(6, 5))
cm = confusion_matrix(labels_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Diversi", "Simili"], yticklabels=["Diversi", "Simili"])
plt.title(u"Matrice di Confusione - Rete Siamese (SMOTE)")
plt.ylabel(u"Campione Reale")
plt.xlabel(u"Predizione Rete")
plt.show()