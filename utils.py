# utils.py
import torch
import numpy as np
import random


def create_siamese_pairs_torch(X, y):
    pairs_A, pairs_B, labels = [], [], []
    indices_per_class = {}
    for idx, label in enumerate(y):
        indices_per_class.setdefault(int(label), []).append(idx)

    unique_labels = list(indices_per_class.keys())

    for label in unique_labels:
        indices = indices_per_class[label]
        n = len(indices)
        for i in range(n - 1):
            idx1, idx2 = indices[i], indices[i + 1]
            pairs_A.append(X[idx1])
            pairs_B.append(X[idx2])
            labels.append(1.0)

            neg_label = random.choice([l for l in unique_labels if l != label])
            neg_idx = random.choice(indices_per_class[neg_label])
            pairs_A.append(X[idx1])
            pairs_B.append(X[neg_idx])
            labels.append(0.0)

    return (torch.tensor(np.array(pairs_A), dtype=torch.float32),
            torch.tensor(np.array(pairs_B), dtype=torch.float32),
            torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(1))


def test_kshot_torch(branch_extractor, x_test_df, X_test_resh, k_shot=5, label_column='CANCER_TYPE'):
    branch_extractor.eval()
    unique_classes = sorted(x_test_df[label_column].unique())
    references = {}

    with torch.no_grad():
        for cls in unique_classes:
            indices_cls = list(x_test_df.index[x_test_df[label_column] == cls])
            selected_indices = random.sample(indices_cls, min(k_shot, len(indices_cls)))
            embeddings = []
            for idx in selected_indices:
                sample = torch.tensor(X_test_resh[idx], dtype=torch.float32).unsqueeze(0).transpose(1, 2)
                emb = branch_extractor(sample, return_embedding=True)
                embeddings.append(emb.squeeze(0).numpy())
            references[cls] = np.stack(embeddings, axis=0)

    predictions = []
    test_ind = list(x_test_df.index)

    with torch.no_grad():
        for i in test_ind:
            current_class = x_test_df.loc[i, label_column]
            query_sample = torch.tensor(X_test_resh[i], dtype=torch.float32).unsqueeze(0).transpose(1, 2)
            query_embedding = branch_extractor(query_sample, return_embedding=True).squeeze(0).numpy()

            indices_cls = list(x_test_df.index[x_test_df[label_column] == current_class])
            if i in indices_cls:
                indices_cls.remove(i)

            if len(indices_cls) > 0:
                selected_indices = random.sample(indices_cls, min(k_shot, len(indices_cls)))
                embeddings = []
                for idx in selected_indices:
                    s = torch.tensor(X_test_resh[idx], dtype=torch.float32).unsqueeze(0).transpose(1, 2)
                    e = branch_extractor(s, return_embedding=True)
                    embeddings.append(e.squeeze(0).numpy())
                ref_embedding = np.mean(embeddings, axis=0)
            else:
                ref_embedding = np.mean(references[current_class], axis=0)

            distances = {}
            for cls, ref_embeddings in references.items():
                if cls == current_class and len(indices_cls) > 0:
                    distances[cls] = np.linalg.norm(ref_embedding - query_embedding)
                else:
                    dists = np.linalg.norm(ref_embeddings - query_embedding, axis=1)
                    distances[cls] = np.mean(dists)

            pred_class = min(distances, key=distances.get)
            predictions.append(pred_class)

    true_labels = list(x_test_df[label_column])
    acc = np.mean([pred == true for pred, true in zip(predictions, true_labels)]) * 100
    return acc