# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseBranch(nn.Module):
    def __init__(self, input_len=43, n_classes=9, embedding_units=64):
        super(SiameseBranch, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        flat_features = 256 * (((input_len // 2) // 2) // 2)

        self.fc1 = nn.Linear(flat_features, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.4)
        self.embedding = nn.Linear(512, embedding_units)
        self.classifier = nn.Linear(embedding_units, n_classes)

    def forward(self, x, return_embedding=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(F.relu(self.fc1(x)))
        emb = self.dropout2(F.relu(self.embedding(x)))
        if return_embedding:
            return emb
        return self.classifier(emb)


class SiameseNetwork(nn.Module):
    def __init__(self, pretrained_branch):
        super(SiameseNetwork, self).__init__()
        self.branch = pretrained_branch
        for param in self.branch.parameters():
            param.requires_grad = False

        self.dense = nn.Linear(64, 256)
        self.classifier = nn.Linear(256, 1)

    def forward(self, input_a, input_b):
        emb_a = self.branch(input_a, return_embedding=True)
        emb_b = self.branch(input_b, return_embedding=True)
        epsilon = 1e-7
        l2_distance = ((emb_a - emb_b) ** 2) / (emb_a + emb_b + epsilon)
        x = F.relu(self.dense(l2_distance))
        return torch.sigmoid(self.classifier(x))