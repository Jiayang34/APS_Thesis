import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
from torch.utils.data import Dataset
from .aps import split_data_set
from torch.utils.data import DataLoader
from .aps_real_probs import aps_scores_real_probs, aps_classification_cifar10h, eval_aps_real_probs, hist_cifar10h


def generate_synthetic_data(k, save_path):
    """
    generate synthetic data and save to pickle-file

    Args:
        K (int): num_classes in range of [3,5,10]
        save_path (str): file (absolute) save path

    Returns:
        None
    """

    exp_seed = 200
    np.random.seed(exp_seed)
    feature_dim = 64  # feature vector 64×1
    n = 10000         # 10000 images

    # generate n random feature vector
    x = np.random.normal(loc=0.0, scale=1.0, size=(n, feature_dim))
    # label weight beta
    beta = np.random.normal(loc=0.0, scale=1.0, size=(feature_dim, k))

    # w: ground-truth real probability
    z = np.exp(np.matmul(x, beta))
    w = z / z.sum(axis=1, keepdims=True)

    # generate true label
    true_labels = np.array([np.random.choice(range(k), p=single_w) for single_w in w])

    # generate(simulate) real probability given by human expert w_annotated
    w_annotated = {}
    for annotator_num in [1, 5, 10, 50, 100, 200, 1000]:
        w_annotated[annotator_num] = np.array([
            np.random.multinomial(annotator_num, single_w) for single_w in w]) / annotator_num

    # data package
    data = {
        "x": x,
        "w": w,
        "true_labels": true_labels,
        "w_annotated": w_annotated
    }

    # save data
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Synthetic Data with {k} classes has been saved at: {save_path}")


class SimplePredictor(nn.Module):
    def __init__(self, feature_dim=64, n_classes=3, dropout_rate=0.3):
        super(SimplePredictor, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.out = nn.Linear(16, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return self.out(x)


def load_synthetic_data(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    x = data['x']
    true_labels = data['true_labels']
    real_probs = data['w_annotated'][1000]  # load real probability with 1000 participants
    return x, true_labels, real_probs


def train_simple_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for x, true_labels in train_loader:
            x, true_labels = x.to(device), true_labels.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, true_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == true_labels).sum().item()
            train_total += x.size(0)

        val_loss, val_correct, val_total = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for x, true_labels in val_loader:
                x, true_labels = x.to(device), true_labels.to(device)
                val_outputs = model(x)
                v_loss = criterion(val_outputs, true_labels)
                val_loss += v_loss.item() * x.size(0)
                val_preds = val_outputs.argmax(dim=1)
                val_correct += (val_preds == true_labels).sum().item()
                val_total += x.size(0)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/train_total:.4f}, Acc: {train_correct/train_total:.4f} "
              f"| Val Loss: {val_loss/val_total:.4f}, Acc: {val_correct/val_total:.4f}")


class SyntheticDataset_and_Probs(Dataset):
    def __init__(self, features, labels, real_probs):
        self.features = features
        self.labels = labels
        self.real_probs = real_probs

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        real_prob = torch.tensor(self.real_probs[idx], dtype=torch.float32)
        return feature, label, real_prob


def aps_synthetic_data(model, synthetic_dataset, device, num_runs=10, alpha=0.1):
    all_avg_set_sizes = []
    all_avg_coverages = []
    all_avg_real_probs = []
    all_real_probs_distribution = []
    print(f"APS Classification on Synthetic Data(alpha={alpha}), Start!\n")

    for i in range(num_runs):
        print(f"Running experiment {i + 1}/{num_runs}...")
        calib_dataset, test_dataset = split_data_set(synthetic_dataset, random_seed=i)
        calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        calib_scores, _ = aps_scores_real_probs(model, calib_loader, alpha, device)
        q_hat = np.quantile(calib_scores, 1 - alpha)

        aps, aps_labels, true_labels, real_probs = aps_classification_cifar10h(model, test_loader, q_hat, device)

        avg_set_size, avg_coverage = eval_aps_real_probs(aps_labels, true_labels)
        sum_real_probs = [sum(probs) for probs in real_probs]
        avg_real_prob = np.mean(sum_real_probs)

        all_avg_set_sizes.append(avg_set_size)
        all_avg_coverages.append(avg_coverage)
        all_avg_real_probs.append(avg_real_prob)
        all_real_probs_distribution.extend(sum_real_probs)

    # calculate the final average result
    final_avg_set_size = np.mean(all_avg_set_sizes)
    final_avg_coverage = np.mean(all_avg_coverages)
    final_avg_real_prob = np.mean(all_avg_real_probs)
    final_set_size_std = np.std(all_avg_set_sizes, ddof=0)
    final_coverage_std = np.std(all_avg_coverages, ddof=0)
    final_real_prob_std = np.std(all_avg_real_probs, ddof=0)

    print(f"Final Average Prediction Set Size: {final_avg_set_size:.2f} ± {final_set_size_std:.2f}")
    print(f"Final Average Coverage: {final_avg_coverage:.4f} ± {final_coverage_std:.4f}")
    print(f"Final Average Real Probability: {final_avg_real_prob:.4f} ± {final_real_prob_std:.4f}")

    hist_cifar10h(all_real_probs_distribution)

