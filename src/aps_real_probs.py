from torch.utils.data import random_split
import torch
import numpy as np
import os
from torch.utils.data import Dataset
import seaborn as sb
from matplotlib import pyplot as plt


class Dataset_and_Probs(Dataset):
    def __init__(self, dataset, npy_files):
        self.dataset = dataset
        self.real_probs = npy_files

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        real_prob = torch.tensor(self.real_probs[idx], dtype=torch.float32)
        return img, label, real_prob


def split_data_set_cifar10h(dataset, random_seed):
    # load real probabilities from CIFAR10-H
    current_dir = os.path.dirname(os.path.abspath(__file__))
    npy_path = os.path.join(current_dir, "../data/cifar10h-probs.npy")
    cifar10h_probs = np.load(npy_path)

    # pack CIFAR10 with real probabilities
    cifar10h = Dataset_and_Probs(dataset, cifar10h_probs)

    if random_seed is not None:
        torch.manual_seed(random_seed)  # set input as random seed

    # split image set ---> half for calibration data set, half for test data set
    dataset_length = len(cifar10h)
    calib_length = dataset_length // 2
    test_length = dataset_length - calib_length

    calib_dataset, test_dataset = random_split(cifar10h, [calib_length, test_length])
    return calib_dataset, test_dataset


def split_data_set_imagenet_real(dataset, random_seed):
    # load real probabilities from ImageNer-Real
    current_dir = os.path.dirname(os.path.abspath(__file__))
    npy_path = os.path.join(current_dir, "../data/imagenet_count.npy")
    imagenet_real_probs = np.load(npy_path)

    # pack ImageNet with real probabilities
    imagenet_real = Dataset_and_Probs(dataset, imagenet_real_probs)

    if random_seed is not None:
        torch.manual_seed(random_seed)  # set input as random seed

    # split image set ---> half for calibration data set, half for test data set
    dataset_length = len(imagenet_real)
    calib_length = dataset_length // 2
    test_length = dataset_length - calib_length

    calib_dataset, test_dataset = random_split(imagenet_real, [calib_length, test_length])
    return calib_dataset, test_dataset


def split_data_set_imagenet_real_normalize(dataset, random_seed):
    # load real probabilities from ImageNer-Real
    current_dir = os.path.dirname(os.path.abspath(__file__))
    npy_path = os.path.join(current_dir, "../data/imagenet_count_normalize.npy")
    imagenet_real_probs = np.load(npy_path)

    # pack ImageNet with real probabilities
    imagenet_real = Dataset_and_Probs(dataset, imagenet_real_probs)

    if random_seed is not None:
        torch.manual_seed(random_seed)  # set input as random seed

    # split image set ---> half for calibration data set, half for test data set
    dataset_length = len(imagenet_real)
    calib_length = dataset_length // 2
    test_length = dataset_length - calib_length

    calib_dataset, test_dataset = random_split(imagenet_real, [calib_length, test_length])
    return calib_dataset, test_dataset


def aps_scores_real_probs(model, dataloader, alpha=0.1, device='cpu'):
    scores = []  # conformal scores of image sets
    labels = []  # true label sets
    with torch.no_grad():
        for images, true_labels, real_probs in dataloader:
            images, true_labels = images.to(device), true_labels.to(device)
            outputs = model(images)
            softmaxs = torch.softmax(outputs, dim=1)

            # sort softmax scores in descending order and then cumulate
            sorted_softmax, sorted_index = torch.sort(softmaxs, descending=True, dim=1)
            cumulative_softmax = torch.cumsum(sorted_softmax, dim=1)

            # find indices of true labels
            true_label_positions = (sorted_index == true_labels.unsqueeze(1)).nonzero(as_tuple=True)[1]

            # random variable u with the same size of sorted_softmax
            u = torch.rand_like(sorted_softmax, dtype=torch.float, device=device)

            # calculate the scores for all the labels
            scores_all_labels = cumulative_softmax - sorted_softmax + u * sorted_softmax
            # select the scores of true label
            conformal_scores = scores_all_labels.gather(1, true_label_positions.unsqueeze(1)).squeeze(1)

            scores.extend(conformal_scores.cpu().numpy().tolist())
            labels.extend(true_labels.cpu().numpy().tolist())

    return np.array(scores), np.array(labels)


def raps_scores_real_probs(model, dataloader, alpha=0.1, lambda_reg=0.1, k_reg=5, device='cpu'):
    scores = []  # conformal scores of image sets
    labels = []  # true label sets
    with torch.no_grad():
        for images, true_labels, real_probs in dataloader:
            images, true_labels = images.to(device), true_labels.to(device)
            outputs = model(images)
            softmaxs = torch.softmax(outputs, dim=1)

            # sort and cumulate
            sorted_softmax, sorted_index = torch.sort(softmaxs, descending=True)
            cumulative_softmax = torch.cumsum(sorted_softmax, dim=1)

            # find indices of true labels
            true_label_positions = (sorted_index == true_labels.unsqueeze(1)).nonzero(as_tuple=True)[1]

            # random variable u with the same size of sorted_softmax
            u = torch.rand_like(sorted_softmax, dtype=torch.float, device=device)

            # calculate the aps-scores for all the labels
            scores_all_labels = cumulative_softmax - sorted_softmax + u * sorted_softmax
            # select the aps-scores of true label
            aps_scores = scores_all_labels.gather(1, true_label_positions.unsqueeze(1)).squeeze(1)

            # regularization term
            regularization_term = lambda_reg * torch.clamp((true_label_positions + 1 - k_reg).float(), min=0)

            # Compute raps-scores = aps-score + regularization term
            conformal_scores = aps_scores + regularization_term

            scores.extend(conformal_scores.cpu().tolist())
            labels.extend(true_labels.cpu().tolist())
    return scores, labels


def saps_scores_real_probs(model, dataloader, alpha=0.1, lambda_=0.1, device='cpu'):
    scores = []  # conformal scores of image sets
    labels = []  # true label sets
    with torch.no_grad():
        for images, true_labels, real_probs in dataloader:
            images, true_labels = images.to(device), true_labels.to(device)
            outputs = model(images)
            softmaxs = torch.softmax(outputs, dim=1)

            # extract true lables' ranking/positions
            sorted_softmax, sorted_indices = torch.sort(softmaxs, descending=True, dim=1)
            true_label_positions = (sorted_indices == true_labels.unsqueeze(1)).nonzero(as_tuple=True)[1]

            # extract maximal probabilities
            max_softmax = sorted_softmax[:, 0]

            # random variable u(s)
            u = torch.rand(true_labels.size(0), device=device)
            # scores of samples whose correct label is top-ranking --> u * max_softmax
            is_top = (true_label_positions == 0)
            scores_top_rank = u * max_softmax

            # scores of samples whose correct label is  not top-ranking
            # s = max_softmax + (o-2+u) * lambda = max_softmax + (true_label_position+1-2+u) * lambda
            scores_other_rank = max_softmax + ((true_label_positions - 1).float() + u) * lambda_

            conformal_scores = torch.where(is_top, scores_top_rank, scores_other_rank)
            scores.extend(conformal_scores.cpu().tolist())
            labels.extend(true_labels.cpu().tolist())
    return scores, labels


def aps_classification_cifar10h(model, dataloader, q_hat, device='cpu'):
    aps = []  # probability set
    aps_labels = []  # label set indicated to the probability set
    labels = []  # true label
    real_probs = []  # real probability of prediction set
    with torch.no_grad():
        for images, true_labels, probs in dataloader:
            images, true_labels, probs = images.to(device), true_labels.to(device), probs.to(device)
            outputs = model(images)
            softmaxs = torch.softmax(outputs, dim=1)

            # sort and cumulate
            sorted_softmax, sorted_index = torch.sort(softmaxs, descending=True, dim=1)
            cumulative_softmax = torch.cumsum(sorted_softmax, dim=1)

            # random variable u with the same size of sorted_softmax
            u = torch.rand_like(sorted_softmax, dtype=torch.float, device=device)

            # compute scores for all labels
            scores = cumulative_softmax - sorted_softmax + u * sorted_softmax

            # cutoff index = the first index above q_hat
            cutoff_indices = torch.searchsorted(scores, torch.full_like(scores[:, :1], q_hat), right=True)

            # extract prediction sets
            batch_size = images.shape[0]
            for i in range(batch_size):
                cutoff_index = cutoff_indices[i].item()

                # prediction set = all the label before cutoff index (exclusive cutoff)
                aps.append(sorted_softmax[i, :cutoff_index].cpu().tolist())
                labels.append(true_labels[i].item())
                # label set and real probability set
                # e.g. C1 = {1,2,3} ; real prob from CIFAR10-H: Label_1=0.4, Label_2=0.3, Label_3=0.1
                # real_prob of C1 = {0.4, 0.3, 0.1}
                pred_labels = sorted_index[i, :cutoff_index].cpu().tolist()
                aps_labels.append(pred_labels)
                real_probs.append(probs[i, pred_labels].cpu().tolist())

    return aps, aps_labels, labels, real_probs


def aps_classification_imagenet_real(model, dataloader, q_hat, device='cpu'):
    aps = []  # probability set
    aps_labels = []  # label set indicated to the probability set
    labels = []  # true label
    real_probs = []  # real probability of prediction set
    with torch.no_grad():
        for images, true_labels, probs in dataloader:
            images, true_labels, probs = images.to(device), true_labels.to(device), probs.to(device)
            outputs = model(images)
            softmaxs = torch.softmax(outputs, dim=1)

            # sort and cumulate
            sorted_softmax, sorted_index = torch.sort(softmaxs, descending=True, dim=1)
            cumulative_softmax = torch.cumsum(sorted_softmax, dim=1)

            # random variable u with the same size of sorted_softmax
            u = torch.rand_like(sorted_softmax, dtype=torch.float, device=device)

            # compute scores for all labels
            scores = cumulative_softmax - sorted_softmax + u * sorted_softmax

            # cutoff index = the first index above q_hat
            cutoff_indices = torch.searchsorted(scores, torch.full_like(scores[:, :1], q_hat), right=True)

            # extract prediction sets
            batch_size = images.shape[0]
            for i in range(batch_size):
                cutoff_index = cutoff_indices[i].item()

                # prediction set = all the label before cutoff index (exclusive cutoff)
                aps.append(sorted_softmax[i, :cutoff_index].cpu().tolist())
                labels.append(true_labels[i].item())
                # label set and real probability set
                # e.g. C1 = {1,2,3} ; real prob from ImageNet-Real: Label_1=0.4, Label_2=0.3, Label_3=0.1
                # real_prob of C1 = {0.4, 0.3, 0.1}
                pred_labels = sorted_index[i, :cutoff_index].cpu().tolist()
                aps_labels.append(pred_labels)
                if torch.all(probs[i] == 0):
                    # if this sample has no real probability e.g. [0, 0, ..., 0] -> real_probs = [None]
                    real_probs.append(None)
                else:
                    # if APS construct an empty set for this sample -> real_probs = []
                    real_probs.append(probs[i, pred_labels].cpu().tolist())

    return aps, aps_labels, labels, real_probs


def raps_classification_cifar10h(model, dataloader, t_cal, lambda_reg=0.1, k_reg=5, device='cpu'):
    raps = []  # probability set
    raps_labels = []  # label set indicated to the probability set
    labels = []  # true label
    real_probs = []  # real probability of prediction set
    with torch.no_grad():
        for images, true_labels, probs in dataloader:
            images, true_labels, probs = images.to(device), true_labels.to(device), probs.to(device)
            outputs = model(images)
            softmaxs = torch.softmax(outputs, dim=1)

            # sort softmax probabilities
            sorted_softmax, sorted_indices = torch.sort(softmaxs, descending=True, dim=1)  # shape: [batch_size, 1000]
            cumulative_softmax = torch.cumsum(sorted_softmax, dim=1)  # shape: [batch_size, 1000]

            # rank of current sorted probability: [1,2,3,...,1000]
            rank = torch.arange(1, sorted_softmax.size(1) + 1, device=device).unsqueeze(0)  # shape: [1, 1000]
            # calculate regularization term
            regularization_term = lambda_reg * torch.clamp(rank - k_reg, min=0)  # shape: [1, 1000]

            # generate random variable u for all samples
            u = torch.rand_like(sorted_softmax)  # shape: [batch_size, 1000]

            # E = cumulative[current-1] + u*sorted[current] + regularization
            # which is equal to: cumulative[current] - sorted[current] + u*sorted[current] + regularization
            e = cumulative_softmax - sorted_softmax + u * sorted_softmax + regularization_term  # shape: [batch_size, 1000]

            e_less_than_t = e <= t_cal
            cutoff_indices = torch.sum(e_less_than_t, dim=1)

            # build prediction sets
            # build prediction sets
            batch_size = images.shape[0]
            for i in range(batch_size):
                cutoff_index = cutoff_indices[i].item()

                if cutoff_index > 0:
                    pred_label = sorted_indices[i, :cutoff_index].cpu().tolist()
                    pred_prob = sorted_softmax[i, :cutoff_index].cpu().tolist()
                    real_prob = probs[i, pred_label].cpu().tolist()
                else:
                    pred_label = []
                    pred_prob = []
                    real_prob = []

                raps.append(pred_prob)
                raps_labels.append(pred_label)
                real_probs.append(real_prob)
                labels.append(true_labels[i].item())

    return raps, raps_labels, labels, real_probs


def raps_classification_imagenet_real(model, dataloader, t_cal, lambda_reg=0.1, k_reg=5, device='cpu'):
    raps = []  # probability set
    raps_labels = []  # label set indicated to the probability set
    labels = []  # true label
    real_probs = []  # real probability of prediction set
    with torch.no_grad():
        for images, true_labels, probs in dataloader:
            images, true_labels, probs = images.to(device), true_labels.to(device), probs.to(device)
            outputs = model(images)
            softmaxs = torch.softmax(outputs, dim=1)

            # sort softmax probabilities
            sorted_softmax, sorted_indices = torch.sort(softmaxs, descending=True, dim=1)  # shape: [batch_size, 1000]
            cumulative_softmax = torch.cumsum(sorted_softmax, dim=1)  # shape: [batch_size, 1000]

            # rank of current sorted probability: [1,2,3,...,1000]
            rank = torch.arange(1, sorted_softmax.size(1) + 1, device=device).unsqueeze(0)  # shape: [1, 1000]
            # calculate regularization term
            regularization_term = lambda_reg * torch.clamp(rank - k_reg, min=0)  # shape: [1, 1000]

            # generate random variable u for all samples
            u = torch.rand_like(sorted_softmax)  # shape: [batch_size, 1000]

            # E = cumulative[current-1] + u*sorted[current] + regularization
            # which is equal to: cumulative[current] - sorted[current] + u*sorted[current] + regularization
            e = cumulative_softmax - sorted_softmax + u * sorted_softmax + regularization_term  # shape: [batch_size, 1000]

            e_less_than_t = e <= t_cal
            cutoff_indices = torch.sum(e_less_than_t, dim=1)

            # build prediction sets
            # build prediction sets
            batch_size = images.shape[0]
            for i in range(batch_size):
                cutoff_index = cutoff_indices[i].item()

                if cutoff_index > 0:
                    pred_label = sorted_indices[i, :cutoff_index].cpu().tolist()
                    pred_prob = sorted_softmax[i, :cutoff_index].cpu().tolist()
                    if torch.all(probs[i] == 0):
                        real_prob = None   # real prob unmarked
                    else:
                        real_prob = probs[i, pred_label].cpu().tolist()
                else:
                    pred_label = []
                    pred_prob = []
                    if torch.all(probs[i] == 0):
                        real_prob = None    # real prob unmarked
                    else:
                        real_prob = []      # real prob marked but APS generate empty set

                raps.append(pred_prob)
                raps_labels.append(pred_label)
                real_probs.append(real_prob)
                labels.append(true_labels[i].item())

    return raps, raps_labels, labels, real_probs


def saps_classification_cifar10h(model, dataloader, t_cal, lambda_=0.1, device='cpu'):
    saps = []  # probability set
    saps_labels = []  # label set indicated to the probability set
    labels = []  # true label
    real_probs = []  # real probability of prediction set
    with torch.no_grad():
        for images, true_labels, probs in dataloader:
            images, true_labels, probs = images.to(device), true_labels.to(device), probs.to(device)
            outputs = model(images)
            softmax = torch.softmax(outputs, dim=1)

            # sort probabilities
            sorted_softmax, sorted_indices = torch.sort(softmax, descending=True, dim=1)

            # random variable u(s)
            u = torch.rand(sorted_softmax.shape, device=device)  # Shape: (batch_size, 100)
            # random variable for maximal probabilities
            u_f_max = torch.rand(sorted_softmax.shape[0], device=device).unsqueeze(1)  # Shape: (batch_size, 1)

            # rank of current sorted probability: [1,2,3,...,1000]
            rank = torch.arange(1, sorted_softmax.size(1) + 1, device=device).unsqueeze(0)  # shape: [1, 100]

            # s = f_max + (o-2+u) * lambda
            # scores --> all the label has been calculate as non-top-ranked label now
            f_max = sorted_softmax[:, 0].unsqueeze(1)  # Shape: (batch_size, 1)
            scores = f_max + ((rank - 2).float() + u) * lambda_  # Shape: (batch_size, 100)

            # replace the firt column with u * f_max
            scores[:, 0] = (u_f_max * f_max).squeeze(1)  # Shape: (batch_size,)

            # construct prediction sets
            batch_size = images.shape[0]
            for i in range(batch_size):
                # select indices whose scores <= t_cal
                selected_indices = (scores[i] <= t_cal).nonzero(as_tuple=True)[0]

                if len(selected_indices) > 0:
                    pred_label = sorted_indices[i][selected_indices].cpu().tolist()
                    pred_prob = sorted_softmax[i][selected_indices].cpu().tolist()
                    real_prob = probs[i, pred_label].cpu().tolist()
                else:
                    pred_label = []
                    pred_prob = []
                    real_prob = []

                saps.append(pred_prob)
                saps_labels.append(pred_label)
                real_probs.append(real_prob)
                labels.append(true_labels[i].item())

    return saps, saps_labels, labels, real_probs


def saps_classification_imagenet_real(model, dataloader, t_cal, lambda_=0.1, device='cpu'):
    saps = []  # probability set
    saps_labels = []  # label set indicated to the probability set
    labels = []  # true label
    real_probs = []  # real probability of prediction set
    with torch.no_grad():
        for images, true_labels, probs in dataloader:
            images, true_labels, probs = images.to(device), true_labels.to(device), probs.to(device)
            outputs = model(images)
            softmax = torch.softmax(outputs, dim=1)

            # sort probabilities
            sorted_softmax, sorted_indices = torch.sort(softmax, descending=True, dim=1)

            # random variable u(s)
            u = torch.rand(sorted_softmax.shape, device=device)  # Shape: (batch_size, 100)
            # random variable for maximal probabilities
            u_f_max = torch.rand(sorted_softmax.shape[0], device=device).unsqueeze(1)  # Shape: (batch_size, 1)

            # rank of current sorted probability: [1,2,3,...,1000]
            rank = torch.arange(1, sorted_softmax.size(1) + 1, device=device).unsqueeze(0)  # shape: [1, 100]

            # s = f_max + (o-2+u) * lambda
            # scores --> all the label has been calculate as non-top-ranked label now
            f_max = sorted_softmax[:, 0].unsqueeze(1)  # Shape: (batch_size, 1)
            scores = f_max + ((rank - 2).float() + u) * lambda_  # Shape: (batch_size, 100)

            # replace the firt column with u * f_max
            scores[:, 0] = (u_f_max * f_max).squeeze(1)  # Shape: (batch_size,)

            # construct prediction sets
            batch_size = images.shape[0]
            for i in range(batch_size):
                # select indices whose scores <= t_cal
                selected_indices = (scores[i] <= t_cal).nonzero(as_tuple=True)[0]

                if len(selected_indices) > 0:
                    pred_label = sorted_indices[i][selected_indices].cpu().tolist()
                    pred_prob = sorted_softmax[i][selected_indices].cpu().tolist()
                    if torch.all(probs[i] == 0):
                        real_prob = None  # real prob unmarked
                    else:
                        real_prob = probs[i, pred_label].cpu().tolist()
                else:
                    pred_label = []
                    pred_prob = []
                    if torch.all(probs[i] == 0):
                        real_prob = None  # real prob unmarked
                    else:
                        real_prob = []    # real prob marked but APS generate empty set

                saps.append(pred_prob)
                saps_labels.append(pred_label)
                real_probs.append(real_prob)
                labels.append(true_labels[i].item())

    return saps, saps_labels, labels, real_probs


def eval_aps_real_probs(aps_labels, true_labels):
    total_set_size = 0
    coveraged = 0
    for aps_label, true_label in zip(aps_labels, true_labels):
        # cumulate total set size
        total_set_size += len(aps_label)
        # cumulate the predictions sets if it contains true label
        if true_label in aps_label:
            coveraged += 1

    # calculate average values
    samples_amount = len(true_labels)
    average_set_size = total_set_size / samples_amount
    average_coverage = coveraged / samples_amount
    return average_set_size, average_coverage


def hist_cifar10h(all_real_probs_distribution):
    # sort real probability
    sorted_probs = np.sort(all_real_probs_distribution)

    # find the peak value ( the most frequent real probability, frequency)
    y_axis, x_axis = np.histogram(sorted_probs, bins=50)
    peak_y = max(y_axis)  # frequency
    peak_x = x_axis[np.argmax(y_axis)]  # the most frequent real probability

    # draw the histogram
    plt.figure(figsize=(8, 5))
    sb.histplot(sorted_probs, bins=50, kde=True, edgecolor='black', alpha=0.7)
    plt.xlabel("Real Probability")
    plt.ylabel("Frequency")
    plt.title("Distribution of Real Probability after APS")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # mark the peak
    plt.axvline(peak_x, color='red', linestyle='--', label=f'Peak at {peak_x:.4f}, Freq={peak_y}')
    plt.legend()

    plt.show()

    peak_coverage = peak_y / len(sorted_probs) * 100
    print(f" {peak_y}({peak_coverage: .2f}%) Samples Reached the Peak of Real Probability at {peak_x:.4f} ")