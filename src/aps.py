from torch.utils.data import random_split
import torch
import numpy as np


# randomly split the dataset
def split_data_set(dataset, random_seed):
    if random_seed is not None:
        torch.manual_seed(random_seed)  # set input as random seed

    # split image set ---> half for calibration data set, half for test data set
    dataset_length = len(dataset)
    calib_length = dataset_length // 2
    test_length = dataset_length - calib_length

    calib_dataset, test_dataset = random_split(dataset, [calib_length, test_length])
    return calib_dataset, test_dataset


# conformal function s(x,y) for aps
def aps_scores(model, dataloader, alpha=0.1, device='cpu'):
    scores = []  # conformal scores of image sets
    labels = []  # true label sets
    with torch.no_grad():
        for images, true_labels in dataloader:
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


def aps_classification(model, dataloader, q_hat, device='cpu'):
    aps = []         # probability set
    aps_labels = []  # label set indicated to the probability set
    labels = []      # true label
    with torch.no_grad():
        for images, true_labels in dataloader:
            images, true_labels = images.to(device), true_labels.to(device)
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
                aps_labels.append(sorted_index[i, :cutoff_index].cpu().tolist())
                labels.append(true_labels[i].item())

    return aps, aps_labels, labels


def eval_aps(aps_labels, true_labels):
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
    print(f"Total set size: {total_set_size}")
    print(f"Total coverage sets: {coveraged}")
    print(f"Total samples amount: {samples_amount}")
    return average_set_size, average_coverage
