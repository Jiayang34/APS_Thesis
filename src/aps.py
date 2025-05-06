from torch.utils.data import random_split
import torch
import numpy as np
from torch.utils.data import DataLoader


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

            batch_size = images.shape[0]
            for i in range(batch_size):
                # select all the labels whose score <= q_hat
                selected_label = scores[i] <= q_hat
                # construct prediction set
                aps.append(sorted_softmax[i][selected_label].cpu().tolist())
                aps_labels.append(sorted_index[i][selected_label].cpu().tolist())
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


def aps_test(model, dataset, device, num_runs=10, alpha=0.1):
    all_avg_set_sizes = []
    all_avg_coverages = []
    print("APS Classification, Start!\n")
    for i in range(num_runs):
        print(f"Running experiment {i + 1}/{num_runs}...")

        # split dataset
        calib_dataset, test_dataset = split_data_set(dataset, random_seed=i)

        # load data set respectively
        calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False) #, num_workers=4
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # , num_workers=4

        # calculate q_hat
        calib_scores, _ = aps_scores(model, calib_loader, alpha, device)
        q_hat = np.quantile(calib_scores, 1 - alpha)  # calculate 1-alpha quantile
        print(f"q_hat = {q_hat}")

        # construct APS
        aps, aps_labels, true_labels = aps_classification(model, test_loader, q_hat, device)

        # evaluate APS
        avg_set_size, avg_coverage = eval_aps(aps_labels, true_labels)
        print(f"Average Prediction Set Size After APS in runs {i + 1}: {avg_set_size}")
        print(f"Average Coverage Rate in runs {i + 1}: {avg_coverage}\n")

        # record current result
        all_avg_set_sizes.append(avg_set_size)
        all_avg_coverages.append(avg_coverage)

    # calculate the final average result
    final_avg_set_size = np.mean(all_avg_set_sizes)
    final_avg_coverage = np.mean(all_avg_coverages)
    final_set_size_std = np.std(all_avg_set_sizes, ddof=0)
    final_coverage_std = np.std(all_avg_coverages, ddof=0)

    print(f"Final Average Prediction Set Size: {final_avg_set_size:.2f} ± {final_set_size_std:.2f}")
    print(f"Final Average Coverage: {final_avg_coverage:.4f} ± {final_coverage_std:.4f}")