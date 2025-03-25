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


# conformal function s(x,y) for saps
def saps_scores(model, dataloader, alpha=0.1, lambda_=0.1, device='cpu'):
    scores = []  # conformal scores of image sets
    labels = []  # true label sets
    with torch.no_grad():
        for images, true_labels in dataloader:
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


def saps_classification(model, dataloader, t_cal, lambda_=0.1, device='cpu'):
    saps = []  # probability set
    saps_labels = []  # label set indicated to the probability set
    labels = []  # true label
    with torch.no_grad():
        for images, true_labels in dataloader:
            images, true_labels = images.to(device), true_labels.to(device)
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
            for i in range(len(images)):
                # select indices whose scores <= t_cal
                selected_indices = (scores[i] <= t_cal).nonzero(as_tuple=True)[0]

                # add selected label to prediction set
                saps.append(sorted_softmax[i][selected_indices].tolist())
                saps_labels.append(sorted_indices[i][selected_indices].tolist())
                labels.append(true_labels[i].item())

    return saps, saps_labels, labels


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
    return average_set_size, average_coverage


def saps_test(model, dataset, device, num_runs=10, alpha=0.1, lambda_=0.1):
    # construct and evaluate repeatedly
    all_avg_set_sizes = []
    all_avg_coverages = []
    print("SAPS Classification, Start!\n")
    for i in range(num_runs):
        print(f"Running experiment {i + 1}/{num_runs}...")

        # split dataset
        calib_dataset, test_dataset = split_data_set(dataset, random_seed=i)

        # load data set respectively
        calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # calculate q_hat
        calib_scores, _ = saps_scores(model, calib_loader, alpha, lambda_, device)
        q_hat = np.quantile(calib_scores, 1 - alpha)  # calculate 1-alpha quantile
        print(f"q_hat = {q_hat}")

        # construct APS
        aps, aps_labels, true_labels = saps_classification(model, test_loader, q_hat, lambda_, device)

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
