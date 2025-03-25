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


# conformal function s(x,y) for raps
def raps_scores(model, dataloader, alpha=0.1, lambda_reg=0.1, k_reg=5, device='cpu'):
    scores = []  # conformal scores of image sets
    labels = []  # true label sets
    with torch.no_grad():
        for images, true_labels in dataloader:
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


def raps_classification(model, dataloader, t_cal, lambda_reg=0.1, k_reg=5, device='cpu'):
    raps = []  # probability set
    raps_labels = []  # label set indicated to the probability set
    labels = []  # true label
    with torch.no_grad():
        for images, true_labels in dataloader:
            images, true_labels = images.to(device), true_labels.to(device)
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
            for i, cutoff_index in enumerate(cutoff_indices):
                if cutoff_index.item() > 0:
                    raps.append(sorted_softmax[i, :cutoff_index].tolist())
                    raps_labels.append(sorted_indices[i, :cutoff_index].tolist())
                else:
                    raps.append([])  # RAPS should allow empty set
                    raps_labels.append([])

                labels.append(true_labels[i].item())
    return raps, raps_labels, labels


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


def raps_test(model, dataset, device, num_runs=10, alpha=0.1, lambda_ =0.1, k_reg=5):
    # construct and evaluate repeatedly
    all_avg_set_sizes = []
    all_avg_coverages = []
    print("RAPS Classification, Start!\n")
    for i in range(num_runs):
        print(f"Running experiment {i + 1}/{num_runs}...")

        # split dataset
        calib_dataset, test_dataset = split_data_set(dataset, random_seed=i)

        # load data set respectively
        calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # calculate q_hat
        calib_scores, _ = raps_scores(model, calib_loader, alpha, lambda_, k_reg, device)
        q_hat = np.quantile(calib_scores, 1 - alpha)  # calculate 1-alpha quantile
        print(f"q_hat = {q_hat}")

        # construct APS
        aps, aps_labels, true_labels = raps_classification(model, test_loader, q_hat, lambda_, k_reg, device)

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
