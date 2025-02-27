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
            # logistic value
            outputs = model(images)
            # logistic value -> softmax
            # dim=1 : convert logistic values for all the classes of the example to the softmax
            softmaxs = torch.softmax(outputs, dim=1)

            for softmax, true_label in zip(softmaxs, true_labels):
                # descending sort softmax
                sorted_softmax, sorted_index = torch.sort(softmax, descending=True)

                # get the position of the true label in the sorted softmax
                true_label_position = (sorted_index == true_label).nonzero(as_tuple=True)[0].item()
                # independent random variable u ~ Uniform(0, 1)
                u = np.random.uniform(0, 1)
                # cumulate sorted softmax
                cumulative_softmax = torch.cumsum(sorted_softmax, dim=0)  # dim=0 -> cumulate by raw direction

                if true_label_position == 0:
                    conformal_score = u * sorted_softmax[true_label_position].item()  # first softmax is true label
                else:
                    conformal_score = cumulative_softmax[true_label_position - 1].item() + u * sorted_softmax[
                        true_label_position].item()

                scores.append(conformal_score)
                labels.append(true_label.item())
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
            for softmax, true_label in zip(softmaxs, true_labels):
                sorted_softmax, sorted_index = torch.sort(softmax, descending=True)
                cumulative_softmax = torch.cumsum(sorted_softmax, dim=0)

                # cumulate until meet q_hat and then cut off
                cutoff_index = torch.searchsorted(cumulative_softmax, q_hat, right=True)
                cutoff_index = max(cutoff_index.item(), 1) # make sure cutoff_index >= 1

                # Select all the probabilities and corresponding labels until cut-off index
                prediction_set_prob = sorted_softmax[:cutoff_index].tolist()
                prediction_set_labels = sorted_index[:cutoff_index].tolist()

                aps.append(prediction_set_prob)
                aps_labels.append(prediction_set_labels)
                labels.append(true_label.item())
    return aps, aps_labels, labels

def eval_aps(aps_labels,  true_labels):
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