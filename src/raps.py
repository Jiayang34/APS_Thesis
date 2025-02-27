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


# conformal function s(x,y) for raps
def raps_scores(model, dataloader, alpha=0.1, lambda_reg=0.1, k_reg=5, device='cpu'):
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

                # calculate p_x(y) + pi_x(y)*u
                if true_label_position == 0:
                    # first softmax is true label -> p_x(y) = 0 ; pi_x(y) = sorted_softmax[0]
                    p_and_pi = u * sorted_softmax[true_label_position].item()
                else:
                    p_and_pi = cumulative_softmax[true_label_position - 1].item() + u * sorted_softmax[
                        true_label_position].item()

                # calculate regularization term: lamba * ( o_x(y) - k_reg)+
                regularization_term = lambda_reg * max(true_label_position + 1 - k_reg, 0)
                conformal_score = p_and_pi + regularization_term

                scores.append(conformal_score)
                labels.append(true_label.item())
    return np.array(scores), np.array(labels)


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