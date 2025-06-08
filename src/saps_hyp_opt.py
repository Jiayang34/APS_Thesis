from torch.utils.data import DataLoader
from src.saps import *


def split_data_set_hyp_opt(dataset, random_seed):
    if random_seed is not None:
        torch.manual_seed(random_seed)  # set input as random seed

    # use one fourth of dataset to optimize hyperparameter
    dataset_length = len(dataset)
    ten_percent_length = len(dataset) // 10
    calib_length = ten_percent_length // 2
    test_length = ten_percent_length - calib_length

    selected_dataset, _ = random_split(dataset, [ten_percent_length, dataset_length - ten_percent_length])
    calib_dataset, test_dataset = random_split(selected_dataset, [calib_length, test_length])
    return calib_dataset, test_dataset


def lambda_optimization_saps(model, dataset, lambda_values, device='cpu', alpha=0.1):
    set_sizes = []
    valid_lambdas = []

    for current_lambda in lambda_values:
        avg_set_sizes = []
        avg_coverages = []

        for i in range(10):
            # run RAPS
            calib_dataset, test_dataset = split_data_set_hyp_opt(dataset, random_seed=i)
            calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)  # set num_workers = 4 while ImageNet
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)    # set num_workers = 4 while ImageNet
            calib_scores, _ = saps_scores(model, calib_loader, alpha, current_lambda, device)
            t_cal = np.quantile(calib_scores, 1 - alpha)
            aps, aps_labels, true_labels = saps_classification(model, test_loader, t_cal, current_lambda, device)
            avg_set_size, avg_coverage = eval_aps_hyp_opt(aps_labels, true_labels)

            avg_set_sizes.append(avg_set_size)
            avg_coverages.append(avg_coverage)

        mean_set_size = np.mean(avg_set_sizes)
        mean_coverage = np.mean(avg_coverages)
        # select valid lambda with coverage guarantee
        max_range = 1-alpha+0.01
        min_range = 1-alpha-0.03
        if min_range <= mean_coverage < max_range:
            set_sizes.append(mean_set_size)
            valid_lambdas.append(current_lambda)

    if len(set_sizes) > 0:
        # optimal lambda has the minimal set size
        optimal_lambda_index = np.argmin(set_sizes)
        optimal_lambda = valid_lambdas[optimal_lambda_index]
    else:
        optimal_lambda = None  # No lambda with valid coverage guarantee

    return optimal_lambda


def eval_aps_hyp_opt(aps_labels, true_labels):
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
