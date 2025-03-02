from torch.utils.data import DataLoader
from src.raps import *


def split_data_set_hyp_opt(dataset, random_seed):
    if random_seed is not None:
        torch.manual_seed(random_seed)  # set input as random seed

    # use 10% dataset to optimize hyperparameter
    dataset_length = len(dataset)
    ten_percent_length = len(dataset) // 10
    calib_length = ten_percent_length // 2
    test_length = ten_percent_length - calib_length

    selected_dataset, _ = random_split(dataset, [ten_percent_length, dataset_length - ten_percent_length])
    calib_dataset, test_dataset = random_split(selected_dataset, [calib_length, test_length])
    return calib_dataset, test_dataset


def lambda_optimization_raps(model, dataset, lambda_values, k_reg, device='cpu'):
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
            calib_scores, _ = raps_scores(model, calib_loader, 0.1, current_lambda, k_reg, device)
            t_cal = np.quantile(calib_scores, 1 - 0.1)
            aps, aps_labels, true_labels = raps_classification(model, test_loader, t_cal, current_lambda, k_reg, device)
            avg_set_size, avg_coverage = eval_aps_hyp_opt(aps_labels, true_labels)

            avg_set_sizes.append(avg_set_size)
            avg_coverages.append(avg_coverage)

        mean_set_size = np.mean(avg_set_sizes)
        mean_coverage = np.mean(avg_coverages)
        # select valid lambda with coverage guarantee
        if 0.85 <= mean_coverage < 0.91:
            set_sizes.append(mean_set_size)
            valid_lambdas.append(current_lambda)

    if len(set_sizes) > 0:
        # optimal lambda has the minimal set size
        optimal_lambda_index = np.argmin(set_sizes)
        optimal_lambda = valid_lambdas[optimal_lambda_index]
    else:
        optimal_lambda = None  # No lambda with valid coverage guarantee

    return optimal_lambda


def k_reg_optimization(model, dataset, optimal_lambda, k_reg_values, device='cpu'):
    set_sizes = []
    valid_k_regs = []

    for k in k_reg_values:
        avg_set_sizes = []
        avg_coverages = []

        for i in range(10):
            # run RAPS
            calib_dataset, test_dataset = split_data_set_hyp_opt(dataset, random_seed=i)
            calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)  # set num_workers = 4 while ImageNet
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)    # set num_workers = 4 while ImageNet
            calib_scores, _ = raps_scores(model, calib_loader, 0.1, optimal_lambda, k, device)
            t_cal = np.quantile(calib_scores, 1 - 0.1)
            aps, aps_labels, true_labels = raps_classification(model, test_loader, t_cal, optimal_lambda, k, device)
            avg_set_size, avg_coverage = eval_aps_hyp_opt(aps_labels, true_labels)

            avg_set_sizes.append(avg_set_size)
            avg_coverages.append(avg_coverage)

        mean_set_size = np.mean(avg_set_sizes)
        mean_coverage = np.mean(avg_coverages)
        # select valid k with coverage guarantee
        if 0.88 <= mean_coverage < 0.91:
            set_sizes.append(mean_set_size)
            valid_k_regs.append(k)

    if len(set_sizes) > 0:
        # optimal k_reg has the minimal set size
        optimal_k_index = np.argmin(set_sizes)
        optimal_k = valid_k_regs[optimal_k_index]
    else:
        optimal_k = None  # No k_reg with valid coverage guarantee

    return optimal_k


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
