from .aps_real_probs import (split_data_set_cifar10h, aps_scores_real_probs, raps_scores_real_probs,
                             saps_scores_real_probs, aps_classification_ground_truth, raps_classification_ground_truth,
                             saps_classification_ground_truth, aps_classification_cifar10h, raps_classification_cifar10h
                             , saps_classification_cifar10h, eval_aps_real_probs, hist_synthetic, scatter_synthetic)
from torch.utils.data import DataLoader
import numpy as np


def aps_cifar10h_hist(model, dataset, device, num_runs=10, alpha=0.1):
    # standard result
    all_avg_set_sizes = []
    all_avg_coverages = []
    all_avg_real_probs = []
    all_q_hat = []
    # histogram: sum of real probs in prediction sets
    all_real_probs_distribution = []
    print(f"APS Classification on CIFAR10-H(alpha={alpha}), Start!\n")

    for i in range(num_runs):
        print(f"Running experiment {i + 1}/{num_runs}...")
        calib_dataset, test_dataset = split_data_set_cifar10h(dataset, random_seed=i)
        calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        calib_scores, _ = aps_scores_real_probs(model, calib_loader, alpha, device)
        q_hat = np.quantile(calib_scores, 1 - alpha)
        aps, aps_labels, true_labels = aps_classification_ground_truth(model, test_loader, q_hat, device)

        avg_set_size, avg_coverage = eval_aps_real_probs(aps_labels, true_labels)
        sum_real_probs = [sum(probs) for probs in aps]
        avg_real_prob = np.mean(sum_real_probs)

        all_avg_set_sizes.append(avg_set_size)
        all_avg_coverages.append(avg_coverage)
        all_avg_real_probs.append(avg_real_prob)
        all_real_probs_distribution.extend(sum_real_probs)
        all_q_hat.append(q_hat)

    # calculate the final average result
    final_avg_set_size = np.mean(all_avg_set_sizes)
    final_avg_coverage = np.mean(all_avg_coverages)
    final_avg_real_prob = np.mean(all_avg_real_probs)
    final_avg_q_hat = np.mean(all_q_hat)
    final_set_size_std = np.std(all_avg_set_sizes, ddof=0)
    final_coverage_std = np.std(all_avg_coverages, ddof=0)
    final_real_prob_std = np.std(all_avg_real_probs, ddof=0)
    final_q_hat_std = np.std(all_q_hat, ddof=0)

    print(f"Final Average q_hat: {final_avg_q_hat:.4f} ± {final_q_hat_std:.4f}")
    print(f"Final Average Prediction Set Size: {final_avg_set_size:.2f} ± {final_set_size_std:.2f}")
    print(f"Final Average Coverage: {final_avg_coverage:.4f} ± {final_coverage_std:.4f}")
    print(f"Final Average Real Probability: {final_avg_real_prob:.4f} ± {final_real_prob_std:.4f}")
    hist_synthetic(all_real_probs_distribution)


def aps_cifar10h_scatter(model, dataset, device, num_runs=10, alpha=0.1):
    # standard result
    all_avg_set_sizes = []
    all_avg_coverages = []
    all_avg_real_probs = []
    all_q_hat = []
    # histogram
    all_real_probs_distribution = []
    # variance-conditional coverage scatter
    all_real_probs = []
    all_pred_probs = []

    print(f"APS Classification on CIFAR10-H(alpha={alpha}), Start!\n")

    for i in range(num_runs):
        print(f"Running experiment {i + 1}/{num_runs}...")
        calib_dataset, test_dataset = split_data_set_cifar10h(dataset, random_seed=i)
        calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        calib_scores, _ = aps_scores_real_probs(model, calib_loader, alpha, device)
        q_hat = np.quantile(calib_scores, 1 - alpha)
        aps, aps_labels, true_labels, real_probs = aps_classification_cifar10h(model, test_loader, q_hat, device)
        all_pred_probs.extend(aps)
        all_real_probs.extend(real_probs)

        avg_set_size, avg_coverage = eval_aps_real_probs(aps_labels, true_labels)
        sum_real_probs = [sum(probs) for probs in real_probs]
        avg_real_prob = np.mean(sum_real_probs)

        all_avg_set_sizes.append(avg_set_size)
        all_avg_coverages.append(avg_coverage)
        all_avg_real_probs.append(avg_real_prob)
        all_real_probs_distribution.extend(sum_real_probs)
        all_q_hat.append(q_hat)

    # calculate the final average result
    final_avg_set_size = np.mean(all_avg_set_sizes)
    final_avg_coverage = np.mean(all_avg_coverages)
    final_avg_real_prob = np.mean(all_avg_real_probs)
    final_avg_q_hat = np.mean(all_q_hat)
    final_set_size_std = np.std(all_avg_set_sizes, ddof=0)
    final_coverage_std = np.std(all_avg_coverages, ddof=0)
    final_real_prob_std = np.std(all_avg_real_probs, ddof=0)
    final_q_hat_std = np.std(all_q_hat, ddof=0)

    print(f"Final Average q_hat: {final_avg_q_hat:.4f} ± {final_q_hat_std:.4f}")
    print(f"Final Average Prediction Set Size: {final_avg_set_size:.2f} ± {final_set_size_std:.2f}")
    print(f"Final Average Coverage: {final_avg_coverage:.4f} ± {final_coverage_std:.4f}")
    print(f"Final Average Real Probability: {final_avg_real_prob:.4f} ± {final_real_prob_std:.4f}")
    scatter_synthetic(all_pred_probs, all_real_probs, all_real_probs_distribution)


def raps_cifar10h_scatter(model, dataset, device, lambda_=0.1, k_reg=2, num_runs=10, alpha=0.1):
    # standard result
    all_avg_set_sizes = []
    all_avg_coverages = []
    all_avg_real_probs = []
    all_q_hat = []
    # histogram
    all_real_probs_distribution = []
    # variance-conditional coverage scatter
    all_real_probs = []
    all_pred_probs = []
    print(f"RAPS Classification on CIFAR10-H(alpha={alpha}), Start!\n")
    for i in range(num_runs):
        print(f"Running experiment {i + 1}/{num_runs}...")
        calib_dataset, test_dataset = split_data_set_cifar10h(dataset, random_seed=i)
        calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        calib_scores, _ = raps_scores_real_probs(model, calib_loader, alpha, lambda_, k_reg, device)
        q_hat = np.quantile(calib_scores, 1 - alpha)
        aps, aps_labels, true_labels, real_probs = raps_classification_cifar10h(model, test_loader, q_hat, lambda_,
                                                                                k_reg, device)
        all_pred_probs.extend(aps)
        all_real_probs.extend(real_probs)
        avg_set_size, avg_coverage = eval_aps_real_probs(aps_labels, true_labels)
        sum_real_probs = [sum(probs) for probs in real_probs]
        avg_real_prob = np.mean(sum_real_probs)  # average real probability

        all_avg_set_sizes.append(avg_set_size)
        all_avg_coverages.append(avg_coverage)
        all_avg_real_probs.append(avg_real_prob)
        all_real_probs_distribution.extend(sum_real_probs)
        all_q_hat.append(q_hat)

    # calculate the final average result
    final_avg_set_size = np.mean(all_avg_set_sizes)
    final_avg_coverage = np.mean(all_avg_coverages)
    final_avg_real_prob = np.mean(all_avg_real_probs)
    final_avg_q_hat = np.mean(all_q_hat)
    final_set_size_std = np.std(all_avg_set_sizes, ddof=0)
    final_coverage_std = np.std(all_avg_coverages, ddof=0)
    final_real_prob_std = np.std(all_avg_real_probs, ddof=0)
    final_q_hat_std = np.std(all_q_hat, ddof=0)

    print(f"Final Average q_hat: {final_avg_q_hat:.4f} ± {final_q_hat_std:.4f}")
    print(f"Final Average Prediction Set Size: {final_avg_set_size:.2f} ± {final_set_size_std:.2f}")
    print(f"Final Average Coverage: {final_avg_coverage:.4f} ± {final_coverage_std:.4f}")
    print(f"Final Average Real Probability: {final_avg_real_prob:.4f} ± {final_real_prob_std:.4f}")
    scatter_synthetic(all_pred_probs, all_real_probs, all_real_probs_distribution)


def raps_cifar10h_hist(model, dataset, device, lambda_=0.1, k_reg=2, num_runs=10, alpha=0.1):
    # standard result
    all_avg_set_sizes = []
    all_avg_coverages = []
    all_avg_real_probs = []
    all_q_hat = []
    # histogram
    all_real_probs_distribution = []
    print(f"RAPS Classification on CIFAR10-H(alpha={alpha}), Start!\n")
    for i in range(num_runs):
        print(f"Running experiment {i + 1}/{num_runs}...")
        calib_dataset, test_dataset = split_data_set_cifar10h(dataset, random_seed=i)
        calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        calib_scores, _ = raps_scores_real_probs(model, calib_loader, alpha, lambda_, k_reg, device)
        q_hat = np.quantile(calib_scores, 1 - alpha)
        aps, aps_labels, true_labels= raps_classification_ground_truth(model, test_loader, q_hat, lambda_,
                                                                                k_reg, device)
        avg_set_size, avg_coverage = eval_aps_real_probs(aps_labels, true_labels)
        sum_real_probs = [sum(probs) for probs in aps]
        avg_real_prob = np.mean(sum_real_probs)  # average real probability

        all_avg_set_sizes.append(avg_set_size)
        all_avg_coverages.append(avg_coverage)
        all_avg_real_probs.append(avg_real_prob)
        all_real_probs_distribution.extend(sum_real_probs)
        all_q_hat.append(q_hat)

    # calculate the final average result
    final_avg_set_size = np.mean(all_avg_set_sizes)
    final_avg_coverage = np.mean(all_avg_coverages)
    final_avg_real_prob = np.mean(all_avg_real_probs)
    final_avg_q_hat = np.mean(all_q_hat)
    final_set_size_std = np.std(all_avg_set_sizes, ddof=0)
    final_coverage_std = np.std(all_avg_coverages, ddof=0)
    final_real_prob_std = np.std(all_avg_real_probs, ddof=0)
    final_q_hat_std = np.std(all_q_hat, ddof=0)

    print(f"Final Average q_hat: {final_avg_q_hat:.4f} ± {final_q_hat_std:.4f}")
    print(f"Final Average Prediction Set Size: {final_avg_set_size:.2f} ± {final_set_size_std:.2f}")
    print(f"Final Average Coverage: {final_avg_coverage:.4f} ± {final_coverage_std:.4f}")
    print(f"Final Average Real Probability: {final_avg_real_prob:.4f} ± {final_real_prob_std:.4f}")
    hist_synthetic(all_real_probs_distribution)


def saps_cifar10h_scatter(model, dataset, device, lambda_=0.1, num_runs=10, alpha=0.1):
    # standard result
    all_avg_set_sizes = []
    all_avg_coverages = []
    all_avg_real_probs = []
    all_q_hat = []
    # histogram
    all_real_probs_distribution = []
    # variance-conditional coverage scatter
    all_real_probs = []
    all_pred_probs = []
    print(f"SAPS Classification on CIFAR10-H(alpha={alpha}), Start!\n")
    for i in range(num_runs):
        print(f"Running experiment {i + 1}/{num_runs}...")
        calib_dataset, test_dataset = split_data_set_cifar10h(dataset, random_seed=i)
        calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        calib_scores, _ = saps_scores_real_probs(model, calib_loader, alpha, lambda_, device)
        q_hat = np.quantile(calib_scores, 1 - alpha)
        aps, aps_labels, true_labels, real_probs = saps_classification_cifar10h(model, test_loader, q_hat, lambda_,
                                                                                device)
        all_pred_probs.extend(aps)
        all_real_probs.extend(real_probs)
        avg_set_size, avg_coverage = eval_aps_real_probs(aps_labels, true_labels)
        sum_real_probs = [sum(probs) for probs in real_probs]
        avg_real_prob = np.mean(sum_real_probs)  # average real probability

        all_avg_set_sizes.append(avg_set_size)
        all_avg_coverages.append(avg_coverage)
        all_avg_real_probs.append(avg_real_prob)
        all_real_probs_distribution.extend(sum_real_probs)
        all_q_hat.append(q_hat)

    # calculate the final average result
    final_avg_set_size = np.mean(all_avg_set_sizes)
    final_avg_coverage = np.mean(all_avg_coverages)
    final_avg_real_prob = np.mean(all_avg_real_probs)
    final_avg_q_hat = np.mean(all_q_hat)
    final_set_size_std = np.std(all_avg_set_sizes, ddof=0)
    final_coverage_std = np.std(all_avg_coverages, ddof=0)
    final_real_prob_std = np.std(all_avg_real_probs, ddof=0)
    final_q_hat_std = np.std(all_q_hat, ddof=0)

    print(f"Final Average q_hat: {final_avg_q_hat:.4f} ± {final_q_hat_std:.4f}")
    print(f"Final Average Prediction Set Size: {final_avg_set_size:.2f} ± {final_set_size_std:.2f}")
    print(f"Final Average Coverage: {final_avg_coverage:.4f} ± {final_coverage_std:.4f}")
    print(f"Final Average Real Probability: {final_avg_real_prob:.4f} ± {final_real_prob_std:.4f}")
    scatter_synthetic(all_pred_probs, all_real_probs, all_real_probs_distribution)


def saps_cifar10h_hist(model, dataset, device, lambda_=0.1, num_runs=10, alpha=0.1):
    # standard result
    all_avg_set_sizes = []
    all_avg_coverages = []
    all_avg_real_probs = []
    all_q_hat = []
    # histogram
    all_real_probs_distribution = []
    print(f"SAPS Classification on CIFAR10-H(alpha={alpha}), Start!\n")
    for i in range(num_runs):
        print(f"Running experiment {i + 1}/{num_runs}...")
        calib_dataset, test_dataset = split_data_set_cifar10h(dataset, random_seed=i)
        calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        calib_scores, _ = saps_scores_real_probs(model, calib_loader, alpha, lambda_, device)
        q_hat = np.quantile(calib_scores, 1 - alpha)
        aps, aps_labels, true_labels = saps_classification_ground_truth(model, test_loader, q_hat, lambda_, device)
        avg_set_size, avg_coverage = eval_aps_real_probs(aps_labels, true_labels)
        sum_real_probs = [sum(probs) for probs in aps]
        avg_real_prob = np.mean(sum_real_probs)  # average real probability

        all_avg_set_sizes.append(avg_set_size)
        all_avg_coverages.append(avg_coverage)
        all_avg_real_probs.append(avg_real_prob)
        all_real_probs_distribution.extend(sum_real_probs)
        all_q_hat.append(q_hat)

    # calculate the final average result
    final_avg_set_size = np.mean(all_avg_set_sizes)
    final_avg_coverage = np.mean(all_avg_coverages)
    final_avg_real_prob = np.mean(all_avg_real_probs)
    final_avg_q_hat = np.mean(all_q_hat)
    final_set_size_std = np.std(all_avg_set_sizes, ddof=0)
    final_coverage_std = np.std(all_avg_coverages, ddof=0)
    final_real_prob_std = np.std(all_avg_real_probs, ddof=0)
    final_q_hat_std = np.std(all_q_hat, ddof=0)

    print(f"Final Average q_hat: {final_avg_q_hat:.4f} ± {final_q_hat_std:.4f}")
    print(f"Final Average Prediction Set Size: {final_avg_set_size:.2f} ± {final_set_size_std:.2f}")
    print(f"Final Average Coverage: {final_avg_coverage:.4f} ± {final_coverage_std:.4f}")
    print(f"Final Average Real Probability: {final_avg_real_prob:.4f} ± {final_real_prob_std:.4f}")
    hist_synthetic(all_real_probs_distribution)


