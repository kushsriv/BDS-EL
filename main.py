import argparse
import os
import csv
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from dp_utils import add_dp, repeat_runs
import matplotlib.pyplot as plt
import numpy as np

def setup_logging(logfile="experiment.log"):
    logging.basicConfig(
        filename=logfile,
        filemode='a',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Differential Privacy Experiment Framework")
    parser.add_argument('--dataset', type=str, default="KDDTrain+.csv", help="Path to dataset CSV")
    parser.add_argument('--features', type=str, nargs='+', default=None, help="List of features to analyze")
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.1, 0.3, 0.5, 1.0, 2.0], help="Epsilon values")
    parser.add_argument('--sensitivity', type=float, default=1.0, help="Sensitivity value")
    parser.add_argument('--mechanisms', type=str, nargs='+', default=["laplace", "gaussian"], help="DP mechanisms to use")
    parser.add_argument('--runs', type=int, default=10, help="Number of runs for each experiment")
    parser.add_argument('--results_dir', type=str, default="results", help="Directory to save results")
    parser.add_argument('--log', type=str, default="experiment.log", help="Log file path")
    return parser.parse_args()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results_csv(results, filename):
    keys = results[0].keys()
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

def plot_errors(epsilons, error_dict, feature, results_dir):
    plt.figure()
    for mech, errors in error_dict.items():
        plt.plot(epsilons, errors, marker='o', label=mech)
    plt.xlabel("Epsilon")
    plt.ylabel("Error (mean over runs)")
    plt.title(f"Privacy vs Utility: {feature}")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_dir, f"{feature}_results.png"))
    plt.close()

def main():
    args = parse_args()
    setup_logging(args.log)
    ensure_dir(args.results_dir)

    spark = SparkSession.builder.appName("DP Project").getOrCreate()
    df = spark.read.csv(args.dataset, header=True, inferSchema=True)

    if args.features is None:
        features = [
            "src_bytes", "dst_bytes", "count", "srv_count", "dst_host_count", "dst_host_srv_count"
        ]
    else:
        features = args.features

    available_features = [feature for feature in features if feature in df.columns]
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        logging.warning(f"Skipping missing columns: {', '.join(missing_features)}")

    results_summary = []

    for feature in available_features:
        logging.info(f"===== FEATURE: {feature} =====")
        df_feature = df.select(F.col(feature).cast("double").alias(feature)).dropna()
        count_val = df_feature.count()
        if count_val == 0:
            logging.warning(f"No usable numeric rows for feature {feature}; skipping.")
            continue
        true_avg = df_feature.agg(F.avg(feature).alias("true_avg")).first()["true_avg"]
        if true_avg is None:
            logging.warning(f"Could not compute average for feature {feature}; skipping.")
            continue

        for mechanism in args.mechanisms:
            error_means = []
            error_stds = []
            for epsilon in args.epsilons:
                # POST: add noise to true average
                def post_func():
                    return add_dp(true_avg, args.sensitivity, epsilon, mechanism=mechanism)
                post_vals = repeat_runs(post_func, args.runs)
                post_errors = [abs(val - true_avg) for val in post_vals]
                post_mean = np.mean(post_errors)
                post_std = np.std(post_errors)

                # AGG: add noise to sum, then divide
                sum_val = df_feature.agg(F.sum(feature).alias("sum_val")).first()["sum_val"]
                def agg_func():
                    noisy_sum = add_dp(sum_val, args.sensitivity, epsilon, mechanism=mechanism)
                    return noisy_sum / count_val
                agg_vals = repeat_runs(agg_func, args.runs)
                agg_errors = [abs(val - true_avg) for val in agg_vals]
                agg_mean = np.mean(agg_errors)
                agg_std = np.std(agg_errors)

                # SHUFFLE: add noise to each record (approximate, not true shuffle model)
                def shuffle_func():
                    noisy_df = df_feature.rdd.map(lambda row: add_dp(row[0], args.sensitivity, epsilon, mechanism=mechanism)).collect()
                    return np.mean(noisy_df)
                shuffle_vals = repeat_runs(shuffle_func, args.runs)
                shuffle_errors = [abs(val - true_avg) for val in shuffle_vals]
                shuffle_mean = np.mean(shuffle_errors)
                shuffle_std = np.std(shuffle_errors)

                # Log and save results
                result = {
                    "feature": feature,
                    "mechanism": mechanism,
                    "epsilon": epsilon,
                    "post_mean_error": post_mean,
                    "post_std_error": post_std,
                    "agg_mean_error": agg_mean,
                    "agg_std_error": agg_std,
                    "shuffle_mean_error": shuffle_mean,
                    "shuffle_std_error": shuffle_std
                }
                results_summary.append(result)
                logging.info(f"{result}")

            # Plot for this mechanism and feature
            error_dict = {
                'Post': [r['post_mean_error'] for r in results_summary if r['feature']==feature and r['mechanism']==mechanism],
                'Aggregation': [r['agg_mean_error'] for r in results_summary if r['feature']==feature and r['mechanism']==mechanism],
                'Shuffle': [r['shuffle_mean_error'] for r in results_summary if r['feature']==feature and r['mechanism']==mechanism]
            }
            plot_errors(args.epsilons, error_dict, f"{feature}_{mechanism}", args.results_dir)

    # Save all results to CSV
    save_results_csv(results_summary, os.path.join(args.results_dir, "experiment_results.csv"))
    spark.stop()

if __name__ == "__main__":
    main()