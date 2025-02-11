"""
Function to generate random parameters for all models
"""
import os
import numpy as np
import pandas as pd
from skeleton.utils import loguniform


def get_fixed_param(args, project_dir=None, seed=None, nr_long_var=10):
    """
    Get a fixed set of hyperparameters
    :param args: arguments from experiment script to save to parameters
    :param project_dir: project directory
    :param seed: random seed
    :return: dict containing the hyperparameters
    """
    print(f"Using fixed set of hyperparameters")
    params = {"project_dir": project_dir,
              "seed": seed,
              "results": None,
              "train_times": None,
              **vars(args),
              # Models
              "encoding_perc": 0.95,
              # Neural Networks:
              "encoding_size": nr_long_var + 5,
              "long_num_layers": 2,
              "long_dropout": 0.3,
              "surv_hidden_size": 32,
              "surv_num_layers": 2,
              "surv_dropout": 0.3,
              "learning_rate": 1e-3,
              "weight_decay": 1e-5,
              "loss_value_long": 1,
              "loss_value_surv": 1,
              "batch_size": 32,
              "max_epochs": 100,
              "freeze_long": False,
              # Random Survival Forest:
              "rsf_n_estimators": 1000,
              "rsf_min_samples_leaf": 16,
              }
    if args.long_model == "RNN_long":
        params["loss_value_surv"] = 0
        params["freeze_long"] = True
    return params


def generate_random_param(args, n=100, project_dir=None, seed=42):
    """
    Generate n sets of random hyperparameters
    :param args: arguments from experiment script to save to wandb parameters
    :param n: (int, default 100) number of sets to generate
    :param project_dir: project directory
    :param seed: random seed to use for the random generator
    :return: Dataframe containing the hyperparameter sets
    """
    print(f"Generating {n} sets of random hyperparameters")
    rng = np.random.default_rng(seed=seed)
    params = {"run_nr": np.arange(n),
              "project_dir": project_dir,
              "seed": seed,
              "val_score": "ND",
              **vars(args),
              # Models
              "encoding_perc": rng.uniform(0.8, 0.99, size=n),
              # Neural Networks:
              "encoding_size": rng.integers(5, 31, size=n),
              "long_num_layers": rng.integers(1, 4, size=n),
              "long_dropout": rng.uniform(0, 0.5, size=n),
              "surv_hidden_size": np.power(2, rng.integers(4, 7, size=n)),
              "surv_num_layers": rng.integers(1, 4, size=n),
              "surv_dropout": rng.uniform(0, 0.5, size=n),
              "learning_rate": rng.uniform(1e-4, 1e-3, size=n),
              "weight_decay": loguniform(rng, 1e-6, 1e-4, size=n),
              "loss_value_long": 1,
              "loss_value_surv": 1,
              "batch_size": 32,
              "max_epochs": 100,
              "freeze_long": False,
              # Random Survival Forest:
              "rsf_n_estimators": 1000,
              "rsf_min_samples_leaf": rng.integers(10, 20, size=n),
              }
    if args.long_model == "RNN_long":
        params["loss_value_surv"] = 0
        params["freeze_long"] = True
    return pd.DataFrame(params)


def get_times_for_data(data_file):
    """
    Determines the time range, prediction horizon and evaluation times based on the data file
    :param data_file: name of data file
    :return: time_range, landmarks, eval_time
    """
    if "adni" in data_file:
        time_range = np.arange(0, 127, 6)
        landmarks = [12, 24, 36, 48]
        eval_time = [12, 24, 36, 48, 60]
    elif "pcb2" in data_file:
        time_range = np.arange(0, 800, 12)
        landmarks = [60, 3*60, 5*60]
        eval_time = [12, 36, 60, 120]
    else:
        time_range = np.arange(0, 10.6, 0.5)
        landmarks = [1, 2, 3, 4]
        eval_time = [1, 2, 3, 4, 5]
    return time_range, landmarks, eval_time


def get_survival_models(args, true_rate=False):
    """
    Create a list of the used survival models
    :param args: arguments
    :return: list of survival model names
    """
    if args.surv_model == "All":
        survival_models = ["CPH", "RSF"]
        if args.long_model in ["RNN", "RNN_long"]:
            survival_models += ["FNN"]
        if true_rate:
            survival_models += ["True_rate"]
    else:
        survival_models = [args.surv_model]
    return survival_models


def start_new_project(args, project_dir, seed=None, nr_long_var=None, random_search=False):
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(f"{project_dir}/models", exist_ok=True)
    os.makedirs(f"{project_dir}/metrics", exist_ok=True)
    if random_search is False:
        params = get_fixed_param(args, project_dir, seed, nr_long_var)
        params["n"] = 0
    elif random_search == 1:
        params = get_fixed_param(args, project_dir, seed, 16)
        os.makedirs(f"{project_dir}/metrics/val", exist_ok=True)
        params["run_nr"] = 0
        params["val_score"] = "ND"
        params = pd.DataFrame(params, index=[0])
    else:
        params = generate_random_param(args, random_search, project_dir=project_dir, seed=seed)
        os.makedirs(f"{project_dir}/metrics/val", exist_ok=True)
    return params
