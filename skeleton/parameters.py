"""
Function to generate random parameters for all models
"""
import os
import numpy as np
import pandas as pd


def get_fixed_param(args, project_dir=None, seed=None, nr_long_var=10):
    """
    Get a fixed set of hyperparameters
    :param args: arguments from experiment script to save to wandb parameters
    :param project_dir
    :param seed: random seed
    :return: dict containing the hyperparameters
    """
    params = {"project_dir": project_dir,
              "seed": seed,
              "results": None,
              "train_times": None,
              **vars(args),
              # Models
              "encoding_perc": 0.95,
              # Neural Networks:
              "encoding_size": nr_long_var + 5,
              "freeze_long": False,
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
              # Random Survival Forest:
              "rsf_n_estimators": 1000,
              "rsf_min_samples_leaf": 16,
              }
    if args.long_model == "RNN_long":
        params["loss_value_surv"] = 0
        params["freeze_long"] = True
    return params


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


def start_new_project(args, project_dir, seed=None, nr_long_var=None):
    params = get_fixed_param(args, project_dir, seed, nr_long_var)
    params["n"] = 0
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(f"{project_dir}/models", exist_ok=True)
    os.makedirs(f"{project_dir}/metrics", exist_ok=True)
    return params
