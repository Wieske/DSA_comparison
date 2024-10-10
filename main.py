"""
Run models with fixed set of parameters
"""

import numpy as np
import pandas as pd
from argparse import ArgumentParser
import os
import time

from skeleton.parameters import get_survival_models, start_new_project
from skeleton.data_processing import SurvData
from skeleton.longitudinal_training import train_long_model
from skeleton.survival_models import train_surv_model
from skeleton.evaluation import evaluate_survival
from skeleton.utils import save_model, get_metrics_mean


def main(args, seed):
    """
    Run random search
    :param args: parsed arguments
    :param seed: random seed
    :return: results (dict)
    """
    project_dir = f"./logs/{args.project}/task_{args.task}"
    data = SurvData(args.filepath, test_file=args.test_file, train_size=args.train_size, test_size=args.test_size,
                    train_landmarking=args.train_landmarking, sets=args.train_sets, cross_validation=args.cross_validation,
                    normalization=args.normalization, missing_impute=args.missing_impute, seed=seed)
    print(f"Data loaded, saving project logs in: {project_dir}")
    param = start_new_project(args, project_dir, seed=seed, nr_long_var=sum(1 for v in data.var if v[:2]=="L_"))
    results = pd.DataFrame(param, index=np.arange(args.train_sets))

    for n in range(args.train_sets):
        if n > 0:
            data.next_train_set()
        if args.train_landmarking == "strict":
            res, train_times = {}, {}
            for lm in data.landmarks:
                data_lm = data.landmark(["train"], lm)
                print(f"Landmark {lm} - run: {n}")
                res[lm], train_times[lm] = train_test_run(args, param, data_lm, n=f"_{n}", lm=f"_lm{lm}")
            res = {sm: get_metrics_mean([res[l][sm] for l in res]) for sm in res[lm]}
            train_times = {sm: np.sum([train_times[l][sm] for l in train_times]) for sm in train_times[lm]}
        else:
            res, train_times = train_test_run(args, param, data, n=f"_{n}")
        results.at[n, "results"] = res
        results.at[n, "train_times"] = train_times
        results.to_csv(f"{project_dir}/params_{args.long_model}.csv")
    print("Finished all runs")


def train_test_run(args, param, data, n='', lm=''):
    results = {}
    train_times = {}
    print(f"Training longitudinal model: {args.long_model}")
    start_time = time.time()
    long_model = train_long_model(data=data, param=param)
    train_times[args.long_model] = time.time() - start_time

    print(f"Training survival models")
    survival_models = get_survival_models(args, true_rate=data.true_surv is not None)
    for survival_model in survival_models:
        name = f"{args.long_model}_{survival_model}{lm}{n}"
        start_time = time.time()
        surv_model = train_surv_model(survival_model, long_model, data, param)
        train_times[survival_model] = time.time() - start_time

        # evaluate model
        metrics = evaluate_survival(long_model, surv_model, data, mode="test")
        if metrics is None:
            results[survival_model] = np.nan
        else:
            metrics.to_csv(f"{param['project_dir']}/metrics/{name}.csv")
            m_avg = metrics.xs("avg", level=1, axis=1).mean()
            results[survival_model] = m_avg.to_dict()
        print(f"{name} finished")
    return results, train_times


if __name__ == '__main__':
    # load arguments
    SEED = 42
    parser = ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--filepath", type=str)
    parser.add_argument("--task", type=int, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--cross_validation", type=bool, default=False)
    parser.add_argument("--train_sets", type=int, default=10)
    parser.add_argument("--long_model", choices=["baseline", "last_visit", "MFPCA", "RNN", "RNN_long"], default="last_visit")
    parser.add_argument("--surv_model", choices=["All", "CPH", "RSF", "FNN", "True_rate"], default="All")
    parser.add_argument("--missing_impute", choices=["False", "zero", "ffill"], default="ffill")
    parser.add_argument("--normalization", choices=[None, "minmax", "max", "standard"], default=None)
    parser.add_argument("--train_landmarking", type=str, choices=["strict", "super", "random", "None"], default="strict")
    arguments = parser.parse_args()
    print(arguments)
    main(arguments, SEED)
