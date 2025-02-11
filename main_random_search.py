"""
Run models with random hyperparameter search
"""

import numpy as np
from argparse import ArgumentParser

from skeleton.parameters import get_survival_models, start_new_project
from skeleton.data_processing import SurvData
from skeleton.longitudinal_training import train_long_model
from skeleton.survival_models import train_surv_model
from skeleton.evaluation import evaluate_survival
from skeleton.utils import save_model, load_model, get_metrics_mean


def main(args, seed):
    """
    Run random search
    :param args: parsed arguments
    :param seed: random seed
    :return: results (dict)
    """
    project_dir = f"./logs/{args.project}/task_{args.task}"
    data = SurvData(args.filepath, train_landmarking=args.train_landmarking, sets=args.train_sets,
                    cross_validation=args.cross_validation, val_size=0.2,
                    normalization=args.normalization, missing_impute=args.missing_impute, seed=seed)
    print(f"Data loaded, saving project logs in: {project_dir}")
    param_sets = start_new_project(args, project_dir, seed=seed, random_search=args.num_search)
    survival_models = get_survival_models(args, true_rate=data.true_surv is not None)

    for n in range(args.train_sets):
        if n > 0:
            data.next_train_set()
        lm = 36
        data_lm = data.landmark(["train", "val"], lm)
        best_scores = {sm: np.inf for sm in survival_models}
        for p in range(args.num_search):
            print(f"Train set {n}, landmark {lm}, hyperparameter run: {p}")
            param = param_sets.loc[p].to_dict()
            val_score = {}
            long_model = train_long_model(data=data_lm, param=param)
            for survival_model in survival_models:
                name = f"{args.long_model}_{survival_model}_lm{lm}_{n}_p{p}"
                surv_model = train_surv_model(survival_model, long_model, data_lm, param)

                # evaluate model
                metrics = evaluate_survival(long_model, surv_model, data_lm, mode="val")
                if metrics is None:
                    val_score[survival_model] = np.nan
                else:
                    m_avg = metrics.xs("avg", level=1, axis=1).mean()
                    val_score[survival_model] = m_avg["brier_score"]
                    print(f"Model {name} finished with score {val_score[survival_model]}")
                    if val_score[survival_model] < best_scores[survival_model]:
                        best_scores[survival_model] = val_score[survival_model]
                        save_model(
                            model={"name": name, "param": param, "long_model": long_model, "surv_model": surv_model},
                            project_dir=param["project_dir"], filename=f"{n}_{args.long_model}_{survival_model}")
            # write parameters and validation scores to file
            param_sets.at[p, "val_score"] = val_score
        param_sets.to_csv(f"{project_dir}/params_{args.long_model}_{n}.csv")

        print(f"Evaluating best model(s) on test set {n}")
        for survival_model in survival_models:
            try:
                best_model = load_model(project_dir, f"{n}_{args.long_model}_{survival_model}")
            except FileNotFoundError:
                print(f"No best model found for {n}_{args.long_model}_{survival_model}")
            else:
                metrics = evaluate_survival(best_model["long_model"], best_model["surv_model"], data_lm, mode="test")
                metrics.to_csv(f"{project_dir}/metrics/test{n}_{best_model['name']}.csv")
    print("Finished all runs")


if __name__ == '__main__':
    # load arguments
    SEED = 42
    parser = ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--filepath", type=str, default="dataset/df_adni_tadpole.csv")
    parser.add_argument("--task", type=int, default=None)
    parser.add_argument("--num_search", type=int, default=100)
    parser.add_argument("--cross_validation", type=bool, default=True)
    parser.add_argument("--train_sets", type=int, default=10)
    parser.add_argument("--long_model", choices=["baseline", "last_visit", "MFPCA", "RNN", "RNN_long"], default="last_visit")
    parser.add_argument("--surv_model", choices=["All", "CPH", "RSF", "FNN", "True_rate"], default="All")
    parser.add_argument("--missing_impute", choices=["False", "zero", "ffill"], default="ffill")
    parser.add_argument("--normalization", choices=[None, "minmax", "max", "standard"], default="standard")
    parser.add_argument("--train_landmarking", type=str, choices=["strict", "super", "random", "None"], default="strict")
    arguments = parser.parse_args()
    print(arguments)
    main(arguments, SEED)
