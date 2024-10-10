"""
Script for training and evaluating survival model
"""
import numpy as np
import pandas as pd
import torch
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from skeleton.longitudinal_training import predict_long_model, pytorch_train_loop
from skeleton.loss_functions import TotalLoss
from skeleton.longitudinal_models import LongSurvModel
from skeleton.utils import SurvivalFunction


def train_surv_model(surv_model, long_model, data, param):
    """
    Train Random Survival Forest by fitting len(param) models and choosing the one with the highest concorance index
    :param string surv_model: string determining which survival model is used
    :param long_model: longitudinal model (lightning)
    :param SurvData data: instance of SurvData class with data information
    :param dict param: dict with parameters (needs rsf_n_estimators and rsf_min_samples_leaf)
    :return: Random Survival Forest
    """
    if surv_model == "FNN":
        torch.manual_seed(param["seed"])
        loss_fn = TotalLoss(data.time_range, surv_val=1, long_val=0)
        savepath = f"{param['project_dir']}/models/{param['long_model']}_{surv_model}_temp.pth"
        if param["freeze_long"]:
            for p in long_model.long_model.parameters():
                p.requires_grad = False
        pytorch_train_loop(param, data, long_model, loss_fn, savepath, patience=10)
        long_model.load_state_dict(torch.load(savepath))
        return long_model
    elif surv_model == "True_rate":
        if data.true_surv is None:
            return None
        else:
            return SurvivalFunction(times=pd.to_numeric(data.true_surv.columns), surv_pred=data.true_surv.to_numpy(),
                                    ids=data.true_surv.index)
    elif surv_model == "CPH":
        sm = CoxPHSurvivalAnalysis(ties="efron")
    elif surv_model == "RSF":
        sm = RandomSurvivalForest(n_estimators=param["rsf_n_estimators"],
                                  min_samples_split=2*param["rsf_min_samples_leaf"],
                                  min_samples_leaf=param["rsf_min_samples_leaf"],
                                  max_features="sqrt",
                                  n_jobs=-1,
                                  random_state=param["seed"])
    else:
        raise NotImplementedError(f"Chosen survival model: {surv_model}, is not implemented")

    # create labels as structured arrays from ids
    ids = data.ids["train"]
    y_train = Surv.from_arrays(event=ids["event"].to_numpy(), time=ids["event_time"].to_numpy())

    # get encoding from longitudinal model
    x = data.X["train"]
    _, encoding = predict_long_model(long_model, x)

    # Fit and return model
    try:
        sm.fit(encoding, y_train)
    except ValueError:
        sm = None
    return sm


def predict_surv_model(ids, x, surv_model, long_model):
    if isinstance(surv_model, SurvivalFunction):
        return surv_model[surv_model.ids[ids["id"]]]
    elif isinstance(surv_model, LongSurvModel):
        nn_pred, _ = predict_long_model(surv_model, x)
        fail_pred = np.cumsum(nn_pred, axis=-1)
        surv_pred = np.concatenate([np.ones([nn_pred.shape[0], 1]), 1 - fail_pred[:, :-1]], axis=-1)
        return SurvivalFunction(times=surv_model.time_range, surv_pred=surv_pred, ids=ids.index)
    else:
        # use longitudinal model to get the encoding
        _, encoding = predict_long_model(long_model, x)
        # use survival model to make predictions
        pred = surv_model.predict_survival_function(encoding, return_array=True)
        return SurvivalFunction(times=surv_model.unique_times_, surv_pred=pred, ids=ids.index)
