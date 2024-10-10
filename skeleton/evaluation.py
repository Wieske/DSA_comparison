import numpy as np
import pandas as pd
from skeleton.utils import load_model, SurvivalFunction
from skeleton.evaluation_utils import (IPCW, compute_auc, compute_likelihood, compute_survdiff, compute_c_index,
                                       compute_brier_score)
from skeleton.survival_models import predict_surv_model


def evaluate_survival(long_model, surv_model, data, mode="val", weighted=True):
    """
    :param long_model: (fitted) longitudinal model (RNN, MFPCA); output of train_long_model function
    :param surv_model: (fitted) survival model (FNN, RSF); output of train_surv_model function
    :param data: SurvData object
    :param mode: whether to use validation or test data for evaluation
    :param bool weighted: indicates if IPCW weighted versions of metrics should be used (default: True)
    :return: metrics
    """
    if surv_model is None:
        return None
    landmarks = data.landmarks
    eval_time = data.eval_time
    auc = pd.DataFrame(index=landmarks, columns=["avg"] + eval_time)
    bs = pd.DataFrame(index=landmarks, columns=["avg"] + eval_time)
    c_index = pd.DataFrame(index=landmarks, columns=["avg"])
    likelihood = pd.DataFrame(index=landmarks, columns=["avg", "nr_zero"])
    mse = None
    if data.true_surv is not None:
        mse = pd.DataFrame(index=landmarks, columns=["avg"] + eval_time)
        true_surv = SurvivalFunction(times=pd.to_numeric(data.true_surv.columns), surv_pred=data.true_surv.to_numpy(),
                                     ids=data.true_surv.index)
    for lm in landmarks:
        # prepare data: select data with events after lm and set measurements after lm to nan
        ipcw = IPCW(data.ids["train"].loc[data.ids["train"]["event_time"] > lm, :])
        data_lm = data.landmark([mode], lm)
        ids_lm = data_lm.ids[mode]
        pred = predict_surv_model(ids_lm, data_lm.X[mode], surv_model, long_model)
        e_time = np.array([e + lm for e in eval_time])
        surv_lm = pred(lm)
        surv_lm[surv_lm == 0] = np.nan
        event = ids_lm["event"].to_numpy()
        event_time = ids_lm["event_time"].to_numpy()
        surv_diff = compute_survdiff(event, event_time, e_time)
        for eh in e_time:
            risk = 1 - (pred(eh) / surv_lm)
            auc.at[lm, eh - lm] = compute_auc(risk, event, event_time, eh, ipcw, weighted=weighted)
            bs.at[lm, eh - lm] = compute_brier_score(risk, event, event_time, eh, ipcw, weighted=weighted)
            if data.true_surv is not None:
                idx = true_surv.ids.isin(ids_lm["id"])
                true_risk = 1 - true_surv(eh)[idx] / true_surv(lm)[idx]
                mse.at[lm, eh - lm] = np.mean((risk - true_risk)**2)
        # compute summary values
        auc.at[lm, "avg"] = np.sum(surv_diff * auc.loc[lm, eval_time]) / np.sum(surv_diff)
        bs.at[lm, "avg"] = np.trapz(bs.loc[lm, eval_time], e_time) / (e_time[-1] - e_time[0])
        c_index.at[lm, "avg"] = compute_c_index(pred, event, event_time)
        likelihood.at[lm, "avg"], likelihood.at[lm, "nr_zero"] = compute_likelihood(pred, event, event_time)
        if mse is not None:
            mse.at[lm, "avg"] = np.mean(mse.loc[lm, eval_time])
    return pd.concat({"auc": auc, "brier_score": bs, "c_index": c_index, "likelihood": likelihood, "mse": mse}, axis=1)
