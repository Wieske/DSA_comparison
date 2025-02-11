"""
Utils file containing several functions
"""
import pickle
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from FDApy.representation.argvals import IrregularArgvals, DenseArgvals
from FDApy.representation.values import IrregularValues
from FDApy.representation.functional_data import IrregularFunctionalData, MultivariateFunctionalData


class SurvivalFunction:
    """ Callable survival function

    Parameters
    ----------
    times: ndarray, shape = (n_times,)
    surv_pred: ndarray, shape = (n_samples, n_times)
    ids:

    Inspired by the StepFunction from the scikit-survival package
    """
    def __init__(self, times, surv_pred, ids=None):
        self.times = np.concatenate([np.array([0]), np.atleast_1d(times)])
        self.surv_pred = np.concatenate([np.ones(len(ids))[:, None], np.atleast_2d(surv_pred)], axis=-1)
        self.ids = ids
        if len(self.times) != np.shape(self.surv_pred)[1]:
            raise ValueError(f"Second dimension of surv_pred should be equal to the number of time points: found "
                             f"{len(self.times)} time points, but shape of surv_pred is {np.shape(self.surv_pred)}")

    def __call__(self, time):
        i = np.searchsorted(self.times, time, side="right") - 1
        # i = np.clip(i, a_min=0, a_max=None)
        if np.size(i) == self.surv_pred.shape[0]:
            return self.surv_pred[np.arange(len(i)), i]
        elif np.size(i) <= len(self.times):
            return self.surv_pred[:, i]
        else:
            raise ValueError(f"Found inconsistent number of time points")

    def __getitem__(self, index):
        sf = copy.copy(self)
        sf.ids = self.ids[index]
        sf.surv_pred = self.surv_pred[index]
        return sf

    def __repr__(self):
        return f'SurvivalFunction(ids={self.ids!r}, times={self.times!r}, survival predictions={self.surv_pred!r})'

    def return_dataframe(self, prepend_col=None):
        if prepend_col is None:
            colnames = self.times
        else:
            colnames = [prepend_col + str(t) for t in self.times]
        return pd.DataFrame(data=self.surv_pred, columns=colnames, index=self.ids)


def save_model(model, project_dir, filename):
    model = model.copy()
    for key in model:
        if isinstance(model[key], torch.nn.Module):
            savepath = Path(f"{project_dir}/models/{filename}_{key}.pth")
            torch.save(model[key], savepath)
            model[key] = savepath
    with open(f"{project_dir}/models/{filename}.pkl", "wb") as f:
        pickle.dump(model, f, -1)


def load_model(project_dir, filename):
    with open(f"{project_dir}/models/{filename}.pkl", "rb") as file:
        model = pickle.load(file)
    for key in model:
        if isinstance(model[key], Path):
            model[key] = torch.load(model[key])
    return model


def fdapy_multivar(x, argvals=None):
    nr_var = x.shape[2]
    meas_times = np.concatenate((np.zeros([x.shape[0], 1]), np.cumsum(x[:, :-1, 0], axis=1)), axis=1)
    missing = np.isnan(x)
    data = []
    for i in range(1, nr_var):  # skip delta variable
        avals = IrregularArgvals({idx: DenseArgvals({"input_dim_0": meas_times[idx, ~missing[idx, :, i]]}) for idx in range(x.shape[0])})
        values = IrregularValues({idx: x[idx, ~missing[idx, :, i], i] for idx in range(x.shape[0])})
        ifd = IrregularFunctionalData(avals, values)
        #if argvals is not None and len(dfd.argvals["input_dim_0"]) != len(argvals["input_dim_0"]):
        #    values = np.full([dfd.n_obs, len(argvals["input_dim_0"])], np.nan)
        #    values[:, np.in1d(argvals["input_dim_0"], dfd.argvals["input_dim_0"])] = dfd.values
        data.append(ifd)
    return MultivariateFunctionalData(data)


def transform_pace(data, ufpca_list, eigenvectors):
    scores = []
    for ufpca, dd in zip(ufpca_list, data.data):
        scores.append(ufpca.transform(data=dd, method="PACE"))
    scores_uni = np.concatenate(scores, axis=1)
    return np.dot(scores_uni, eigenvectors)


def random_cov(n, rng, e_val=0.3, tries=10):
    """
    Generate a random (positive semidefinite) covariance matrix
    :param n: number of variables
    :param rng: numpy random generator
    :param e_val: range for value that is used for covariance between different variables
    :param int tries: number of times to try to find a positive matrix
    :return: ndarray [n, n] covariance matrix
    """
    for i in range(tries):
        sigma = rng.uniform(1, 2, n)
        eta = rng.uniform(-e_val, e_val, [n, n])
        cov = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                if i == j:
                    cov[i, j] = sigma[i] ** 2
                elif j < i:
                    cov[i, j] = cov[j, i]
                else:
                    cov[i, j] = sigma[i] * sigma[j] * eta[i, j]
        if np.all(np.linalg.eigvalsh(cov) >= 0):
            return cov
    raise RuntimeError("Can not find a positive semidefinite covariance matrix, try to lower cov_val")


def get_metrics_mean(ld):
    if np.all([isinstance(d, dict) for d in ld]):
        return {m: np.mean([d[m] for d in ld]) for m in ld[0]}
    else:
        return np.nan


def loguniform(rng, lower, upper, size=1):
    """
    Function to create an array of numbers randomly sampled from a log-uniform distribution
    :param lower: (float) Lower boundary for sampling interval
    :param upper: (float) Upper boundary for sampling interval
    :param size: (int or tuple of ints) Output shape
    :return: out: (ndarray) samples from log-uniform distribution
    """
    return np.exp(rng.uniform(np.log(lower), np.log(upper), size))
