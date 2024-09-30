"""
Utils file containing several functions for evaluation (IPCW and metrics)
"""

import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.metrics import roc_auc_score


class IPCW:
    """
    Class to compute Inverse Probability Censoring Weights
    """
    def __init__(self, train_ids):
        """
        Initialize the weights using the training set
        :param train_ids: id_info for the training set, used to determine the censoring probability
        """
        train_event = train_ids["event"]
        train_time = train_ids["event_time"]
        # estimate the probability of censoring using kaplan-meier:
        x, y = kaplan_meier_estimator(train_event == 0, train_time)
        if any(y == 0):
            # fill zeros with last non-zero value to avoid division by zero:
            y[y == 0] = y[y != 0][-1]
        self.times = x
        self.weights = 1.0 / y

    def __call__(self, t):
        """
        Compute IPCW for time(s) t
        :param t: time(s) for which the weights should be computes (scalar, list or numpy array)
        :return: array with the weights for t
        """
        # find the correct index idx such that x[idx] <= test_time < x[idx+1]:
        idx = np.searchsorted(self.times, t, side="right") - 1
        return self.weights[idx]


def compute_survdiff(event, event_time, time_range):
    """
    Function to compute survival estimates for a range of time points
    :param event: event indicator
    :param event_time: event time
    :param time_range:
    :return:
    """
    x, y = kaplan_meier_estimator(event == 1, event_time)
    if any(y == 0):
        # fill zeros with last non-zero value to avoid division by zero:
        print("Found zeros in survival estimates, filling with last available value")
        y[y == 0] = y[y != 0][-1]
    s_times = y[np.searchsorted(x, time_range, side="right") - 1]
    surv_diff = - np.diff(np.r_[1., s_times])
    return surv_diff


def compute_c_index(pred, event, event_time):
    """
    Function to compute the c_index according to the formula:
    C_index = P(S_i(T_i) < S_j(T_i) | T_i <= T_j, D_i = 1) + 0.5 * P(S_i(T_i) = S_j(T_i) | T_i <= T_j, D_i = 1)
    :param pred: SurvivalFunction with the survival probabilities over time for all samples
    :param event: event indicator
    :param event_time: event time
    :return: float c_index
    """
    N = len(event)
    order = np.argsort(event_time)
    pairs = 0
    correct = 0
    for i in range(N):
        idx = order[i]
        if event[idx] == 1:
            T = event_time[idx]
            # compare i to all with a larger event time
            compare_idx = order[order > i]
            # determine survival probabilities
            surv_T = pred(T)
            surv_i = surv_T[idx]
            surv_j = pred(T)[compare_idx]
            # compute number of comparable pairs and number of correct pairs
            pairs += len(compare_idx)
            correct += np.sum(surv_j > surv_i)
            correct += 0.5 * np.sum(surv_j == surv_i)
    if pairs == 0:
        return np.nan
    else:
        return correct / pairs


def compute_likelihood(pred, event, event_time, k=2, zero=1e-6):
    """
    Function to compute the likelihood according to the formula:
    Likelihood = D log f(T) + (1 - D) log S(T)
    where D = event, T = event_time, f = density of the event time, S = survival function
    f is estimated according to the formula in the paper Rindt2022Survival
    :param pred: SurvivalFunction with the survival probabilities over time for all samples
    :param event: event indicator
    :param event_time: event time
    :param k: width of the interval used to approximate f
    :param zero: clipping value of S and f for calculation of the log
    :return: likelihood, percentage of log(zero)
    """
    N = len(event)
    t = pred.times
    surv_pred = pred.surv_pred
    t_idx = np.searchsorted(t, event_time, side="right") - 1
    # estimate f(T)
    idx_min = np.clip(t_idx - k + 1, 0, None)
    idx_plus = np.clip(t_idx + k, None, len(t) - 1)
    f = np.divide(surv_pred[np.arange(N), idx_min] - surv_pred[np.arange(N), idx_plus], t[idx_plus] - t[idx_min])
    # compute S(T)
    S_T = surv_pred[np.arange(N), t_idx]
    # count number of log(0) issues and then clip these values
    nr_zero = event * (f == 0) + (1 - event) * (S_T == 0)
    S_T = np.clip(S_T, zero, None)
    f = np.clip(f, zero, None)
    # compute likelihood
    likelihood = np.log(f[event == 1]).sum() + np.log(S_T[event == 0]).sum()
    return likelihood / N, np.mean(nr_zero)


def compute_auc(risk, event, event_time, t, ipcw_function, weighted=True):
    """
    Compute AUC based on inputs passed to update
    :return: list with AUC for each time point in eval_times and the mean AUC
    """
    if weighted:
        weights = ipcw_function(event_time)
    else:
        weights = np.ones(len(event_time))
    case = (event == 1) * (event_time <= t)
    control = (event_time > t) + (event == 0) * (event_time == t)
    if case.sum() > 0 and control.sum() > 0:
        include = case | control
        pred = risk[include]
        target = case[include]
        w = weights[include]
        try:
            return roc_auc_score(y_true=target, y_score=pred, sample_weight=w)
        except ValueError:
            return np.nan
    else:
        return np.nan


def compute_brier_score(risk, event, event_time, t, ipcw_function, weighted=True):
    """
    Compute Brier score based on inputs passed to update
    :return: list with Brier score for every time point and the integrated Brier score
    """
    case = (event == 1) * (event_time <= t)
    # control = (event_time > t) + (event == 0) * (event_time == t)
    control = (event_time > t)
    if weighted:
        w_case = ipcw_function(event_time)
        w_control = ipcw_function(t)
        N = len(risk)
    else:
        w_case = 1
        w_control = 1
        N = np.sum(case) + np.sum(control)
    bs = case * (1 - risk)**2 * w_case + control * (0 - risk)**2 * w_control
    return np.sum(bs) / N
