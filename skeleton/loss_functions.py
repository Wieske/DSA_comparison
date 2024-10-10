
import torch
from torch import nn


def negative_log_likelihood(predictions, event, event_time_idx):
    """
    Compute the log likelihood loss
    :param predictions: (Tensor)
    :param event: (Tensor, Bool)
    :param event_time_idx: idx of time_range at which event occurs
    :return:
    This function is used to compute the survival loss
    NLL = event * log(P(t=T_i|x_i))     + (1 - event) * log(S(T_i|x_i))
    """
    n = len(event)
    f = torch.zeros(n)
    for i in range(len(event)):
        if event[i] == 1:
            f[i] = predictions[i, event_time_idx[i]]
        else:
            f[i] = torch.sum(predictions[i, event_time_idx[i]+1:])
    f[f == 0] = 1e-8
    f_hat_log = torch.log(f)
    loss = - torch.sum(f_hat_log) / n
    return loss


def longitudinal_loss(long_pred, x):
    """
    Computes the mean squared error between the longitudinal prediction of the model and the data
    :param long_pred: longitudinal prediction(tensor of size [batch, seq_len, num_var]
    :param x: data tensor of size [batch, seq_len, num_var]
    :return: sum of mean squared error
    """
    # remove delta value from x:
    x = x[:, :, 1:]
    # create mask of x with for each id True values up to the previous-to-last measurement
    histmask = ~x[:, :, 0].isnan()
    idx_last = torch.sum(histmask, dim=1) - 1
    histmask[torch.arange(x.shape[0]), idx_last] = False
    # long_pred[i, t, :] predicts x[i, t+1, :] so add a column of False values to shift the histmask
    false_tensor = torch.full([x.shape[0], 1], False, device=x.device)
    futuremask = torch.cat([false_tensor, histmask[:, :-1]], dim=1)
    # compute the mse loss between long_pred and x (comparing long_pred[i, t, :] with x[i, t+1, :] for all i, t
    mse = torch.nn.MSELoss(reduction='mean')(long_pred[histmask], x[futuremask])
    return mse


class TotalLoss(nn.Module):
    """
    Class to compute the loss function
    """
    def __init__(self, time_range, surv_val, long_val, rank_val=0):
        super(TotalLoss, self).__init__()
        self.device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.time_range = torch.tensor(time_range, dtype=torch.float, device=self.device)
        self.surv_val = surv_val
        self.long_val = long_val

    def forward(self, long_pred, surv_pred, data, labels):
        """
        Calculate total loss as a * negative_log_likelihood + b * longitudinal_loss
        :param long_pred: longitudinal prediction(tensor of size [batch, seq_len, num_var]
        :param surv_pred: (P(T=t|x) for t in time_range), so estimation of probability of event at each time point
        :param data: (x) data tensor of size [batch, seq_len, num_var]
        :param labels: (y) tensor of size (ids, 6) with i (0), id_nr (1), seq_len (2), max visit time (3),
                        event indicator (4) and event time (5)
        :return: total loss
        """
        event = torch.tensor(labels["event"].array, device=self.device)
        event_time = torch.tensor(labels["event_time"].array, device=self.device)
        event_time_idx = torch.searchsorted(self.time_range, event_time, right=True) - 1
        zero = torch.tensor(0., device=event.device)
        nll = zero if self.surv_val == 0 else negative_log_likelihood(surv_pred, event, event_time_idx)
        ll = zero if self.long_val == 0 else longitudinal_loss(long_pred, data)
        return self.surv_val * nll + self.long_val * ll
