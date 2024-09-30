"""
Neural network models
"""
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from FDApy.preprocessing.dim_reduction.fpca import MFPCA
from skeleton.utils import fdapy_multivar, transform_pace


class MFPCA_model(MFPCA):
    def __init__(self, data, nr_long=0.95, traindata="train"):
        self.base_idx = [i + 1 for i, v in enumerate(data.var) if v[:2] == "B_"]
        self.long_idx = [0] + [i + 1 for i, v in enumerate(data.var) if v[:2] == "L_"]
        self.argvals = None
        self.nr_long = nr_long
        self.npc = None
        train_long = data.X[traindata][:, :, self.long_idx]
        self.train_multivar = fdapy_multivar(train_long)
        super().__init__(n_components=[nr_long for _ in range(self.train_multivar.n_functional)], normalize=False)

    def train(self, method="PACE"):
        # with np.errstate(all='ignore'):
        self.fit(self.train_multivar, scores_method=method)

    def long_predict(self, x):
        x_base = np.nan_to_num(x[:, 0, self.base_idx], nan=0)
        x_long = x[:, :, self.long_idx]
        x_multivar = fdapy_multivar(x_long, argvals=self.argvals)
        # scores = self.transform_pace(data=x_multivar, method="PACE")
        scores = transform_pace(data=x_multivar, ufpca_list=self._ufpca_list, eigenvectors=self._scores_eigenvectors)
        if self.npc is None:
            if self.nr_long is None:
                self.npc = len(self.eigenvalues)
            elif self.nr_long < 1:
                exp_variance = np.cumsum(self.eigenvalues) / np.sum(self.eigenvalues)
                self.npc = np.sum(exp_variance < self.nr_long) + 1
            elif self.nr_long < scores.shape[1]:
                self.npc = self.nr_long
            else:
                self.npc = len(self.eigenvalues)
            print(f"MFPCA is using {self.npc} principal components")
        scores = scores[:, :self.npc]
        return np.concatenate([x_base, scores], axis=-1)


def create_fcnet(input_size, output_size, num_layers=1, hidden_size=None, dropout=0, activation=nn.ReLU()):
    """
    Create a fully connected neural network which applies an activation and optionally dropout after every hidden layer
    After the final layers no activation function is used and dropout is not applied
    :param input_size: input size
    :param output_size: output size
    :param num_layers: number of layers (default 1)
    :param hidden_size: number of nodes in the hidden layers (default None)
    :param dropout: dropout probability (applies dropout when value > 0) (default 0)
    :param activation: activation function (default nn.ReLU())
    :return: nn.Sequential Container with the defined modules
    """
    modules = []

    for _ in range(num_layers-1):
        modules.append(nn.Linear(input_size, hidden_size))
        modules.append(activation)
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout))
        input_size = hidden_size

    modules.append(nn.Linear(input_size, output_size))

    return nn.Sequential(*modules)


class LongSurvModel(nn.Module):
    def __init__(self, data_shape, time_range, param, base_idx=None):
        super(LongSurvModel, self).__init__()
        self.input_size = data_shape[2]  # number of variables
        self.base_idx = base_idx  # indices of baseline variables
        self.seq_len = data_shape[1]  # maximum sequence length
        self.time_range = time_range
        self.longitudinal_model = param["long_model"]
        self.long_param = {"hidden_size": param["encoding_size"],
                           "num_layers": param["long_num_layers"],
                           "dropout": param["long_dropout"],
                           "batch_first": True}
        self.surv_param = {"hidden_size": param["surv_hidden_size"],
                           "num_layers": param["surv_num_layers"],
                           "dropout": param["surv_dropout"]}

        # Longitudinal model:
        if self.longitudinal_model == "LSTM":
            self.long_model = nn.LSTM(self.input_size, **self.long_param)
        elif self.longitudinal_model == "GRU":
            self.long_model = nn.GRU(self.input_size, **self.long_param)
        elif self.longitudinal_model in ["RNN", "RNN_long"]:
            self.long_model = nn.RNN(self.input_size, **self.long_param)
        elif self.longitudinal_model == "Linear_last":
            self.long_model = nn.Linear(self.input_size, self.long_param["hidden_size"])
        elif self.longitudinal_model == "Linear_all":
            self.long_model = nn.Linear(self.input_size * self.seq_len, self.long_param["hidden_size"])
        else:
            raise NotImplementedError("Currently only LSTM, GRU and RNN are available as longitudinal models")

        # Longitudinal prediction
        # self.long_predict = nn.Linear(model_param["encoding_size"], self.input_size)
        if self.base_idx is None:
            encode_size = param["encoding_size"] + self.input_size - 1
        else:
            encode_size = param["encoding_size"] + len(self.base_idx)
        self.long_predict = create_fcnet(encode_size, self.input_size - 1, num_layers=2,
                                         hidden_size=encode_size, dropout=param["long_dropout"])
        self.surv_model = create_fcnet(encode_size, len(self.time_range), **self.surv_param)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # compute sequence length for id and change nan to 0
        inputmask = ~x[:, :, 0].isnan()
        seq_length = torch.sum(inputmask, dim=1)
        x = x.nan_to_num(nan=0)
        x_long = x
        if self.base_idx is None:
            x_baseline = x[:, 0, 1:]
        else:
            x_baseline = x[:, 0, self.base_idx]

        if self.longitudinal_model in ["LSTM", "GRU", "RNN", "RNN_long"]:
            # compute rnn on packed history (by packing padded values are not computed, so fewer computations)
            x_long = pack_padded_sequence(x_long, seq_length.cpu(), enforce_sorted=False, batch_first=True)
            hidden, _ = self.long_model(x_long)
            hidden, _ = pad_packed_sequence(hidden, batch_first=True, total_length=self.seq_len)

            # add baseline values (without delta) to each measured timepoint of rnn output
            hidden = torch.cat([x_baseline[:, None, :].expand(-1, hidden.shape[1], -1), hidden], dim=-1)
            # get output of rnn for the last measurement
            encoding = hidden[torch.arange(hidden.shape[0]), seq_length - 1, :]
            # remove last measurement for longitudinal prediction
            hidden[torch.arange(hidden.shape[0]), seq_length - 1, :] = 0
            long_pred = self.long_predict(hidden)
        elif self.longitudinal_model == "Linear_last":
            x_last = x_long[torch.arange(x_long.shape[0]), seq_length - 1, :]
            x_hist = x_long.clone()
            x_hist[torch.arange(x.shape[0]), seq_length - 1, :] = 0
            encoding = self.long_model(x_last)
            long_pred = None
        elif self.longitudinal_model == "Linear_all":
            flat = x_long.flatten()
            encoding = self.long_model(flat)
            long_pred = None
        else:
            raise NotImplementedError("Forward function not implemented for chosen model")

        # output network (multiple causes: substitute for cause-specific subnetworks)
        # combine baseline values (without delta var) with longitudinal encoding
        surv_out = self.surv_model(encoding)
        surv_prob = self.softmax(surv_out)
        return long_pred, surv_prob, encoding
