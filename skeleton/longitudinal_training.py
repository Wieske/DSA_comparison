import numpy as np
import torch
from torch import nn

from skeleton.longitudinal_models import MFPCA_model, LongSurvModel
from skeleton.loss_functions import TotalLoss


def pytorch_train_loop(param, data, model, loss_fn, savepath, patience=10):
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(seed=param["seed"])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=param["learning_rate"], weight_decay=param["weight_decay"])

    x_train = torch.tensor(data.X["train"], dtype=torch.float, device=device)
    ids_train = data.ids["train"]
    data_index = np.arange(x_train.shape[0])
    if "val" in data.X.keys():
        early_stopping = True
        x_val = torch.tensor(data.X["val"], dtype=torch.float, device=device)
    else:
        early_stopping = False

    best_loss = np.inf
    pat = 0
    for ep in range(param["max_epochs"]):
        model.train()
        rng.shuffle(data_index)
        for i in range(0, len(data_index), param["batch_size"]):
            idx = data_index[i:i+param["batch_size"]]
            x, ids = x_train[idx], ids_train.iloc[idx]
            if param["train_landmarking"] == "random":
                seq_length = torch.sum(~x[:, :, 0].isnan(), dim=1)
                mask_idx = torch.tensor([rng.integers(0, int(high)) for high in seq_length], device=device) + 1
                seq_index = torch.tensor(np.indices(x.shape)[1], device=device)
                x[seq_index >= mask_idx[:, None, None]] = np.nan

            optimizer.zero_grad()
            long_pred, surv_pred, _ = model(x)
            loss = loss_fn(long_pred, surv_pred, x, ids)
            loss.backward()
            optimizer.step()

        if early_stopping:
            model.eval()
            with torch.no_grad():
                long_pred, surv_pred, _ = model(x_val)
                val_loss = loss_fn(long_pred, surv_pred, x_val, data.ids["val"]).item()
            if val_loss < best_loss:
                pat = 0
                torch.save(model.state_dict(), savepath)
            else:
                pat += 1
                if pat >= patience:
                    return
    if not early_stopping:
        torch.save(model.state_dict(), savepath)


def train_long_model(data, param):
    if param["long_model"] == "baseline":
        return "baseline"
    elif param["long_model"] == "last_visit":
        return "last_visit"
    elif param["long_model"] == "MFPCA":
        mfpca = MFPCA_model(data, nr_long=param["encoding_perc"])
        mfpca.train(method="PACE")
        return mfpca
    elif param["long_model"] in ["RNN", "RNN_long", "LSTM", "GRU"]:
        torch.manual_seed(param["seed"])
        base_idx = [i + 1 for i, v in enumerate(data.var) if v[:2] == "B_"]
        model = LongSurvModel(data_shape=data.X["train"].shape, time_range=data.time_range, param=param, base_idx=base_idx)
        loss_fn = TotalLoss(data.time_range, surv_val=param["loss_value_surv"], long_val=param["loss_value_long"])
        savepath = f"{param['project_dir']}/models/{param['long_model']}_temp.pth"
        pytorch_train_loop(param, data, model, loss_fn, savepath, patience=10)
        model.load_state_dict(torch.load(savepath))
        return model
    else:
        print("Longitudinal model not found")
        raise NotImplementedError(f"Chosen longitudinal model: {param['long_model']}, is not implemented")


def predict_long_model(model, x):
    """
    :param model: (fitted) longitudinal model
    :param x: ndarray of size [n_subjects, max_seq_len, nr_var] to make predictions for
    :return: survival prediction, longitudinal encoding
    """
    surv_pred = None
    if model == "baseline":
        x = np.nan_to_num(x, nan=0)
        encoding = x[:, 0, 1:]
    elif model == "last_visit":
        last_idx = np.sum(~np.isnan(x[:, :, 0]), axis=1) - 1
        encoding = x[np.arange(x.shape[0]), last_idx, 1:]
    elif isinstance(model, nn.Module):
        model.eval()
        device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        with torch.no_grad():
            _, surv_pred, encoding = model(torch.tensor(x, dtype=torch.float, device=device))
        surv_pred = surv_pred.cpu().numpy()
        encoding = encoding.cpu().numpy()
    else:
        encoding = model.long_predict(x)
    return surv_pred, encoding
