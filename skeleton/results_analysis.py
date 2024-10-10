import os
import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(fdir, **kwargs):
    dfs = []
    for f in os.scandir(fdir):
        name = f.name.split(".")[0].split("_")
        metrics = pd.read_csv(f.path, header=[0, 1], index_col=0).reset_index(names="landmark")
        metrics["n"] = name[-1]
        if name[0] == "last" or name[1] == "long":
            metrics["long_model"] = "_".join(name[:2])
            metrics["surv_model"] = name[2]
        else:
            metrics["long_model"] = name[0]
            metrics["surv_model"] = name[1]
        for k, v in kwargs.items():
            metrics[k] = v
        dfs.append(metrics)
    return pd.concat(dfs)


def load_param(directory, long_models, name):
    params = {}
    for long_model in long_models:
        try:
            param = pd.read_csv(f"{directory}/params_{long_model}.csv", index_col=0)
            params[name] = param
        except FileNotFoundError:
            print(f"File {directory}/params_{long_model}.csv not found.")
    if len(params) > 0:
        return pd.concat(params, axis=1)


def combine_files(dirs: list, savename=None, task_lib=None):
    long_models = ["baseline", "last_visit", "MFPCA", "RNN", "RNN_long"]
    param = []
    metrics = []
    for project_dir in dirs:
        project = project_dir.split("/")[-1]
        dataset = project.split("_")[1]
        train_size = project.split("_")[-1]
        for d in os.scandir(project_dir):
            task_nr = d.name.split("_")[-1]
            task = task_nr if task_lib is None else task_lib[task_nr]
            param.append(load_param(d.path, long_models, name=f"{project}_{task}"))
            metrics.append(load_metrics(f"{d.path}/metrics", dataset=dataset, train_size=train_size, task=task))
    metrics = pd.concat(metrics)
    params = pd.concat(param, axis=1)
    if savename is not None:
        metrics.to_csv(f"results/Metrics_{savename}.csv", index=False)
        params.to_csv(f"results/Params_{savename}.csv")
    return metrics, params


def metric_avg(df, ax, task, metric, colors=None):
    df = df.set_index("landmark")
    task_list = sorted(df[task].unique(), key=lambda x: x.lower() if isinstance(x, str) else x)
    ax.grid(visible=True, which="major", axis="y")
    for t in task_list:
        y = df.loc[(df[task] == t), (metric, "avg")]
        y = y.groupby(y.index).agg(["mean", "std"])
        if colors is None:
            ax.errorbar(y.index, y["mean"], fmt='o:', yerr=y["std"], elinewidth=0.7, label=f"{t} landmarking")
        else:
            ax.errorbar(y.index, y["mean"], fmt='o:', yerr=y["std"], elinewidth=0.7, label=f"{t} landmarking", color=colors[t])
    return ax


def lm_over_time(df, ax, colors=None):
    df = df.drop(columns="avg")
    ax.grid(visible=True, which="major", axis="y")
    for lm in sorted(df.index.unique()):
        x = pd.to_numeric(df.columns) + lm
        y = df.loc[lm]
        if colors is None:
            ax.errorbar(x, y.mean(), yerr=y.std(), elinewidth=0.7, label=f"landmark {lm}")
        else:
            ax.errorbar(x, y.mean(), yerr=y.std(), elinewidth=0.7, label=f"landmark {lm}", color=colors[lm])
    return ax


def task_over_time(df, ax, lm, colors=None):
    df = df.drop(columns="avg")
    ax.grid(visible=True, which="major", axis="y")
    for t in sorted(df.index.unique()):
        x = pd.to_numeric(df.columns) + lm
        y = df.loc[t]
        if colors is None:
            ax.errorbar(x, y.mean(), yerr=y.std(), elinewidth=0.7, label=f"{t} landmarking")
        else:
            ax.errorbar(x, y.mean(), yerr=y.std(), elinewidth=0.7, label=f"{t} landmarking", color=colors[t])
    return ax


def landmark_subplots(df, x_col, y_col, plot_var="landmark", metric="auc", title=None):
    if x_col == "metric":
        x_list = metric
    elif isinstance(x_col, str):
        x_list = sorted(df[x_col].unique(), key=lambda x: x.lower() if isinstance(x, str) else x)
    else:
        x_list = x_col
    if isinstance(y_col, str):
        y_list = sorted(df[y_col].unique(), key=lambda x: x.lower() if isinstance(x, str) else x)
    else:
        y_list = y_col
    nx, ny = len(x_list), len(y_list)
    fig, axs = plt.subplots(nx, ny, squeeze=False, sharey="row", figsize=(4*ny+5, 3*nx+5))
    colorlist = ["orange", "teal", "aqua", "sienna"]
    colordict = dict(zip(sorted(df[plot_var].unique()), colorlist))
    for i, xc in enumerate(x_list):
        if x_col == "metric":
            metric = xc
            xc = None
        metric_label = {"auc": "tdAUC", "mse": "MSE", "brier_score": "Brier score"}
        axs[i, 0].set_ylabel(metric_label[metric], fontsize="large")
        if metric in ["auc", "c_index"]:
            axs[i, 0].set_ylim(bottom=0.5, top=1)
        elif metric in ["mse", "brier_score"]:
            axs[i, 0].set_ylim(bottom=0, top=0.25)

        for j, yc in enumerate(y_list):
            letter = chr(ord('A') + (i*len(y_list) + j))
            # select subset of the data
            name = f"{yc} - {xc}"
            if xc is None:
                cond1 = True
            elif isinstance(x_col, str):
                cond1 = (df[x_col] == xc)
            else:
                cond1 = (df["long_model"] == xc[0]) & (df["surv_model"] == xc[1])
                name = f"{xc[0]} - {xc[1]}"
            if isinstance(y_col, str):
                cond2 = (df[y_col] == yc)
            else:
                cond2 = (df["long_model"] == yc[0]) & (df["surv_model"] == yc[1])
                name = f"{yc[0]} - {yc[1]}"
            subset = df.loc[cond1 & cond2, :]
            axs[i, j].set_title(f'({letter}): {name}')

            # create plot:
            if len(subset) > 0:
                axs[-1, j].set_xlabel("prediction time (years)", fontsize="large")
                if plot_var == "landmark":
                    subset = subset.set_index("landmark")
                    axs[i, j] = lm_over_time(subset.loc[:, metric], axs[i, j], colors=colordict)
                elif (x_col == "metric") or ("landmark" in [x_col, y_col]):
                    subset = subset.set_index(plot_var)
                    lm = df["landmark"].iloc[0]
                    axs[i, j] = task_over_time(subset.loc[:, metric], axs[i, j], lm, colors=colordict)
                else:
                    axs[i, j] = metric_avg(subset, axs[i, j], plot_var, metric, colors=colordict)
                    axs[-1, j].set_xlabel("landmark time (years)", fontsize="large")
            else:
                axs[i, j].axis("off")
                axs[i, j].set_title("")
    handles, labels = axs[-1, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.93), ncols=len(labels), fontsize="large")
    fig.suptitle(f"{title}", fontsize="x-large")
    return fig
