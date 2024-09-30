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


def set_ylim_metric(ax, metric):

    return ax


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
    # colorlist = [(0, 158, 115), (230, 159, 0), (86, 180, 233), (204, 121, 167), (0, 114, 178), (213, 94, 0)]
    # colorlist = [(c[0]/255, c[1]/255, c[2]/255) for c in colorlist]
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


def plot_landmarking(df, model, task, metric="auc", title=None):
    fig, axs = plt.subplots(1, len(task), sharey=True, figsize=(20, 8))
    landmarks = df["landmark"].unique()
    df = df.set_index("landmark")
    colorlist = dict(zip(sorted(landmarks), ["orange", "teal", "aqua", "sienna"]))
    axs[0].set_ylabel(metric)
    df = df[(df["long_model"] == model[0]) & (df["surv_model"] == model[1])]
    for j, l in enumerate(task):
        axs[j].set_xlabel("time")
        axs[j].set_title(f'{l} landmarking')
        subset = df.loc[df["task"] == l, metric]
        axs[j] = lm_over_time(subset, axs[j], metric, colors=colorlist)
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.925), ncols=len(labels), fontsize="large")
    fig.suptitle(f"{title}", fontsize="x-large")
    return fig


def plot_all_models(df, long_models, task=0, metric="likelihood"):
    df = df[df["task"] == task]
    fig, axs = plt.subplots(1, len(long_models), sharey=True, figsize=(18, 4))
    colorlist = dict(zip(sorted(df["surv_model"].unique()), ["crimson", "goldenrod", "green", "navy"]))
    for i, lm in enumerate(long_models):
        grouped = df[df["long_model"] == lm].groupby(["surv_model"])
        for name, group in grouped:
            axs[i].set_title(f'{lm} model')
            axs[i].plot(group["landmarks"], group[metric], label=f"{name[0]}", color=colorlist[name[0]])
            axs[i].set_xlabel("prediction horizon")
    axs[0].set_ylabel(metric)
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='right')
    fig.suptitle(f"{metric} for different model combinations")
    fig.show()


def plot_tasks(df, models, metric="likelihood", title="", labels=None):
    fig, axs = plt.subplots(1, len(models), sharey=True, figsize=(15, 5))
    colorlist = dict(zip(sorted(df["task"].unique()), ["orange", "teal", "aqua", "sienna"]))
    # colorlist = dict(zip(sorted(df["task"].unique()), ["crimson", "goldenrod", "green", "navy", "magenta"]))
    axs[0].set_ylabel(metric)
    for i, model in enumerate(models):
        lm, sm = model
        grouped = df[(df["long_model"] == lm) & (df["surv_model"] == sm)].groupby(["task"])
        axs[i].set_title(f'{lm}-{sm} model')
        axs[i].set_xlabel("landmark")
        for tasknr, group in grouped:
            if all(group["lm_strict"].isna()):
                x = group["landmark"]
                y = pd.to_numeric(group[(metric, "avg")])
            else:
                # select points where training landmark equals evaluation landmark
                subset = group[group["landmark"] == group["lm_strict"]]
                x = subset["landmark"]
                y = pd.to_numeric(subset[(metric, "avg")])
            lbl = tasknr[0] if labels is None else labels[tasknr[0]]
            axs[i].plot(x, y, label=lbl, color=colorlist[tasknr[0]], marker="o", linestyle="dashed")
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='right')
    fig.suptitle(f"{title}")
    fig.show()


def plot_tasks2d(df, models, metric="likelihood", title="", labels=None):
    fig, axs = plt.subplots(models.shape[0], models.shape[1], sharex=True, sharey=True, figsize=(9, 9))
    colorlist = dict(zip(sorted(df["task"].unique()), ["orange", "teal", "aqua", "sienna"]))
    # colorlist = dict(zip(sorted(df["task"].unique()), ["crimson", "goldenrod", "green", "navy", "magenta"]))
    for i in range(models.shape[0]):
        axs[i, 0].set_ylabel(metric)
        for j in range(models.shape[1]):
            lm, sm = models.iloc[i, j]
            grouped = df[(df["long_model"] == lm) & (df["surv_model"] == sm)].groupby(["task"])
            axs[i, j].set_title(f'{lm}_{sm} model')
            axs[-1, j].set_xlabel("landmark")
            for tasknr, group in grouped:
                if all(group["lm_strict"].isna()):
                    x = group["landmark"]
                    y = pd.to_numeric(group[(metric, "avg")])
                else:
                    # select points where training landmark equals evaluation landmark
                    subset = group[group["landmark"] == group["lm_strict"]]
                    x = subset["landmark"]
                    y = pd.to_numeric(subset[(metric, "avg")])
                lbl = tasknr[0] if labels is None else labels[tasknr[0]]
                axs[i, j].plot(x, y, label=lbl, color=colorlist[tasknr[0]], marker="o", linestyle="dashed")
    handles, labels = axs[-1, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='right')
    fig.suptitle(f"{title}")
    fig.show()


def plot_params(params, long_model, surv_models, tasks):
    params = params[params["long_model"] == long_model]
    fig, axs = plt.subplots(len(tasks), len(surv_models))
    for i, t in enumerate(tasks):
        for j, surv_model in enumerate(surv_models):
            p = params[params["task"] == t]
            axs[i, j].boxplot()


def simulation_plots(metrics, metric="mse", train_size=500):
    for lm in ["No_lm", "Super", "Random", "Strict"]:
        df = metrics[(metrics["train_size"] == train_size) & (metrics["landmarking"] == lm)]
        df = df[(df["long_model"] != "baseline") & (df["surv_model"] != "True")]
        title = f"Landmarking: {lm} with train size: {train_size}"
        landmark_subplots(df, "surv_model", "long_model", metric=metric, title=title)

    long_model = "RNN"
    surv_model = "FNN"
    df = metrics[(metrics["long_model"] == long_model) & (metrics["surv_model"] == surv_model) & (
                metrics["landmarking"] != "No_lm")]
    title = f"MSE for {long_model}_{surv_model} model"
    landmark_subplots(df, "train_size", "landmarking", metric=metric, title=title)

    long_model = "MFPCA"
    df = metrics[(metrics["long_model"] == long_model) & (metrics["train_size"] == train_size)]
    df = df[(df["landmarking"] != "No_lm") & (df["surv_model"] != "True")]
    title = f"MSE for {long_model} model with train_size {train_size}"
    landmark_subplots(df, "surv_model", "landmarking", metric=metric, title=title)
