import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(fdir, **kwargs):
    """
    Load and concatenate .csv files containing metric results
    @param fdir: directory containing (only) .csv files containing metric results
    @param kwargs: additional arguments that should be added to metrics
    @return: dataframe with concatenated metrics
    """
    dfs = []
    for f in os.scandir(fdir):
        if f.is_file():
            name = f.name.split(".")[0].split("_")
            metrics = pd.read_csv(f.path, header=[0, 1], index_col=0).reset_index(names="landmark")
            if name[0].startswith("test"):
                # indicates hyperparameter search is used, where the last part of the name will indicate which
                # validation run is chosen for test evaluation ("_p#"), save this and remove from name
                metrics["p"] = name[-1][1:]
                name = name[1:-1]
            metrics["n"] = name[-1]
            # extract longitudinal and survival model from name
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


def load_param(directory):
    """
    Load and concatenate .csv files containing parameter settings
    @param directory: directory containing parameter files
    @return: dataframe containing parameter settings of all models
    """
    params = []
    for f in os.scandir(directory):
        if f.name.startswith("param"):
            param = pd.read_csv(f.path, index_col=0)
            params.append(param)
    if len(params) > 0:
        return pd.concat(params, axis=0)


def combine_files(dirs: list, savename=None, task_lib=None):
    """
    Combine multiple .csv files into a single .csv file for metrics as well as parameters
    @param dirs: list of directories that should be searched for files
    @param savename: (optional) name to use for saving results, if no name is provided results are not saved
    (will save to results/Metrics_{savename}.csv and results/Parameters_{savename}.csv)
    @param task_lib: (optional) dictionary containing numbers and names for the tasks that should be included,
    if not provided these will be inferred from the directories in dirs
    @return: metrics, params
    """
    param = {}
    metrics = []
    for project_dir in dirs:
        # extract dataset and train size from directory name
        project = project_dir.split("/")[-1]
        dataset = project.split("_")[1]
        train_size = project.split("_")[-1]
        if task_lib is None:
            for d in os.scandir(project_dir):
                task = d.name.split("_")[-1]
                param[task] = load_param(d.path)
                metrics.append(load_metrics(f"{d.path}/metrics", dataset=dataset, train_size=train_size, task=task))
        else:
            for task_nr, task in task_lib.items():
                tdir = f"{project_dir}/task_{task_nr}"
                param[task] = load_param(tdir)
                metrics.append(load_metrics(f"{tdir}/metrics", dataset=dataset, train_size=train_size, task=task))
    metrics = pd.concat(metrics)
    params = pd.concat(param, axis=0, names=["task", "run_nr"])
    if savename is not None:
        metrics.to_csv(f"results/Metrics_{savename}.csv", index=False)
        params.to_csv(f"results/Params_{savename}.csv")
    return metrics, params


def get_label(v):
    if isinstance(v, (int, float)):
        return f"landmark {v}"
    elif v in ["No", "Super", "Random", "Strict"]:
        return f"{v} landmarking"
    else:
        return str(v)


def metric_avg(df, ax, task, metric, colors=None, markers=None):
    df = df.set_index("landmark")
    task_list = sorted(df[task].unique(), key=lambda x: x.lower() if isinstance(x, str) else x)
    for t in task_list:
        y = df.loc[(df[task] == t), (metric, "avg")]
        y = y.groupby(y.index).agg(["mean", "std"])
        c = None if colors is None else colors[t]
        m = None if markers is None else markers[t]
        ax.errorbar(y.index, y["mean"], fmt=':', yerr=y["std"], elinewidth=0.7, label=get_label(t), color=c, marker=m)
    return ax


def lm_over_time(df, ax, colors=None, markers=None):
    df = df.drop(columns="avg")
    for lm in sorted(df.index.unique()):
        x = pd.to_numeric(df.columns) + lm
        y = df.loc[lm]
        c = None if colors is None else colors[lm]
        m = None if markers is None else markers[lm]
        ax.errorbar(x, y.mean(), yerr=y.std(), elinewidth=0.7, label=get_label(lm), color=c, marker=m)
    return ax


def task_over_time(df, ax, lm, colors=None, markers=None):
    df = df.drop(columns="avg")
    for t in sorted(df.index.unique()):
        x = pd.to_numeric(df.columns) + lm
        y = df.loc[t]
        c = None if colors is None else colors[t]
        m = None if markers is None else markers[t]
        ax.errorbar(x, y.mean(), yerr=y.std(), elinewidth=0.7, label=get_label(t), color=c, marker=m)
    return ax


def sort_varlist(df, col):
    if isinstance(col, str):
        l = sorted(df[col].unique(), key=lambda x: x.lower() if isinstance(x, str) else x)
        if l == ["CPH", "FNN", "RSF"]:
            l = ["CPH", "RSF", "FNN"]
    else:
        l = col
    return l


def landmark_subplots(df, x_col, y_col, plot_var="landmark", metric="auc", title=None, saveto=None):
    if x_col == "metric":
        x_list = metric
    else:
        x_list = sort_varlist(df, x_col)
    y_list = sort_varlist(df, y_col)
    nx, ny = len(x_list), len(y_list)
    fig, axs = plt.subplots(nx, ny, squeeze=False, sharey="row")
    colordict = dict(zip(sorted(df[plot_var].unique()), ["orange", "teal", "aqua", "sienna"]))
    markerdict = dict(zip(sorted(df[plot_var].unique()), ["o", "v", "x", "d"]))
    for i, xc in enumerate(x_list):
        if x_col == "metric":
            metric = xc
            xc = None
        metric_label = {"auc": "tdAUC", "mse": "MSE", "brier_score": "Brier score"}
        axs[i, 0].set_ylabel(metric_label[metric])
        if metric in ["auc", "c_index"]:
            axs[i, 0].set_ylim(bottom=0.7, top=1)
        elif metric in ["mse", "brier_score"]:
            axs[i, 0].set_ylim(bottom=0, top=0.2)

        for j, yc in enumerate(y_list):
            letter = chr(ord('A') + (i*len(y_list) + j))
            # select subset of the data
            name = f"{yc} - {xc}"
            if xc is None:
                cond1 = True
                name = f"{yc}"
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
                xlabel = "prediction time (years)"
                if plot_var == "landmark":
                    subset = subset.set_index("landmark")
                    axs[i, j] = lm_over_time(subset.loc[:, metric], axs[i, j], colors=colordict, markers=markerdict)
                elif (x_col == "metric") or ("landmark" in [x_col, y_col]):
                    subset = subset.set_index(plot_var)
                    lm = df["landmark"].iloc[0]
                    axs[i, j] = task_over_time(subset.loc[:, metric], axs[i, j], lm, colors=colordict, markers=markerdict)
                else:
                    axs[i, j] = metric_avg(subset, axs[i, j], plot_var, metric, colors=colordict, markers=markerdict)
                    xlabel = "landmark time (years)"
                axs[-1, j].set_xlabel(xlabel)
            else:
                axs[i, j].axis("off")
                axs[i, j].set_title("")
                axs[i-1, j].set_xlabel(xlabel)
                if j < len(y_list):
                    axs[i, j+1].set_ylabel(metric_label[metric])
    handles, labels = axs[-1, -1].get_legend_handles_labels()
    fig.suptitle(f"{title}")
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.95), ncols=len(labels))
    fig.set_layout_engine(layout="constrained", rect=(0, 0, 1, 0.95))
    if saveto is None:
        fig.show()
    else:
        fig.savefig(saveto, bbox_inches='tight', dpi=300)
        plt.close()


def train_times_plot(mean_times, saveto=None):
    fig, axs = plt.subplots(5, 1, sharex="col")
    fig.suptitle("Average training times on the ADNI dataset")
    for i, lm in enumerate(['baseline', 'last_visit', 'MFPCA', 'RNN', 'RNN_long']):
        subset = mean_times.xs(lm, level="long_model")
        if "RNN" not in lm:
            subset = subset
        y_pos = np.arange(len(subset)) + 0.05
        axs[i].barh(y_pos, width=subset["long_training"], height=0.9, align="edge", color="orange",
                    label="Longitudinal model")
        axs[i].set_yticks(y_pos + 0.45, [s + " landmarking" for s in subset.index])
        axs[i].set_title(lm)
        surv_models = ["FNN", "RSF", "CPH"] if "RNN" in lm else ["RSF", "CPH"]
        cdict = {"FNN": "teal", "RSF": "aqua", "CPH": "sienna"}
        h = 0.9 / len(surv_models)
        for j, sm in enumerate(surv_models):
            axs[i].barh(y_pos + j * h, width=subset[sm], left=subset["long_training"], height=h, align="edge",
                        color=cdict[sm], label=f"{sm}")
    axs[-1].set_xlabel("Training time (seconds)")
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.95), ncols=len(labels))
    fig.set_layout_engine(layout="constrained", rect=(0, 0, 1, 0.95))
    if saveto is None:
        fig.show()
    else:
        fig.savefig(saveto, bbox_inches='tight', dpi=300)
        plt.close()


def plot_event_dist(ids):
    x = ids["event_time"].groupby(ids["event"]).value_counts().unstack(level=0).fillna(value=0)
    fig, ax = plt.subplots()
    ax.bar(x.index, x.loc[:, 0], width=2, label="Censored", bottom=np.zeros(len(x)))
    ax.bar(x.index, x.loc[:, 1], width=2, label="Event", bottom=x.loc[:, 0])
    ax.set_xlabel("Event time (months)")
    ax.set_ylabel("Count")
    fig.suptitle("ADNI event distribution")
    fig.legend(loc="right")
    return fig
