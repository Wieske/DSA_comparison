import matplotlib.pyplot as plt
import pandas as pd
from skeleton.results_analysis import combine_files, landmark_subplots, train_times_plot
import ast

fig_dir = "results/figures"
sup = False  # Create and save figures for supplementary material

# Set style parameters
plt.rc('axes', titlesize=12, labelsize=12, grid=True)
plt.rcParams["axes.grid.axis"] = "y"
plt.rc('figure', figsize=[11, 8], labelsize=12, titlesize=14)
plt.rc('font', size=12)
plt.rc('legend', fontsize=12)
plt.rc("lines", markersize=6)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

landmarking = {"0": "No", "1": "Super", "2": "Random", "3": "Strict"}

# Load and combine experiment results
# Run these once to combine all separate metric files of an experiment into one file
# logs_dir = "logs/"
# combine_files([f"{logs_dir}/Landmarking_s{n}_1000" for n in [1, 2, 3, 4]], savename="landmarking_sim", task_lib=landmarking)
# combine_files([f"{logs_dir}/Landmarking_ADNI"], savename="landmarking_ADNI", task_lib=landmarking)
# combine_files([f"{logs_dir}/Randomsearch_ADNI"], savename="Randomsearch_ADNI", task_lib={3: "Random search"})


def read_results(filename):
    """Load .csv file containing metric results"""
    df = pd.read_csv(filename, header=[0, 1])
    df = df.rename(columns=lambda x: "" if "Unnamed" in x else x, level=1)
    if "ADNI" in filename:
        # convert months to years in adni for consistency:
        df = df.rename(columns=lambda x: int(x) / 12 if x in ["12", "24", "36", "48", "60"] else x, level=1)
        df["landmark"] = df["landmark"] / 12
    return df


# Read results data
m_sim = read_results("results/Metrics_landmarking_sim.csv")
m_s3 = m_sim[m_sim["dataset"] == "s3"]
m_adni = read_results("results/Metrics_landmarking_ADNI.csv")
m_adni_rs = read_results("results/Metrics_Randomsearch_ADNI.csv")

# get average scores on adni for quantitative results in abstract:
adni_avg = m_adni.groupby(["long_model", "surv_model", "task"]).agg(auc_mean=(("auc", "avg"), "mean"),
                                                                    auc_std=(("auc", "avg"), "std"),
                                                                    bs_mean=(("brier_score", "avg"), "mean"),
                                                                    bs_std=(("brier_score", "avg"), "std"))

# True results:
df = m_sim[(m_sim["task"] == "Strict") & (m_sim["long_model"] == "baseline") & (m_sim["surv_model"] == "True")]
landmark_subplots(df, "metric", "dataset", metric=["auc", "brier_score"],
                  title="tdAUC and brier score obtained by the true survival probabilities",
                  saveto=f"{fig_dir}/true_metrics.jpg")

# MSE across different synthetic datasets
df = m_sim[(m_sim["task"] == "Strict") & (m_sim["dataset"].isin(["s1", "s2", "s3"]))]
models = [("baseline", "CPH"), ("baseline", "RSF"), ("last_visit", "RSF"), ("RNN", "FNN")]
landmark_subplots(df, "dataset", models, metric="mse", title="MSE across different synthetic datasets",
                  saveto=f"{fig_dir}/sim_scenarios.jpg")

df = m_sim[(m_sim["task"] == "Strict") & (m_sim["dataset"].isin(["s4"]))]
models = [("last_visit", "RSF"), ("MFPCA", "RSF"), ("RNN_long", "FNN"), ("RNN", "FNN")]
landmark_subplots(df, "metric", models, metric=["mse", "auc"], title="MSE and tdAUC on synthetic datasets 4",
                  saveto=f"{fig_dir}/sim_scenarios4.jpg")

# Average tdAUC with different landmarking methods
df = m_s3[(m_s3["task"] != "Random") & (m_s3["surv_model"].isin(["CPH", "RSF"])) &
          (m_s3["long_model"].isin(["last_visit", "MFPCA", "RNN"]))]
landmark_subplots(df, "surv_model", "long_model", "task", metric="mse",
                  title="Average tdAUC with different landmarking methods", saveto=f"{fig_dir}/Landmarking.jpg")

# RNN results at landmark time: 3 years
df = m_s3[m_s3["landmark"] == 3]
landmark_subplots(df, "metric", [("RNN", "FNN"), ("RNN_long", "FNN")], "task", metric=["auc", "mse"],
                  title="RNN results at landmark time: 3 years", saveto=f"{fig_dir}/RNN_lm3.jpg")

# ADNI results
landmark_subplots(m_adni, "surv_model", "long_model", "task", metric="auc",
                  title="Average tdAUC with different landmarking methods on ADNI", saveto=f"{fig_dir}/LM_adni.jpg")

# ADNI results at landmark time: 3 years
df = m_adni[m_adni["landmark"] == 3]
landmark_subplots(df, "metric", [("last_visit", "RSF"), ("RNN_long", "RSF"), ("RNN_long", "FNN")],
                  plot_var="task", metric=["auc", "brier_score"], title="ADNI results at landmark time: 3 years",
                  saveto=f"{fig_dir}/ADNI_lm3.jpg")

# Comparison to random search:
adni_fixed = m_adni[(m_adni["landmark"] == 3) & (m_adni["task"] == "Strict")].copy()
adni_fixed["task"] = "Fixed parameters"
df = pd.concat([m_adni_rs, adni_fixed])
landmark_subplots(df, "metric", [("last_visit", "RSF"), ("RNN_long", "RSF"), ("RNN_long", "FNN"), ("RNN", "FNN")],
                  plot_var="task", metric=["auc", "brier_score"], title="ADNI results at landmark time: 3 years",
                  saveto=f"{fig_dir}/ADNI_RS_vs_fixed.jpg")

# Training times
par = pd.read_csv("results/Params_landmarking_ADNI.csv", index_col=["task", "long_model", "run_nr"])
train_times = par["train_times"].map(lambda x: ast.literal_eval(x))
mean_times = (train_times.groupby(["task", "long_model"]).apply(lambda x: pd.DataFrame(x.tolist()).mean())
              .rename(index=lambda x: x if x in ["CPH", "RSF", "FNN"] else "long_training", level=2)
              .unstack(level=2).swaplevel(0, 1, axis=0).sort_index())
train_times_plot(mean_times, saveto=f"{fig_dir}/ADNI_traintimes.jpg")


# Supplementary results
if sup:
    plt.rc('figure', figsize=[11, 7], labelsize=12, titlesize=14)
    landmark_subplots(m_s3, "landmark", [("RNN", "FNN"), ("RNN_long", "FNN")], "task", metric="auc",
                      title="Comparison of neural network methods",
                      saveto=f"{fig_dir}/Supplementary/RNN_vs_RNN_long.jpg")
    landmarking = ['No', 'Random', 'Strict', 'Super']
    for i, xc in enumerate(landmarking):
        # simulation plots
        df = m_s3[(m_s3["task"] == xc) & (m_s3["surv_model"] != "True")]
        landmark_subplots(df, "surv_model", "long_model", metric="mse",
                          title=f"MSE for simulation scenario 3 with {xc} landmarking",
                          saveto=f"{fig_dir}/Supplementary/s3_{xc}_mse.png")
        landmark_subplots(df, "surv_model", "long_model", metric="auc",
                          title=f"AUC for simulation scenario 3 with {xc} landmarking",
                          saveto=f"{fig_dir}/Supplementary/s3_{xc}_auc.png")
        # ADNI plots
        df = m_adni[(m_adni["task"] == xc)]
        landmark_subplots(df, "surv_model", "long_model", metric="auc",
                          title=f"tdAUC for ADNI data with {xc} landmarking",
                          saveto=f"{fig_dir}/Supplementary/ADNI_{xc}_auc.png")
        landmark_subplots(df, "surv_model", "long_model", metric="brier_score",
                          title=f"Brier score for ADNI data with {xc} landmarking",
                          saveto=f"{fig_dir}/Supplementary/ADNI_{xc}_bs.png")
