import matplotlib.pyplot as plt
import pandas as pd
from skeleton.results_analysis import combine_files, landmark_subplots


fig_dir = "figures"
sup = False  # create supplementary figures
plt.rcParams.update({'font.size': 12})

# Combine results from experiments into one file
# logs_dir = "logs/"
# combine_files([f"{logs_dir}/Landmarking_s{n}_1000" for n in [1, 2, 3]], savename="landmarking_sim", task_lib=landmarking)
# combine_files([f"{logs_dir}/Landmarking_ADNI"], savename="landmarking_ADNI", task_lib=landmarking)

# Load simulation results
m_sim = pd.read_csv(f"results/Metrics_landmarking_sim.csv", header=[0, 1])
m_sim = m_sim.rename(columns=lambda x: "" if "Unnamed" in x else x, level=1)
m_s3 = m_sim[m_sim["dataset"] == "s3"]

# Load adni results
m_adni = pd.read_csv(f"results/Metrics_landmarking_ADNI.csv", header=[0, 1])
m_adni = m_adni.rename(columns=lambda x: "" if "Unnamed" in x else x, level=1)
# convert months to years in adni for consistency:
m_adni = m_adni.rename(columns=lambda x: int(x)/12 if x in ["12", "24", "36", "48", "60"] else x, level=1)
m_adni["landmark"] = m_adni["landmark"] / 12

# MSE across different synthetic datasets
df = m_sim[(m_sim["task"] == "Strict")]
models = [("baseline", "CPH"), ("baseline", "RSF"), ("last_visit", "RSF"), ("RNN", "FNN")]
f = landmark_subplots(df, "dataset", models, metric="mse", title="MSE across different synthetic datasets")
f.savefig(f"{fig_dir}/sim_scenarios.jpg")

# Average tdAUC with different landmarking methods
df = m_s3[(m_s3["task"] != "Random") & (m_s3["surv_model"].isin(["CPH", "RSF"])) &
          (m_s3["long_model"].isin(["last_visit", "MFPCA", "RNN"]))]
f = landmark_subplots(df, "surv_model", "long_model", "task", metric="mse",
                      title="Average tdAUC with different landmarking methods")
f.savefig(f"{fig_dir}/Landmarking.jpg")

# RNN results at landmark time: 3 years
df = m_s3[m_s3["landmark"] == 3]
f = landmark_subplots(df, "metric", [("RNN", "FNN"), ("RNN_long", "FNN")], "task",
                      metric=["auc", "mse"],
                      title="RNN results at landmark time: 3 years")
f.savefig(f"{fig_dir}/RNN_lm3.jpg")

# ADNI results
f = landmark_subplots(m_adni, "surv_model", "long_model", "task", metric="auc",
                      title="Average tdAUC with different landmarking methods on ADNI")
f.savefig(f"{fig_dir}/LM_adni.jpg")

# ADNI results at landmark time: 3 years
df = m_adni[m_adni["landmark"] == 3]
f = landmark_subplots(df, "metric", [("last_visit", "RSF"), ("RNN_long", "RSF"), ("RNN_long", "FNN")],
                      plot_var="task", metric=["auc", "brier_score"], title="ADNI results at landmark time: 3 years")
f.savefig(f"{fig_dir}/ADNI_lm3.jpg")

# Supplementary results
if sup:
    f = landmark_subplots(m_s3, "landmark", [("RNN", "FNN"), ("RNN_long", "FNN")], "task", metric="auc",
                          title="Comparison of neural network methods")
    f.savefig(f"{fig_dir}/Supplementary/RNN_vs_RNN_long.jpg")
    x_col = "task"
    landmarking = sorted(df[x_col].unique(), key=lambda x: x.lower() if isinstance(x, str) else x)
    for i, xc in enumerate(landmarking):
        # simulation plots
        df = m_s3[(m_s3["task"] == xc) & (m_s3["surv_model"] != "True")]
        title = f"MSE for simulation scenario 3 with {xc} landmarking"
        f = landmark_subplots(df, "surv_model", "long_model", metric="mse", title=title)
        f.savefig(f"{fig_dir}/Supplementary/s3_{xc}_mse.png")
        title = f"AUC for simulation scenario 3 with {xc} landmarking"
        f = landmark_subplots(df, "surv_model", "long_model", metric="auc", title=title)
        f.savefig(f"{fig_dir}/Supplementary/s3_{xc}_auc.png")
        # ADNI plots
        df = m_adni[(m_adni["task"] == xc)]
        title = f"tdAUC for ADNI data with {xc} landmarking"
        f = landmark_subplots(df, "surv_model", "long_model", metric="auc", title=title)
        f.savefig(f"{fig_dir}/Supplementary/ADNI_{xc}_auc.png")
        title = f"Brier score for ADNI data with {xc} landmarking"
        f = landmark_subplots(df, "surv_model", "long_model", metric="brier_score", title=title)
        f.savefig(f"{fig_dir}/Supplementary/ADNI_{xc}_bs.png")
