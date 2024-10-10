import pandas as pd
import numpy as np


def tadpole_tte(g, features):
    """
    Function to convert Tadpole dataset (as dataframe) to time to event dataframe.
    Example use: adni.groupby("RID").apply(tadpole_tte)
    :param features: list of feature/ column names
    :param g: ADNIMERGE data as pandas dataframe, with at least columns "DX" with diagnosis ("Dementia" for dementia/AD)
    and "M" with months since baseline
    :return: pandas series with column "E" for event (where 1 = conversion to dementia, 0 = censoring) and
    "T" for the time of the event (months since baseline)
    """
    d = {}
    timepoints = sorted(g["M"].values)

    # Add event (E) (where 1=conversion to dementia, 0=censoring) and time of the event (T)
    if "Dementia" in g["DX"].values:
        # find event time: first time point that dementia diagnosis is given
        tte = g.loc[g["DX"] == "Dementia", "M"].min()
        timepoints = [t for t in timepoints if t <= tte]
        for time in timepoints:
            d["event_time", time] = tte
            d["E", time] = 1
            if time < tte:
                d["event", time] = 0
            else:
                d["event", time] = 1
    else:
        # find censoring time: last time point that a (non-dementia) diagnosis is given
        tte = g.loc[g["DX"].notna(), "M"].max()
        timepoints = [t for t in timepoints if t <= tte]
        for time in timepoints:
            d["event_time", time] = tte
            d["event", time] = 0
            d["E", time] = 0
    idx = pd.MultiIndex.from_product([["E", "event", "event_time"] + features, timepoints],
                                     names=["features", "visit_time"])
    if len(timepoints) > 0:
        # add features for all timepoints
        for f in features:
            if f == "delta":
                for i, m in enumerate(timepoints[:-1]):
                    d["delta", m] = timepoints[i + 1] - m
                d["delta", timepoints[-1]] = 0
            else:
                for m in timepoints:
                    d[f, m] = g.loc[g["M"] == m, f].squeeze()
        return pd.Series(d, index=idx)
    else:
        return pd.Series([], dtype=object, index=idx)


# Load tadpole dataset
data = pd.read_csv("dataset/TADPOLE_D1_D2.csv")
# data_dict = pd.read_csv("dataset/TADPOLE_D1_D2_Dict.csv")

# Define features
feat_id = ["RID", "PTID", "VISCODE", "EXAMDATE", "DX_bl", "DXCHANGE", "DX", "M"]
feat_stat = ["AGE", "APOE4", "PTEDUCAT", "PTETHCAT", "PTGENDER", "PTMARRY", "PTRACCAT"]
feat_bio = ["Entorhinal", "Fusiform", "Hippocampus", "ICV", "MidTemp", "Ventricles", "WholeBrain"]
feat_cog = ["CDRSB", "ADAS11", "ADAS13", "MMSE", "RAVLT_forgetting", "RAVLT_immediate", "RAVLT_learning", "RAVLT_perc_forgetting"]
feat = feat_stat + feat_bio + feat_cog

# Extract features from tadpole dataset
data = data[feat_id + feat]
# Remove subjects with AD at baseline
data = data[data["DX_bl"] != "AD"]

# Apply function to transform to time to event data
data_sa = data.groupby("RID").apply(lambda x: tadpole_tte(x, feat + ["delta"])).unstack("features").reset_index()

# Transform categorical features to dummies
feat_categorical = ["PTETHCAT", "PTGENDER", "PTMARRY", "PTRACCAT"]
data_sa[feat_categorical] = data_sa[feat_categorical].replace("Unknown", np.nan)
data_sa = pd.get_dummies(data_sa, columns=feat_categorical, drop_first=True)

# create survival dataset with only id, time and event for FSF
data_sa_surv = data_sa[["RID", "event_time", "E"]].groupby("RID").max().reset_index()

df = data_sa.rename(columns={"RID": "id",
                             "PTETHCAT_Not Hisp/Latino": "PTETHCAT_NotHL",
                             "PTMARRY_Never married": "PTMARRY_Never",
                             "PTRACCAT_Hawaiian/Other PI": "PTRACCAT_Other",
                             "PTRACCAT_More than one": "PTRACCAT_More"})
df["AGE_t"] = df["AGE"] + df["visit_time"] / 12

# Save preprocessed datasets
df.to_csv("dataset/df_adni_tadpole.csv", index=False)
# data_sa.to_csv("dataset/Tadpole_SA_long.csv")
# data_sa_surv.to_csv("dataset/Tadpole_SA_surv.csv")
