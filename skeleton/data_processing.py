"""
Data Loader
"""
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from skeleton.parameters import get_times_for_data
from skeleton.evaluation_utils import IPCW


class SurvData:
    """
    Class that transforms and stores the data for use in survival analysis models
    :param string filepath: The path to the CSV file containing the data.
    :param string test_file: (optional) The path to the CSV file containing the test data.
    :param train_size: (optional) reduce size of the training set to this number
    :param float test_size: (optional) The proportion of data to use for testing.
    :param float val_size: (optional) The proportion of the training data to use for validation. Defaults to None.
    :param train_landmarking: landmarking method for training set (
    :param missing_impute: one of [zero, ffill, None] that determines the method for missing data imputation
    :param normalization: one of [max, minmax, standard, None] that determines the method for normalization

    :param int seed: The random seed for data splitting. Defaults to 42.

    Methods:
     - normalize(data): Normalize the input data
    """
    def __init__(self, filepath, test_file=None, train_size=None, test_size=None, val_size=None, train_landmarking=None,
                 sets=1, cross_validation=False, normalization=None, missing_impute="ffill", seed=42):
        self.filename = filepath.split('/')[-1]
        self.train_size = train_size
        self.val_size = val_size
        self.train_landmarking = train_landmarking
        self.cross_validation = cross_validation
        self.missing_impute = missing_impute
        self.normalization = normalization
        self.time_range, self.landmarks, self.eval_time = get_times_for_data(filepath)
        self.seed = seed

        self.X = {}  # type: dict
        self.ids = {}  # type: dict
        rng = np.random.default_rng(seed=seed)

        # load data from csv and transform to numpy arrays:
        df = dataframe_from_csv(filepath)
        self.X_all, self.ids_all = surv_long_to_wide(df, missing_impute=self.missing_impute)
        self.var = [v for v in df.columns if v not in ["id", "event", "event_time", "visit_time"]]

        if test_file is not None:
            df_test = dataframe_from_csv(test_file)
            self.X["test"], self.ids["test"] = surv_long_to_wide(df_test, missing_impute=self.missing_impute)

            if sets > 1:
                n = len(self.ids_all)
                set_idx = np.concatenate((np.repeat(np.arange(sets), n // sets), np.arange(n % sets)))
                self.ids_all["set"] = set_idx
                self.current_set = 0
                self.ids["train"] = self.ids_all.loc[self.ids_all["set"] == 0]
            else:
                self.ids["train"] = self.ids_all
        elif cross_validation:
            n = len(self.ids_all)
            set_idx = np.concatenate((np.repeat(np.arange(sets), n // sets), np.arange(n % sets)))
            rng.shuffle(set_idx)
            self.ids_all["set"] = set_idx
            self.current_set = 0
            self.ids["test"] = self.ids_all.loc[self.ids_all["set"] == 0]
            self.X["test"] = self.X_all[self.ids["test"].index, :, :]
            self.ids["train"] = self.ids_all.loc[self.ids_all["set"] != 0]
        else:
            self.ids["train"], self.ids["test"] = train_test_split(self.ids_all, test_size=test_size, random_state=seed)
            self.X["test"] = self.X_all[self.ids["test"].index, :, :]

        if val_size is not None:
            # split training data in train, val:
            self.ids["train"], self.ids["val"] = train_test_split(self.ids["train"], test_size=val_size, random_state=seed)
            self.X["val"] = self.X_all[self.ids["val"].index, :, :]

        if train_size is not None:
            # reduce training set to train_size
            self.ids["train"] = self.ids["train"].iloc[:train_size, :]

        self.X["train"] = self.X_all[self.ids["train"].index, :, :]

        # normalize data:
        if self.normalization is not None:
            # normalize data with chosen method (don't normalize delta variable)
            norm_fun = self.get_normalization_function(self.X["train"][:, :, 1:], method=normalization)
            for s in self.X.keys():
                self.X[s][:, :, 1:] = norm_fun(self.X[s][:, :, 1:])

        if self.train_landmarking == "super":
            self.make_super_landmarking_set()

        # Get true probabilities
        try:
            self.true_surv = pd.read_csv("dataset/truesurv/" + self.filename, index_col="id")
        except FileNotFoundError:
            self.true_surv = None

    def get_normalization_function(self, x_train, method="minmax"):
        """
        Get normalization function based on characteristics of the training data.
        :param array_like x_train: training data used to determine characteristics
        :param method: method used for normalization, one of [max, minmax, standard] (default=minmax)
        :return: function used to normalize the data
        """
        if method == "max":
            train_var_max = np.nanmax(x_train, axis=(0, 1))
            train_var_max[train_var_max == 0] = 1  # don't divide by 0
            return lambda x: np.divide(x, train_var_max)
        elif method == "minmax":
            train_var_max = np.nanmax(x_train, axis=(0, 1))
            train_var_min = np.nanmin(x_train, axis=(0, 1))
            diff = train_var_max - train_var_min
            diff[diff == 0] = 1  # don't divide by 0
            return lambda x: np.divide(x - train_var_min, diff)
        elif method == "standard":
            trainmean = np.nanmean(x_train, axis=(0, 1))
            trainstd = np.nanstd(x_train, axis=(0, 1))
            trainstd[trainstd == 0] = 1  # don't divide by 0
            return lambda x: np.divide((x - trainmean), trainstd)
        else:
            raise NotImplementedError(f"Chosen normalization method (method=", method, ") is not implemented.")

    def landmark(self, sets, lm_time):
        """
        Return a copy of the data, where the data and ids for sets are landmarked to lm_time
        :param list sets: list containing the sets (train, train_long, val and/or test)
        :param int lm_time: landmark time
        :return:
        """
        lm_data = copy.copy(self)
        lm_data.X = copy.copy(self.X)
        lm_data.ids = copy.copy(self.ids)
        lm_data.landmarks = [lm_time]
        for s in sets:
            meas_times = np.concatenate((np.zeros([self.X[s].shape[0], 1]), np.cumsum(self.X[s][:, :-1, 0], axis=1)),
                                        axis=1)
            id_subset = lm_data.ids[s]["event_time"] > lm_time
            lm_data.X[s] = lm_data.X[s][id_subset, :, :]
            lm_data.X[s][meas_times[id_subset, :] > lm_time, :] = np.nan
            lm_data.ids[s] = lm_data.ids[s].loc[id_subset, :]
        return lm_data

    def next_train_set(self):
        self.current_set += 1
        if self.cross_validation:
            self.ids["test"] = self.ids_all.loc[self.ids_all["set"] == self.current_set]
            self.X["test"] = self.X_all[self.ids["test"].index, :, :]
            self.ids["train"] = self.ids_all.loc[self.ids_all["set"] != self.current_set]

            if self.val_size is not None:
                # split training data in train, val:
                self.ids["train"], self.ids["val"] = train_test_split(self.ids["train"], test_size=self.val_size,
                                                                      random_state=self.seed)
                self.X["val"] = self.X_all[self.ids["val"].index, :, :]
            self.X["train"] = self.X_all[self.ids["train"].index, :, :]

            if self.normalization is not None:
                norm_fun = self.get_normalization_function(self.X["train"][:, :, 1:], method=self.normalization)
                for s in self.X.keys():
                    self.X[s][:, :, 1:] = norm_fun(self.X[s][:, :, 1:])
        else:
            self.ids["train"] = self.ids_all.loc[self.ids_all["set"] == self.current_set]
            if self.train_size is not None:
                # reduce training set to train_size
                self.ids["train"] = self.ids["train"].iloc[:self.train_size, :]
            self.X["train"] = self.X_all[self.ids["train"].index, :, :]

        if self.train_landmarking == "super":
            self.make_super_landmarking_set()

    def make_super_landmarking_set(self):
        meas_times = np.concatenate((np.zeros([self.X["train"].shape[0], 1]),
                                     np.cumsum(self.X["train"][:, :-1, 0], axis=1)), axis=1)
        x = []
        ids = []
        for lm_time in self.landmarks:
            id_subset = self.ids["train"]["event_time"] > lm_time
            x_lm = self.X["train"][id_subset, :, :].copy()
            x_lm[meas_times[id_subset, :] > lm_time, :] = np.nan
            ids.append(self.ids["train"].loc[id_subset, :])
            x.append(x_lm)
        self.X["train"] = np.concatenate(x, axis=0)
        self.ids["train"] = pd.concat(ids, axis=0)


def dataframe_from_csv(data_file, variables=None):
    """
    Read csv data and select relevant variables
    :param string data_file: The path to the CSV file containing the data.Must contain columns:
    'id', 'event', 'event_time' and 'visit_time'
    :param variables:   random to create one randomly generated variable,
                        list of variables to select only those variables from the dataset
                        None (default) to either select variables based on data_file name or use all variables
    :return: pandas dataframe
    """
    # load data
    df = pd.read_csv(data_file)

    # define variables and columns
    if variables == "random":
        var_bl = []
        var_time = ["random_var"]
        df["random_var"] = np.random.rand(len(df))
    elif variables is not None:
        var_bl = []
        var_time = variables
    elif "adni" in data_file:
        var_bl = ["APOE4", "PTEDUCAT", "PTETHCAT_NotHL", "PTGENDER_Male", "PTMARRY_Married", "PTMARRY_Never",
                  "PTMARRY_Widowed", "PTRACCAT_Asian", "PTRACCAT_Black", "PTRACCAT_Other", "PTRACCAT_More",
                  "PTRACCAT_White"]
        var_time = ["AGE_t", "ADAS11", "ADAS13", "CDRSB", "Entorhinal", "Fusiform", "Hippocampus", "ICV", "MMSE",
                    "MidTemp", "RAVLT_forgetting", "RAVLT_immediate", "RAVLT_learning", "RAVLT_perc_forgetting",
                    "Ventricles", "WholeBrain"]
    elif "pbc2" in data_file:
        var_bl = ["drug", "age", "sex", "ascites", "hepatomegaly", "spiders"]
        var_time = ['edema', 'serBilir', 'serChol', 'albumin', 'alkaline', 'SGOT', 'platelets', 'prothrombin',
                    'histologic']
        df.rename(columns={"tte": "event_time", "times": "visit_time", "label": "event"}, inplace=True)
    elif "simdata" in data_file:
        var_bl = [var for var in df.columns.to_list() if "bcov" in var]
        var_time = [var for var in df.columns.to_list() if "lcov" in var]
    else:
        print("No specific set of variables given, using all variables")
        var_bl = []
        var_time = [v for v in df.columns if v not in ["id", "event", "event_time", "visit_time"]]

    columns = ["id", "event", "event_time", "visit_time"] + var_bl + var_time
    rename_base = {v: "B_" + v for v in var_bl}
    rename_long = {v: "L_" + v for v in var_time}
    df = df[columns].rename(columns={**rename_long, **rename_base})
    return df


def surv_long_to_wide(df, missing_impute="zero"):
    """
    Transform data from pandas dataframe in long format to 3D numpy array
    :param DataFrame df: a Pandas DataFrame with columns "id", "event", "event_time", "visit_time" and any variables
    :param string missing_impute: The imputation method for missing data. Defaults to "zero".
    :return: data, id_set
    The pandas dataframe should contain the column id, event, event_time, visit_time and a column for each variable
    The dataframe should have one row for each measurement/ visit
    data is 3D numpy array with dimensions (nr_ids, max_seq_len, nr_variables)
    """
    # group data by id
    grouped = df.groupby("id", as_index=False)
    id_info = grouped.agg(seq_len=("id", "size"),
                          max_visit=("visit_time", "max"),
                          event=("event", "max"),
                          event_time=("event_time", "max"))
    var = [v for v in df.columns if v not in ["id", "event", "event_time", "visit_time"]]

    # compute variable median for ffill missing data imputation
    var_means = df.median(skipna=True)[var]

    # fill all data in numpy array
    nr_ids = len(grouped)
    max_seq_len = grouped.size()["size"].max()
    nr_var = len(var) + 1  # provided variables + delta variable
    data = np.full([nr_ids, max_seq_len, nr_var], np.nan)
    i = 0
    for id_nr, group in grouped:
        seq_len = len(group)
        if missing_impute == "zero":
            data[i, :seq_len, 1:] = group[var].fillna(value=0)
        elif missing_impute == "ffill":
            data[i, :seq_len, 1:] = group[var].ffill().fillna(value=var_means)
        else:
            data[i, :seq_len, 1:] = group[var]
        data[i, :seq_len, 0] = np.append(np.diff(group["visit_time"]), 0)  # delta variable
        i += 1
    return data, id_info
