"""
Script for data simulation
"""

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.interpolate import BSpline
from utils import random_cov


class SimulatedDataset:
    """
    Create a simulated dataset based on the multivariate joint model linear
    Based on the description in "Functional survival forests for multivariate longitudinal outcomes: Dynamic prediction
    of Alzheimer’s disease progression" by Jeffrey Lin
    """
    def __init__(self, N, scenario, nr_datasets=1, seed=42):
        self.N = N
        self.scenario = scenario
        self.rng = default_rng(seed=seed)
        self.visit_times = np.arange(0, 10.1, 0.5)
        self.nr_times = len(self.visit_times)
        self.nr_bcov = 2
        self.nr_lcov = 3
        self.df_list = []

        for i in range(nr_datasets):
            self.cens = self.rng.uniform(1, 22, size=self.N)
            if scenario == 4:
                self.nr_bcov = 0
                self.nr_lcov = 10
                self.bcov, self.lcov, self.hazard = self.hazard_bspline(c_long=[0.2, 1, 5, -0.2, -5])
            else:
                self.bcov, self.lcov, self.hazard = self.hazard_FSF()
            self.df, self.true_surv = self.create_dataframe()
            self.df_list.append(self.df)

    def hazard_FSF(self):
        """
        Return the coefficients used in the Functional Survival Forests paper
        :return: dictionary with the coefficients beta0, beta1, betat, alpha, gamma and sigma
        """
        s = [1, 1.5, 2]
        rho = [-0.2, 0.1, -0.3]
        sigma = np.diag(s)
        sigma[0, 1] = sigma[1, 0] = np.sqrt(s[0] * s[1]) * rho[0]
        sigma[0, 2] = sigma[2, 0] = np.sqrt(s[0] * s[2]) * rho[1]
        sigma[1, 2] = sigma[2, 1] = np.sqrt(s[1] * s[2]) * rho[2]
        c = {"beta0": np.array([1.5, 2, 0.5])[None, None, :],
             "beta1": np.array([2, -1, 1])[None, None, :],
             "betat": np.array([1.5, -1, 0.6])[None, None, :],
             "alpha": np.array([0.2, -0.2, 0.4]),
             "gamma": np.array([-4, -2]),
             "sigma": sigma}

        # baseline covariates
        if self.scenario == 2:
            bcov = np.zeros((self.N, 1, self.nr_bcov + 1))
        else:
            bcov = np.zeros((self.N, 1, self.nr_bcov))
        bcov[:, 0, 0] = self.rng.binomial(1, 0.5, size=self.N)
        bcov[:, 0, 1] = self.rng.normal(0, 1, size=self.N)
        if self.scenario == 2:
            # add interaction term bcov1 * bcov2 to baseline covariates that is not used in model specification
            bcov[:, 0, 2] = bcov[:, 0, 0] * bcov[:, 0, 1]
            c["gamma"] = np.array([-4, -2, 4])

        # generate scalar covariate:
        x = self.rng.normal(3, 1, size=[self.N, 1, self.nr_lcov])
        # generate subject-specific random effects:
        b = self.rng.multivariate_normal(np.zeros(self.nr_lcov), c["sigma"], size=[self.N, 1])
        times = self.visit_times[None, :, None]
        if self.scenario == 3:
            cov_x = c["beta0"] + c["beta1"] * x + (c["betat"] + b) * times
        else:
            cov_x = c["beta0"] + c["beta1"] * x + b + c["betat"] * times
        h0 = np.exp(-7)
        hazard = h0 * np.exp(np.dot(bcov, c["gamma"]) + np.dot(cov_x, c["alpha"]))
        # add measurement error to longitudinal covariates
        error = self.rng.normal(0, 1, size=[self.N, self.nr_times, self.nr_lcov])
        lcov = cov_x + error
        return bcov, lcov, hazard

    def hazard_bspline(self, c_long=1):
        # baseline covariates
        bcov = np.zeros((self.N, 1, self.nr_bcov))

        # longitudinal coviariates based on b-splines
        lcov = np.zeros(shape=[self.N, self.nr_times, self.nr_lcov])
        seq_len = self.visit_times[-1]
        degree = 3
        num_coef = degree + 2
        knots = np.linspace(-seq_len, 2 * seq_len, num_coef + degree + 1)

        # create a random positive_semidefinite covariance matrix
        cov_val = 0.3
        cov = random_cov(self.nr_lcov, self.rng, e_val=cov_val)

        # generate random coefficients from multivariate normal distribution
        for n in range(self.N):
            c = self.rng.multivariate_normal(np.zeros(self.nr_lcov), cov, size=num_coef).T
            for i in range(self.nr_lcov):
                bspline = BSpline(knots, c[i], degree)
                lcov[n, :, i] = bspline(self.visit_times)

        # hazard
        c_base = np.ones(self.nr_bcov)
        c_long = np.ones(self.nr_lcov)
        # c_long = np.repeat(c_long, int(self.nr_lcov/len(np.atleast_1d(c_long))+1))[:self.nr_lcov]
        h0 = np.exp(-4)
        hazard = h0 * np.exp(np.dot(bcov, c_base) + np.dot(lcov, c_long))

        # add measurement error to longitudinal covariates
        error = self.rng.normal(0, 1, size=[self.N, self.nr_times, self.nr_lcov])
        lcov += error
        return bcov, lcov, hazard

    def generate_survival_model(self, hazard):
        """ Generate the hazard and survival function
        :param hazard: ndarray of size [N, nr_times] with the hazard
        :return: survival, event_time and event
        """
        # determine the survival based on the hazard:
        surv = np.exp(-np.cumsum(hazard, axis=1))
        # the event happens when random probability u is greater than the survival probability surv
        u = self.rng.uniform(0, 1, size=[self.N, 1])
        # the index of the event time can be found by counting the instances before this is true
        time_idx = np.count_nonzero(surv > u, axis=1)
        # when all instances are nonzero the subject is censored at the last time step ("end of study" censoring)
        # this censoring does not influence (most?) metric results
        cens_end = time_idx == self.nr_times
        event_time = self.visit_times[np.where(time_idx == self.nr_times, self.nr_times - 1, time_idx)]
        # add more censoring by generating random censoring times ("dropout")
        cens_time = self.visit_times[np.searchsorted(self.visit_times, self.cens, side="right") - 1]
        event = np.where(np.logical_or(cens_end, cens_time < event_time), 0, 1)
        event_time = np.minimum(event_time, cens_time)
        return surv, event_time, event

    def create_dataframe(self):
        """
        Generate the covariates and survival model and combine all relevant information in a dataframe

        :return: dataframe with id, visit_time, hazard, survival, event_time, event and all covariates
        """
        surv, event_time, event = self.generate_survival_model(self.hazard)

        df = pd.DataFrame({
            "id": np.repeat(np.arange(self.N), self.nr_times),
            "visit_time": np.tile(self.visit_times, self.N),
            "hazard": np.ravel(self.hazard),
            "survival": np.ravel(surv),
            "event_time": np.repeat(event_time, self.nr_times),
            "event": np.repeat(event, self.nr_times),
        })
        # add covariates:
        for i in range(self.nr_lcov):
            df["lcov_" + str(i + 1)] = np.ravel(self.lcov[:, :, i])
        for i in range(self.nr_bcov):
            df["bcov_" + str(i + 1)] = np.repeat(self.bcov[:, 0, i], self.nr_times)

        # save true survival rates for all timepoints (also after event time)
        true_surv = df.pivot(index="id", columns="visit_time", values="survival")
        # remove visit times that occur after the event time:
        df = df.loc[lambda x: x["visit_time"] < x["event_time"], :]
        return df, true_surv


def save_simdatasets(trainseed=42, testseed=0):
    # training sets
    n = 11000
    for s in [1, 2, 3, 4]:
        sim_data = SimulatedDataset(n, scenario=s, seed=trainseed)
        sim_data.df.to_csv(f"../dataset/train/simdata_s{s}.csv")
    # test sets
    n = 3000
    for s in [1, 2, 3, 4]:
        sim_data = SimulatedDataset(n, scenario=s, seed=testseed)
        sim_data.df.to_csv(f"../dataset/test/simdata_s{s}.csv")
        sim_data.true_surv.to_csv(f"../dataset/truesurv/simdata_s{s}.csv")
