import numpy as np
import warnings
from joblib import Parallel, delayed
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import check_array_survival
from sksurv.tree._criterion import get_unique_times
from sklearn.tree._tree import DTYPE
from sklearn.ensemble._forest import _get_n_samples_bootstrap, _parallel_build_trees
from sklearn.utils.validation import check_random_state

from skeleton.longitudinal_training import predict_long_model

MAX_INT = np.iinfo(np.int32).max


class RandomSurvivalForest(RandomSurvivalForest):
    def masked_fit(self, X, y, input, long_model, sample_weight=None):
        """Build a forest of survival trees from the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix

        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        self
        """
        self._validate_params()

        X = self._validate_data(X, dtype=DTYPE, accept_sparse="csc", ensure_min_samples=2)
        event, time = check_array_survival(X, y)

        self.n_features_in_ = X.shape[1]
        time = time.astype(np.float64)
        self.unique_times_, self.is_event_time_ = get_unique_times(time, event)
        self.n_outputs_ = self.unique_times_.shape[0]

        y_numeric = np.empty((X.shape[0], 2), dtype=np.float64)
        y_numeric[:, 0] = time
        y_numeric[:, 1] = event.astype(np.float64)

        # Get bootstrap sample size
        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples=X.shape[0], max_samples=self.max_samples)

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                f"n_estimators={self.n_estimators} must be larger or equal to "
                f"len(estimators_)={len(self.estimators_)} when warm_start==True"
            )

        if n_more_estimators == 0:
            warnings.warn("Warm-start fitting without increasing n_estimators does not fit new trees.", stacklevel=2)
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [self._make_estimator(append=False, random_state=random_state) for i in range(n_more_estimators)]

            y_tree = (
                y_numeric,
                self.unique_times_,
                self.is_event_time_,
            )
            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer="threads")(
                delayed(_parallel_masked_trees)(
                    long_model,
                    t,
                    self.bootstrap,
                    input,
                    y_tree,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    n_samples_bootstrap=n_samples_bootstrap,
                )
                for i, t in enumerate(trees)
            )

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score_and_attributes(X, (event, time))

        return self


def _parallel_masked_trees(long_model, tree, bootstrap, X, y, sample_weight, tree_idx, n_trees, verbose, n_samples_bootstrap):
    seq_length = np.sum(~np.isnan(X[:, :, 0]), axis=1)
    mask_idx = np.concatenate([np.random.randint(0, high, (1,)) for high in seq_length]) + 1
    index = np.indices(X.shape)[1]
    X_masked = X.copy()
    X_masked[index >= mask_idx[:, None, None]] = np.nan
    _, encoding = predict_long_model(long_model, X_masked)
    return _parallel_build_trees(tree,
                                 bootstrap,
                                 encoding,
                                 y,
                                 sample_weight,
                                 tree_idx,
                                 n_trees,
                                 verbose=verbose,
                                 n_samples_bootstrap=n_samples_bootstrap)

