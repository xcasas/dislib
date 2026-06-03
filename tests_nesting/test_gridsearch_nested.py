import numpy as np
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from sklearn import datasets
from sklearn.utils import shuffle

import dislib as ds
from dislib.classification import CascadeSVM
from dislib.model_selection import GridSearchCV
from tests import BaseTimedTestCase


@task(returns=bool)
def test_gridsearch_csvm_nested():
    """Tests GridSearchCV fit() using the nested candidate evaluation path."""
    x_np, y_np = datasets.load_iris(return_X_y=True)
    x_np, y_np = shuffle(x_np, y_np, random_state=0)
    x = ds.array(x_np, (60, 4))
    y = ds.array(y_np[:, np.newaxis], (60, 1))

    parameters = {
        "gamma": [0.1, 0.2],
        "c": [0.1, 0.2],
    }
    csvm = CascadeSVM(check_convergence=True, max_iter=5, random_state=0)
    searcher = GridSearchCV(csvm, parameters, cv=5, nested=True)
    searcher.fit(x, y)

    expected_keys = {
        "param_c",
        "param_gamma",
        "params",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    }
    split_keys = {"split%d_test_score" % i for i in range(5)}
    expected_keys.update(split_keys)

    condition = hasattr(searcher, "cv_results_")
    condition = condition and set(searcher.cv_results_.keys()) == expected_keys
    condition = condition and hasattr(searcher, "best_estimator_")
    condition = condition and hasattr(searcher, "best_score_")
    condition = condition and hasattr(searcher, "best_params_")
    condition = condition and hasattr(searcher, "best_index_")
    condition = condition and hasattr(searcher, "scorer_")
    condition = condition and searcher.n_splits_ == 5
    return condition


class GridSearchNestedTest(BaseTimedTestCase):
    def test_gridsearch_csvm_nested(self):
        self.assertTrue(compss_wait_on(test_gridsearch_csvm_nested()))


@task()
def main():
    test = compss_wait_on(test_gridsearch_csvm_nested())
    if test:
        print("Result tests: Passed", flush=True)
    else:
        print("Result tests: Failed", flush=True)


if __name__ == "__main__":
    main()
