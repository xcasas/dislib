import numbers

from dislib.data.array import Array
from dislib.data.array import ds_array_from_sendable_parameter
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import INOUT, Depth, Type, COLLECTION_IN
import numpy as np
import sys


def fit(estimator, train_ds, parameters, fit_params):
    if parameters is not None:
        estimator.set_params(**parameters)
    x_train, y_train = train_ds
    estimator.fit(x_train, y_train, **fit_params)
    return estimator


def score_func(estimator, validation_ds, scorer):
    x_test, y_test = validation_ds
    test_scores = _score(estimator, x_test, y_test, scorer)

    return [test_scores]

def fit_and_score_and_validate(estimator, train, validation, scorer, parameters, fit_params):
    ds_x_train, ds_y_train = train
    ds_x_test, ds_y_test = validation

    (x_traing_blocks, x_traing_top_left_shape, x_traing_reg_shape, x_traing_shape) = ds_x_train.make_sendable_parameter()
    (y_traing_blocks, y_traing_top_left_shape, y_traing_reg_shape, y_traing_shape) = ds_y_train.make_sendable_parameter()
    (x_test_blocks, x_test_top_left_shape, x_test_reg_shape, x_test_shape) = ds_x_test.make_sendable_parameter()
    (y_test_blocks, y_test_top_left_shape, y_test_reg_shape, y_test_shape) = ds_y_test.make_sendable_parameter()
    print("Invoking _fit_and_score_and_validate", flush=True)
    return _fit_and_score_and_validate_task(estimator,
                                            x_traing_blocks, x_traing_top_left_shape, x_traing_reg_shape, x_traing_shape,
                                            y_traing_blocks, y_traing_top_left_shape, y_traing_reg_shape, y_traing_shape,
                                            x_test_blocks, x_test_top_left_shape, x_test_reg_shape, x_test_shape,
                                            y_test_blocks, y_test_top_left_shape, y_test_reg_shape, y_test_shape,
                                            scorer, parameters, fit_params)

@task(is_distributed=True, x_traing_blocks={Type: COLLECTION_IN, Depth: 2},
      y_traing_blocks={Type: COLLECTION_IN, Depth: 2},
      x_test_blocks={Type: COLLECTION_IN, Depth: 2},
      y_test_blocks={Type: COLLECTION_IN, Depth: 2},)
def _fit_and_score_and_validate_task(estimator,x_traing_blocks, x_traing_top_left_shape, x_traing_reg_shape, x_traing_shape, y_traing_blocks, y_traing_top_left_shape, y_traing_reg_shape, y_traing_shape, x_test_blocks, x_test_top_left_shape, x_test_reg_shape, x_test_shape, y_test_blocks, y_test_top_left_shape, y_test_reg_shape, y_test_shape, scorer, parameters,
                                     fit_params):


    print("______ empezando fit and score and validate task")
    sys.stdout.flush()
    ds_x_train = ds_array_from_sendable_parameter((x_traing_blocks, x_traing_top_left_shape, x_traing_reg_shape, x_traing_shape))
    ds_y_train = ds_array_from_sendable_parameter((y_traing_blocks, y_traing_top_left_shape, y_traing_reg_shape, y_traing_shape))
    ds_x_test = ds_array_from_sendable_parameter((x_test_blocks, x_test_top_left_shape, x_test_reg_shape, x_test_shape))
    ds_y_test = ds_array_from_sendable_parameter((y_test_blocks, y_test_top_left_shape, y_test_reg_shape, y_test_shape))

    #fit
    if parameters is not None:
        estimator.set_params(**parameters)
    estimator.fit(ds_x_train, ds_y_train, **fit_params)


    #score
    scores = _score(estimator, ds_x_test, ds_y_test, scorer)

    #validate
    # scores = compss_wait_on(scores)
    for scorer_name, score in scores.items():
        score = compss_wait_on(score)
        scores[scorer_name] = validate_score(score, scorer_name)

    print("______ finalizado fit and score and validate task")
    return [scores]

@task(est=INOUT, blocks_x={Type: COLLECTION_IN, Depth: 2},
      blocks_y={Type: COLLECTION_IN, Depth: 2})
def fit_sklearn_estimator(est, blocks_x, blocks_y, **fit_params):
    x = Array._merge_blocks(blocks_x)
    y = Array._merge_blocks(blocks_y)
    return est.fit(x, y, **fit_params)


@task(blocks_x={Type: COLLECTION_IN, Depth: 2},
      blocks_y={Type: COLLECTION_IN, Depth: 2},
      returns=1)
def score_sklearn_estimator(est, scorer,  blocks_x, blocks_y):
    x = Array._merge_blocks(blocks_x)
    y = Array._merge_blocks(blocks_y)
    return _score(est, x, y, scorer)


def execute_simulation(simulation, **parameters):
    return simulation(**parameters)


def simulation_execution(simulation, parameters,
                         simulation_params, number_simulations):
    simulations_result = []
    if parameters is not None:
        for _ in range(number_simulations):
            simulations_result.append(execute_simulation(simulation,
                                                         **parameters,
                                                         **simulation_params))
    return simulations_result


def sklearn_fit(estimator, train_ds,
                parameters, fit_params):
    if parameters is not None:
        estimator.set_params(**parameters)
    x_train, y_train = train_ds

    return fit_sklearn_estimator(estimator, x_train._blocks,
                                 y_train._blocks, **fit_params)


def sklearn_score(estimator, validation_ds, scorer):
    x_test, y_test = validation_ds
    test_scores = score_sklearn_estimator(estimator, scorer,
                                          x_test._blocks, y_test._blocks)

    return [test_scores]


def _score(estimator, x, y, scorers):
    """Return a dict of scores"""
    scores = {}

    for name, scorer in scorers.items():
        try:
            score = scorer(estimator, x, y)
        except TypeError:
            prediction = estimator.predict(x)
            if not isinstance(prediction, Array):
                score = scorer(np.block(prediction), np.block(y))
            else:
                y = y.collect()
                score = scorer(np.block(prediction.collect()),
                               y)
        scores[name] = score
    return scores


def validate_score(score, name):
    if not isinstance(score, numbers.Number) and \
            not (isinstance(score, np.ndarray) and len(score.shape) == 0):
        raise ValueError("scoring must return a number, got %s (%s) "
                         "instead. (scorer=%s)"
                         % (str(score), type(score), name))
    return score


def aggregate_score_dicts(scores):
    """Aggregate the results of each scorer
    Example
    -------
    >>> scores = [{'a': 1, 'b':10}, {'a': 2, 'b':2}, {'a': 3, 'b':3},
    ...           {'a': 10, 'b': 10}]
    >>> aggregate_score_dicts(scores)
    {'a': array([1, 2, 3, 10]),
     'b': array([10, 2, 3, 10])}
    """
    return {key: np.asarray([score[key] for score in scores])
            for key in scores[0]}


def check_scorer(estimator, scorer):
    if scorer is None:
        if hasattr(estimator, 'score'):
            def _passthrough_scorer(estimator, *args, **kwargs):
                """Function that wraps estimator.score"""
                return estimator.score(*args, **kwargs)
            return _passthrough_scorer
        else:
            raise TypeError(
                "If a scorer is None, the estimator passed should have a "
                "'score' method. The estimator %r does not." % estimator)
    elif callable(scorer):
        return scorer
    raise ValueError("Invalid scorer %r" % scorer)
