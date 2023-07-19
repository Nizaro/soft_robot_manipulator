"""Random sample consensus (RANSAC) module.

This module contains the core code for the RANSAC algorithm.
"""

# Standard library imports
from dataclasses import dataclass
from math import log
import random
from typing import List
import copy

# Local application imports
from pyransac.base import Model


@dataclass
class RansacParams:
    """Random sample consensus (RANSAC) function parameters.

    This class contains the parameters for the RANSAC algorithm.
    """
    samples: int
    """The number of random samples to take per iteration."""

    iterations: int
    """Maximum number iterations to complete."""

    confidence: float
    """The RANSAC confidence value (0 <= confidence <= 1)."""

    threshold: float
    """The error threshold to consider a point an inlier"""


def find_inliers(points: List, model: Model, params: RansacParams):
    """Find the inliers from a data set.

    Finds the inliers from a given data set given a model and
    an error function.

    :param points: data points to evaluate
    :param model: type of model to which the data should adhere
    :param params: parameters for the RANSAC algorithm
    :return: inliers
    """
    inliers = []
    inliers_ind = []
    max_support = 0
    iterations = params.iterations
    i = 0
    ratio=0
    best_model=copy.deepcopy(model)
    while i < iterations:
        npoints= list(range(len(points)))
        sample_points = random.sample(npoints, k=params.samples)
        lpoints= [points[i] for i in sample_points]
        Valid=model.make_model(lpoints)
        supporters_ind = _find_supporters(points, model, params.threshold)
        supporters=[points[i] for i in supporters_ind ]
        

        if len(supporters) > max_support and Valid==True:
            max_support = len(supporters)
            inliers = supporters
            inliers_ind=supporters_ind
            best_model=copy.deepcopy(model)

            confidence = 1 - params.confidence
            ratio = len(supporters) / len(points)
            

            # We cannot get more support than all data points
            if ratio == 1:
                break

            iterations = log(confidence) / log(1 - ratio ** params.samples)
            

        #print('iteration:',i,'/',iterations,'Ratio:',ratio,' Validity:',Valid)
        i += 1
    outliers_ind=[i for i in range(len(points)) if inliers_ind.count(i)==0]
    outliers=[points[i] for i in outliers_ind ]
    return inliers,outliers, best_model,ratio


def _find_supporters(points: List, model: Model, threshold: float) -> List:
    """Find data points (supporters) that support the given hypothesis.

    :param points: data points to test against the hypothesis
    :param model: type of model to which the data should adhere
    :param threshold: error threshold to consider data point an inlier
    :return: data points that support the hypothesis
    """
    #return [point for point in points if model.calc_error(point) <= threshold]
    return [i for i in range(len(points)) if model.calc_error(points[i]) <= threshold]
