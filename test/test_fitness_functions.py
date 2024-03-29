# TODO: Add test cases for the fitness functions
from utils import (
    jaccard_similarity,
    mean_squared_error,
    cosine_similarity,
)

import numpy as np
import pytest


@pytest.mark.parametrize(
    "image1, image2, expected",
    [
        (np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), 1.0),
        (np.array([[255, 255], [255, 255]]), np.array([[255, 255], [255, 255]]), 1.0),
        (np.array([[0, 0], [0, 0]]), np.array([[255, 255], [255, 255]]), 0.0),
        (np.array([[255, 255], [255, 255]]), np.array([[0, 0], [0, 0]]), 0.0),
        (np.array([[0, 0], [0, 0]]), np.array([[0, 0], [255, 255]]), 0.5),
        (np.array([[255, 255], [255, 255]]), np.array([[255, 255], [0, 0]]), 0.5),
        (np.array([[0, 0], [0, 0]]), np.array([[255, 255], [0, 0]]), 0.5),
        (np.array([[255, 255], [0, 0]]), np.array([[0, 0], [255, 255]]), 0.0),
    ],
)
def test_jaccard_similarity(image1, image2, expected):
    assert jaccard_similarity(image1, image2) == pytest.approx(expected)


@pytest.mark.parametrize(
    "image1, image2, expected",
    [
        (np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), 1.0),
        (np.array([[255, 255], [255, 255]]), np.array([[255, 255], [255, 255]]), 1.0),
        (np.array([[0, 0], [0, 0]]), np.array([[255, 255], [255, 255]]), 0.0),
        (np.array([[255, 255], [255, 255]]), np.array([[0, 0], [0, 0]]), 0.0),
        (np.array([[0, 0], [0, 0]]), np.array([[0, 0], [255, 255]]), 0.2928932188),
        (
            np.array([[255, 255], [255, 255]]),
            np.array([[255, 255], [0, 0]]),
            0.2928932188,
        ),
        (
            np.array([[0, 0], [0, 0]]),
            np.array([[255, 255], [0, 0]]),
            0.2928932188,
        ),
        (np.array([[255, 255], [0, 0]]), np.array([[0, 0], [255, 255]]), 0.0),
    ],
)
def test_mean_squared_error(image1, image2, expected):
    assert mean_squared_error(image1, image2) == pytest.approx(expected)


@pytest.mark.parametrize(
    "image1, image2, expected",
    [
        (np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), 1.0),
        (np.array([[255, 255], [255, 255]]), np.array([[255, 255], [255, 255]]), 1.0),
        (np.array([[0, 0], [0, 0]]), np.array([[255, 255], [255, 255]]), 0.0),
        (np.array([[255, 255], [255, 255]]), np.array([[0, 0], [0, 0]]), 0.0),
        (np.array([[0, 0], [0, 0]]), np.array([[0, 0], [255, 255]]), 0.5),
        (np.array([[255, 255], [255, 255]]), np.array([[255, 255], [0, 0]]), 0.5),
        (np.array([[0, 0], [0, 0]]), np.array([[255, 255], [0, 0]]), 0.5),
        (np.array([[255, 255], [0, 0]]), np.array([[0, 0], [255, 255]]), 0.0),
    ],
)
def test_cross_correlation(image1, image2, expected):
    assert cosine_similarity(image1, image2) == pytest.approx(expected)
