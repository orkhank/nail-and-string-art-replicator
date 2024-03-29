# TODO: Add test cases for the fitness functions
from utils import (
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
def test_cross_correlation(image1, image2, expected):
    assert cosine_similarity(image1, image2) == pytest.approx(expected)
