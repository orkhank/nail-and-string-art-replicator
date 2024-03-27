from utils import mean_match, cross_correlation

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
def test_sum_of_squared_differences_duplicates(image1, image2, expected):
    assert mean_match(image1, image2) == expected
