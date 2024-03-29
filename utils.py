import numpy as np
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr,
)


def mean_match(image1, image2) -> float:
    assert image1.shape == image2.shape

    mean_matches = np.mean(np.logical_not(np.logical_xor(image1, image2)))
    return mean_matches


def mean_squared_error(image1, image2) -> float:
    assert image1.shape == image2.shape

    mse = np.sqrt(np.mean(np.square(image1 - image2)))
    return mse / 255  # Normalize to [0, 1]


def cross_correlation(image1, image2) -> float:
    """
    Cross-correlation similarity metric.
    Args:
        image1 (np.ndarray): Binary black and white image.
        image2 (np.ndarray): Binary black and white image.

    Returns:
        float: Cross-correlation similarity metric.
    """

    assert image1.shape == image2.shape
    normalized_image1 = image1 / 255
    normalized_image2 = image2 / 255

    cross_correlation = np.sum(
        (normalized_image1 - np.mean(normalized_image1))
        * (normalized_image2 - np.mean(normalized_image2))
    ) / (
        np.sqrt(
            np.sum((normalized_image1 - np.mean(normalized_image1)) ** 2)
            * np.sum((normalized_image2 - np.mean(normalized_image2)) ** 2)
        )
        + 1e-10
    )
    # Normalize to [0, 1]
    cross_correlation = (cross_correlation + 1) / 2
    return cross_correlation


def structural_similarity(image1, image2) -> float:
    assert image1.shape == image2.shape

    score = ssim(image1, image2, data_range=255)
    # Normalize to [0, 1]
    return (score + 1) / 2


def peak_signal_noise_ratio(image1, image2) -> float:
    assert image1.shape == image2.shape

    return psnr(image1, image2, data_range=255) / 100  # Normalize to [0, 1]
