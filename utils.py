import numpy as np
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr,
)
import cv2 as cv


def mean_match(image1, image2) -> float:
    assert image1.shape == image2.shape

    mean_matches = np.mean(np.logical_not(np.logical_xor(image1, image2)))
    return mean_matches


def mean_squared_error(image1, image2) -> float:
    assert image1.shape == image2.shape

    mse = np.sqrt(np.mean(np.square(image1 - image2)))
    return mse / 255  # Normalize to [0, 1]


def cross_correlation(image1, image2) -> float:
    assert image1.shape == image2.shape
    normalized_image1 = image1 / 255
    normalized_image2 = image2 / 255

    Crr = np.sum(np.multiply(normalized_image1, normalized_image2))
    norm = np.sqrt(
        np.sum(np.square(normalized_image1)) * np.sum(np.square(normalized_image2))
    )
    return Crr / norm


def structural_similarity(image1, image2) -> float:
    assert image1.shape == image2.shape

    return ssim(image1, image2, data_range=255)


def peak_signal_noise_ratio(image1, image2) -> float:
    assert image1.shape == image2.shape

    return psnr(image1, image2, data_range=255) / 100  # Normalize to [0, 1]
