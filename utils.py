import cv2 as cv
import numpy as np
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr,
)


def mean_match(image1, image2) -> float:
    assert image1.shape == image2.shape

    mean_mismatches = np.mean(np.logical_xor(image1, image2))
    return 1 - mean_mismatches  # Normalize to [0, 1]


def mean_squared_error(image1, image2) -> float:
    assert image1.shape == image2.shape

    mse = np.sqrt(np.mean(np.square(image1 - image2)))
    return 1 - mse / 255  # Normalize to [0, 1]


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


def get_points_on_circle(
    center: tuple[int, int], radius: int, num_points: int = 360
) -> dict[int, tuple[int, int]]:
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append((int(x), int(y)))

    named_points = {i + 1: point for i, point in enumerate(points)}
    return named_points


def read_binary_image(image_path: str) -> np.ndarray:
    original_image = cv.imread(image_path)
    if original_image is None:
        print("Image not found")
        exit()

    gray = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    _, binary_image = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    return binary_image
