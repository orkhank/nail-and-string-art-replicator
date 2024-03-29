import cv2 as cv
import numpy as np


def simple_matching_coefficient(image1, image2) -> float:
    """
    Simple matching coefficient metric.

    Args:
        image1 (np.ndarray): Binary black and white image.
        image2 (np.ndarray): Binary black and white image.

    Returns:
        float: The amount of pixels that are the same in both images divided by the total amount of pixels.
    """
    assert image1.shape == image2.shape
    return np.count_nonzero(image1 == image2) / image1.size


def dice_similarity(image1, image2) -> float:
    """
    Dice similarity metric.

    Args:
        image1 (np.ndarray): Binary black and white image.
        image2 (np.ndarray): Binary black and white image.

    Returns:
        float: Dice similarity metric.
    """

    assert image1.shape == image2.shape

    a, b, c = get_a_b_c(image1, image2)
    return 2 * c / (a + b)


def cosine_similarity(image1, image2) -> float:
    """
    Cross-correlation similarity metric.
    Args:
        image1 (np.ndarray): Binary black and white image.
        image2 (np.ndarray): Binary black and white image.

    Returns:
        float: Cross-correlation similarity metric.
    """

    assert image1.shape == image2.shape
    # normalized_image1 = image1 / 255
    # normalized_image2 = image2 / 255

    # cross_correlation = np.sum(
    #     (normalized_image1 - np.mean(normalized_image1))
    #     * (normalized_image2 - np.mean(normalized_image2))
    # ) / (
    #     np.sqrt(
    #         np.sum((normalized_image1 - np.mean(normalized_image1)) ** 2)
    #         * np.sum((normalized_image2 - np.mean(normalized_image2)) ** 2)
    #     )
    #     + 1e-10
    # )
    # # Normalize to [0, 1]
    # cross_correlation = (cross_correlation + 1) / 2
    # return cross_correlation

    a, b, c = get_a_b_c(image1, image2)
    return c / np.sqrt(a * b)


# same as dice_similarity
def tanimoto_similarity(image1, image2) -> float:
    """
    Tanimoto similarity metric.

    Args:
        image1 (np.ndarray): Binary black and white image.
        image2 (np.ndarray): Binary black and white image.

    Returns:
        float: Tanimoto similarity metric.
    """
    assert image1.shape == image2.shape

    a, b, c = get_a_b_c(image1, image2)
    return c / (a + b - c)


# this metric is bad
def manhattan_similarity(image, target_image) -> float:
    """
    Manhattan similarity metric.

    Args:
        image (np.ndarray): Binary black and white image.
        target_image (np.ndarray): Binary black and white image.

    Returns:
        float: Manhattan similarity metric.
    """
    assert image.shape == target_image.shape

    a, b, c = get_a_b_c(image, target_image)

    # shittt
    return a + b - 2 * c


# this metric is as bad as manhattan_similarity. (literally. it is the same thing)
def euclidean_similarity(image, target_image) -> float:
    """
    Euclidean similarity metric.

    Args:
        image (np.ndarray): Binary black and white image.
        target_image (np.ndarray): Binary black and white image.

    Returns:
        float: Euclidean similarity metric.
    """
    assert image.shape == target_image.shape

    return np.sqrt(manhattan_similarity(image, target_image))


def mean_squared_error(image1, image2) -> float:
    """
    Mean squared error similarity metric.

    Args:
        image1 (np.ndarray): Binary black and white image.
        image2 (np.ndarray): Binary black and white image.

    Returns:
        float: Mean squared error similarity metric.
    """
    assert image1.shape == image2.shape

    mse = np.sqrt(np.mean(np.square(image1 - image2)))
    return 1 - mse / 255  # Normalize to [0, 1]


# This is the same things as dice_similarity
def jaccard_similarity(image1, image2) -> float:
    """
    Jaccard similarity metric.

    Args:
        image1 (np.ndarray): Binary black and white image.
        image2 (np.ndarray): Binary black and white image.

    Returns:
        float: Jaccard similarity metric.
    """

    assert image1.shape == image2.shape

    a, b, c = get_a_b_c(image1, image2)
    return c / (a + b - c)


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


def get_important_color(image) -> tuple[int, int]:
    """
    Get the majority color of a binary black and white image.

    Args:
        image (np.ndarray): Binary black and white image.

    Returns:
        tuple[int, int]: Minority color and its count.

    """

    white = 255
    black = 0
    white_count = np.sum(image == white)
    black_count = np.sum(image == black)
    if white_count > black_count:
        return black, black_count
    return white, white_count


def get_a_b_c(image, target_image) -> tuple[int, int, int]:
    """
    Get a, b, c values for the Sorensen-Dice similarity metric.

    Args:
        image (np.ndarray): Binary black and white image.
        target_image (np.ndarray): Binary black and white image.

    Returns:
        tuple[int, int, int]: a, b, c values.
    """
    assert image.shape == target_image.shape

    important_pixel = get_important_color(target_image)[0]

    a = np.count_nonzero(image == important_pixel)
    b = np.count_nonzero(target_image == important_pixel)
    c = np.count_nonzero(
        np.logical_and(image == important_pixel, target_image == important_pixel)
    )

    return a, b, c
