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

    a, b, c = get_a_b_c(image1, image2)
    return c / np.sqrt(a * b)


def get_a_b_c(image, target_image) -> tuple[int, int, int]:
    """
    Get the a, b, c values for the matching coefficient metrics based on the important color.

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
