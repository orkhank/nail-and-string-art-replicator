import cv2 as cv
from matplotlib import pyplot as plt
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


def draw_points_on_image(
    image: np.ndarray, points: dict[int, tuple[int, int]]
) -> np.ndarray:
    image_with_points = image.copy()
    for point in points.values():
        cv.circle(image_with_points, point, 1, (0, 0, 0), 1)
    return image_with_points


def get_points_on_image(
    image: np.ndarray, radius: int, num_points: int = 360
) -> dict[int, tuple[int, int]]:
    # get center of the image
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # get points on the circle
    points = get_points_on_circle(center, radius, num_points)
    return points


def visualize_fitness(
    title: str,
    fitness_over_time: list[float],
    fitness_function_name: str,
):
    plt.figure()
    plt.plot(fitness_over_time)
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel(f"Fitness ({fitness_function_name})")
    plt.grid()
    plt.tight_layout()
    return plt.gcf()


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
