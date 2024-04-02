import json
from pathlib import Path
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from dna import DNA


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


def save_train_results(
    fitness_plot: Figure,
    best_dna: DNA,
    train_time: float,
    *,
    base_output_path: Path = Path("outputs"),
    image_name: str,
    **kwargs,
):
    """
    Save the results of the training.

    Args:
        fitness_plot (Figure): Fitness plot.
        best_dna (DNA): Best DNA object.
        train_time (float): Training time in seconds.
        base_output_path (Path): Base output path.
        image_name (str): Name of the reference image.
        **kwargs: Hyperparameters used for training.
    """

    # save the fitness plot
    image_output_name = Path(
        image_name
        + f"_{'_'.join([f'{key}{value}' for key, value in sorted(list(kwargs.items()), key=lambda x: x[0])])}.png"
    )
    fitness_plot_path = base_output_path / "fitness_plots" / image_output_name
    # make the directory if it does not exist
    fitness_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fitness_plot.savefig(fitness_plot_path)
    print(f"Fitness plot saved at {fitness_plot_path}")

    # save the binary image
    output_path = base_output_path / "output_images" / image_output_name
    # make the directory if it does not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv.imwrite(str(output_path), best_dna.get_image_with_lines())
    print(f"Output image saved at {output_path}")

    # save the best DNA, its fitness and the training time
    other_output_path = (
        base_output_path / "other_outputs" / image_output_name.with_suffix(".json")
    )

    # make the directory if it does not exist
    other_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(other_output_path, "w") as f:
        json.dump(
            {
                "fitness": best_dna.fitness(),
                "sequence": best_dna.sequence.tolist(),
                "training_time": train_time,
            },
            f,
            indent=4,
        )

    print(f"Other output saved at {other_output_path}")
