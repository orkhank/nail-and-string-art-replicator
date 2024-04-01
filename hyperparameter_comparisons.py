from itertools import product
import json
import cv2 as cv
import time
import numpy as np
from main import (
    train,
    default_fitness_function,
    default_population_size,
    default_keep_percentile,
    default_mutation_rate,
    default_number_of_generations,
    default_sequence_length,
    DNA,
)
from pathlib import Path

from utils import get_points_on_image, read_binary_image, visualize_fitness


def get_results(
    image_path_str: str,
    radius: int,
    fitness_function: str,
    population_size: int,
    mutation_rate: float,
    keep_percentile: int,
    sequence_length: int,
    generations: int,
    verbose: bool = False,
):
    assert (
        fitness_function in DNA.fitness_function_dict.keys()
    ), f"Invalid fitness function: {fitness_function}"
    assert 0 < population_size, f"Invalid population size: {population_size}"
    assert 0 < mutation_rate < 1, f"Invalid mutation rate: {mutation_rate}"
    assert 0 < keep_percentile < 100, f"Invalid keep percentile: {keep_percentile}"
    assert 0 < generations, f"Invalid number of generations: {generations}"
    assert 0 < sequence_length, f"Invalid sequence length: {sequence_length}"
    assert 0 < generations, f"Invalid generations: {generations}"

    image_path = Path(image_path_str)
    assert image_path.exists(), f"Image path does not exist: {image_path}"
    assert image_path.is_file(), f"Image path is not a file: {image_path}"

    # Read the image as binary black and white
    image = read_binary_image(str(image_path))

    # get the points on the image
    points = get_points_on_image(image, radius)

    print("Training the model...")
    # set the fitness function and mutation rate
    DNA.init_params(fitness_function, mutation_rate, points)

    # train the model
    start_time = time.time()
    best_dnas, fitness_over_time = train(
        points,
        population_size,
        generations,
        keep_percentile,
        sequence_length,
        image,
        verbose=verbose,
    )
    end_time = time.time()
    train_time = end_time - start_time
    print(f"Training time: {train_time:.2f} seconds")

    # show the best DNA
    best_dna: DNA = best_dnas[-1]

    # print details of the best DNA
    print(f"Best sequence found (fitness: {best_dna.fitness()}): {best_dna.sequence}")

    # get the fitness plot
    fitness_plot = visualize_fitness(
        f"Fitness over {generations} generations.",
        fitness_over_time,
        fitness_function,
    )

    base_output_path = Path("outputs")
    # save the fitness plot
    image_output_name = Path(
        Path(image_path).stem
        + f"_r{radius}_s{sequence_length}_p{population_size}_g{generations}_k{keep_percentile}_m{mutation_rate}_f{fitness_function}.png"
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

    return best_dna, fitness_over_time, train_time


if __name__ == "__main__":
    np.random.seed(42)
    images = [
        {"image_path": "images\\apple.png", "radius": 40},
        {"image_path": "images\\dog.png", "radius": 50},
        {"image_path": "images\\duck.png", "radius": 33},
        {"image_path": "images\\flower.png", "radius": 33},
        {"image_path": "images\\house.png", "radius": 37},
        {"image_path": "images\\ice_cream.png", "radius": 45},
        {"image_path": "images\\mail.png", "radius": 35},
        {"image_path": "images\\mario_mushroom.png", "radius": 36},
        {"image_path": "images\\mountains.png", "radius": 40},
        {"image_path": "images\\panda.png", "radius": 47},
    ]
    parameters = {
        # "fitness_function": ["smc", "dice", "cosine"],
        # "population_size": [10, 50, 100, 200, 500],
        "mutation_rate": [0.001, 0.01, 0.05, 0.10, 0.50]
        # "keep_percentile": [1, 10, 20, 50, 99],
    }
    fixed_parameters = {
        "population_size": default_population_size,
        "fitness_function": default_fitness_function,
        # "mutation_rate": default_mutation_rate,
        "keep_percentile": default_keep_percentile,
        "sequence_length": default_sequence_length,
        "generations": default_number_of_generations,
    }

    assert len(images) == 10
    assert (
        set(parameters.keys()).intersection(fixed_parameters.keys()) == set()
    ), "Parameters overlap"

    results = {}
    for image in images:
        image_path = image["image_path"]
        radius = image["radius"]
        results[image_path] = {}
        # generate results for each image with permutations of the parameters
        product_values = product(
            *[v if isinstance(v, (list, tuple)) else [v] for v in parameters.values()]
        )
        changing_parameters = [
            dict(zip(parameters.keys(), values)) for values in product_values
        ]
        all_parameters = [{**p, **fixed_parameters} for p in changing_parameters]

        for i, params in enumerate(all_parameters):
            print(
                f"Processing image {image_path} ({radius=}) with parameters {params} ({i+1}/{len(all_parameters)})"
            )
            best_dna, fitness_over_time, train_time = get_results(
                image_path,
                radius,
                **params,
            )
            results[image_path][i] = {
                "radius": radius,
                "parameters": params,
                "best_dna": best_dna.sequence.tolist(),
                "fitness": best_dna.fitness(),
                "train_time": train_time,
            }

    # save the results
    output_path = Path("outputs") / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved at {output_path}")
