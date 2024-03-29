import argparse
from pathlib import Path
from typing import Callable, Optional, Union
import cv2 as cv
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
from utils import (
    cross_correlation,
    mean_match,
    mean_squared_error,
    structural_similarity,
    peak_signal_noise_ratio,
    get_points_on_circle,
    read_binary_image,
)


class DNA:
    fitness_function_dict: dict[str, Callable] = {
        "ssd": mean_match,
        "mse": mean_squared_error,
        "crco": cross_correlation,
        "ssim": structural_similarity,
        "psnr": peak_signal_noise_ratio,
    }
    fitness_function_name: Union[str, None] = None
    mutation_rate: Union[float, None] = None
    points: Union[dict[int, tuple[int, int]], None] = None

    @classmethod
    def set_fitness_function(cls, name: str):
        assert name in cls.fitness_function_dict, f"Unknown fitness function: {name}"
        cls.fitness_function_name = name

    @classmethod
    def set_mutation_rate(cls, rate: float):
        cls.mutation_rate = rate

    @classmethod
    def set_points(cls, points: dict[int, tuple[int, int]]):
        cls.points = points

    @classmethod
    def init_params(
        cls,
        fitness_function_name: str,
        mutation_rate: float,
        points: dict[int, tuple[int, int]],
    ):
        cls.set_fitness_function(fitness_function_name)
        cls.set_mutation_rate(mutation_rate)
        cls.set_points(points)

    def __init__(
        self,
        sequence: np.ndarray,
        target_image: np.ndarray,
    ):
        assert self.fitness_func is not None, "Fitness function is not set"
        assert isinstance(sequence, np.ndarray), "Sequence should be a numpy array"
        assert isinstance(
            target_image, np.ndarray
        ), "Target image should be a numpy array"
        self.sequence = sequence
        self.target_image = target_image

    def __str__(self):
        return str(self.sequence)

    def __repr__(self):
        return str(self.sequence)

    def __len__(self):
        return len(self.sequence)

    def get_image_with_lines(self):
        # create white background image
        height, width = self.target_image.shape[:2]
        image_with_lines = np.ones((height, width)) * 255

        # draw lines on the white background
        for i in range(len(self.sequence) - 1):
            start_point = DNA.get_point(self.sequence[i])
            end_point = DNA.get_point(self.sequence[i + 1])
            cv.line(
                image_with_lines,
                start_point,
                end_point,
                (0, 0, 0),
                1,
            )

        return image_with_lines

    def fitness(self) -> float:
        dna_image = self.get_image_with_lines()

        # calculate the fitness
        return self.fitness_func(dna_image, self.target_image)

    def mutate(self, mutation_rate: Optional[float] = None):
        possible_points = DNA.get_possible_point_names()
        mutation_rate = mutation_rate or DNA.mutation_rate

        assert mutation_rate is not None, "Mutation rate is not set"
        # mutate the sequence
        mutated_sequence = [
            (
                np.random.choice(possible_points)
                if np.random.rand() < mutation_rate
                else point
            )
            for point in self.sequence
        ]

        # update the sequence
        self.sequence = np.array(mutated_sequence)

    @property
    def fitness_func(self):
        assert DNA.fitness_function_name is not None, "Fitness function is not set"
        return DNA.fitness_function_dict[DNA.fitness_function_name]

    @staticmethod
    def crossover(parent1, parent2):
        # select a random point to split the sequence
        split_point = np.random.randint(0, len(parent1))

        # create the child sequences
        child1 = DNA(
            np.append(parent1.sequence[:split_point], parent2.sequence[split_point:]),
            parent1.target_image,
        )
        child2 = DNA(
            np.append(parent2.sequence[:split_point], parent1.sequence[split_point:]),
            parent2.target_image,
        )

        return child1, child2

    @classmethod
    def get_possible_point_names(cls) -> list[int]:
        assert cls.points is not None, "Points are not set"
        return list(cls.points.keys())

    @classmethod
    def get_point(cls, name: int) -> tuple[int, int]:
        assert cls.points is not None, "Points are not set"
        return cls.points[name]

    def visualize(self, title: str = "DNA", wait: int = 0):
        # show the image
        plt.imshow(self.get_image_with_lines(), cmap="gray")
        plt.title(title)
        plt.show(block=False)
        plt.pause(wait / 1000)
        plt.close()


def get_args() -> argparse.Namespace:
    default_sequence_length = 50
    default_population_size = 100
    default_mutation_rate = 0.01
    default_number_of_generations = 1000
    default_loss_function = "crco"
    default_keep_percentile = 50
    parser = argparse.ArgumentParser(
        prog="Nail and String Art",
        description="Create nail and string art from an image",
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Path to the output image file"
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=int,
        required=True,
        help="Radius of the circle, located at the center of the image, in pixels. "
        "The circle will be divided into 360 points and nails will be placed at these points. "
        "The radius should be less than the minimum dimension of the image. ",
    )
    parser.add_argument(
        "-s",
        "--sequence_length",
        type=int,
        default=default_sequence_length,
        help=f"The length of the DNA sequence. Default is {default_sequence_length} points.",
    )
    parser.add_argument(
        "-p",
        "--population_size",
        type=int,
        default=default_population_size,
        help=f"The size of the population. Default is {default_population_size}.",
    )
    parser.add_argument(
        "-m",
        "--mutation_rate",
        type=float,
        default=default_mutation_rate,
        help=f"The mutation rate. Default is {default_mutation_rate}.",
    )
    parser.add_argument(
        "-g",
        "--generations",
        type=int,
        default=default_number_of_generations,
        help=f"The number of generations. Default is {default_number_of_generations}.",
    )
    parser.add_argument(
        "-k",
        "--keep_percentile",
        type=float,
        default=default_keep_percentile,
        help=f"Which percentile of the population to keep for the next generation. Default is {default_keep_percentile}.",
    )
    parser.add_argument(
        "-l",
        "--loss_function",
        type=str,
        choices=DNA.fitness_function_dict.keys(),
        default=default_loss_function,
        help="The loss function to use. "
        # TODO: automatically get the list of available loss functions and display them here
        "Default is ssd.",
    )

    return parser.parse_args()


def get_initial_population(points, population_size, sequence_length, target_image):
    initial_population = []
    for _ in range(population_size):
        random_sequence = np.random.choice(list(points.keys()), sequence_length)
        initial_population.append(DNA(random_sequence, target_image))
    return initial_population


def train(
    points: dict[int, tuple[int, int]],
    population_size: int,
    generations: int,
    keep_percentile: float,
    sequence_length: int,
    target_image: np.ndarray,
):
    assert (
        len(points) >= sequence_length
    ), "Number of points should be greater than or equal to the sequence length"

    # create the initial population
    population = get_initial_population(
        points, population_size, sequence_length, target_image
    )

    # create the probabilities for selection
    probabilities = np.linspace(1, 0, population_size) / np.sum(
        np.linspace(1, 0, population_size)
    )
    keep_index = int(population_size * keep_percentile / 100)
    dropout_index = population_size - keep_index
    crossover_count = dropout_index // 2 + 1

    best_dnas = []
    fitness_over_time = []
    for generation in tqdm(range(generations), desc=f"Training", unit="generation"):
        # sort the population based on fitness
        population.sort(key=lambda dna: dna.fitness(), reverse=True)

        # create the next generation
        next_generation = []
        # create the next generation
        for _ in range(crossover_count):
            # select two parents
            parent1 = np.random.choice(population, p=probabilities)
            parent2 = np.random.choice(population, p=probabilities)

            # crossover
            child1, child2 = DNA.crossover(parent1, parent2)

            # mutate
            child1.mutate()
            child2.mutate()

            next_generation.extend([child1, child2])

        # keep the top keep_percentile % of the population
        population = population[:keep_index]
        population.extend(next_generation[:dropout_index])

        # sanity check
        assert (
            len(population) == population_size
        ), f"Population size is not correct: {len(population)} != {population_size}"

        # store the fitness of the best DNA object
        best_dna: DNA = max(population, key=lambda dna: dna.fitness())
        best_dnas.append(best_dna)
        fitness_over_time.append(best_dna.fitness())

        # print the fitness of the best DNA object
        if generation % 100 == 0:
            print(f"\nGeneration: {generation}, Fitness: {best_dna.fitness()}")
            best_dna.visualize(
                f"Generation {generation}, Fitness: {best_dna.fitness()}", wait=500
            )

    return best_dnas, fitness_over_time


def visualize_fitness(
    title: str,
    fitness_over_time: list[float],
    loss_function: str,
):
    plt.plot(fitness_over_time)
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel(f"Fitness ({loss_function})")
    plt.grid()
    plt.tight_layout()
    plt.show()
    return plt


def main():
    np.random.seed(42)
    args = get_args()

    print(args)
    print("Reading the image...")
    # Read the image as binary black and white
    image = read_binary_image(args.image_path)

    # get center of the image
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # get points on the circle
    points = get_points_on_circle(center, args.radius)

    # draw the points on the image
    image_with_points = image.copy()
    for point in points.values():
        cv.circle(image_with_points, point, 1, (0, 0, 0), 1)

    # show the image on a plot
    plt.imshow(image_with_points, cmap="gray")
    plt.title("Image with Points")
    plt.show()

    print("Training the model...")
    # set the fitness function and mutation rate
    DNA.init_params(args.loss_function, args.mutation_rate, points)

    # train the model
    start_time = time.time()
    best_dnas, fitness_over_time = train(
        points,
        args.population_size,
        args.generations,
        args.keep_percentile,
        args.sequence_length,
        image,
    )
    end_time = time.time()
    train_time = end_time - start_time
    print(f"Training time: {train_time:.2f} seconds")

    # show the best DNA
    best_dna = best_dnas[-1]
    best_dna.visualize(f"Best Sequence (Fitness: {best_dna.fitness()})", wait=0)

    # print details of the best DNA
    print(f"Best sequence found (fitness: {best_dna.fitness()}): {best_dna.sequence}")

    # visualize the fitness over generations
    fitness_plot = visualize_fitness(
        f"Fitness over {args.generations} generations.",
        fitness_over_time,
        args.loss_function,
    )
    fitness_plot_path = Path("outputs") / (
        Path(args.image_path).stem
        + f"_fitness_r{args.radius}_s{args.sequence_length}_p{args.population_size}_g{args.generations}_k{args.keep_percentile}_m{args.mutation_rate}_l{args.loss_function}.png"
    )
    fitness_plot.savefig(fitness_plot_path)
    print(f"Fitness plot saved at {fitness_plot_path}")

    # save the binary image
    output_path = args.output or Path("outputs") / (
        Path(args.image_path).stem
        + f"_r{args.radius}_s{args.sequence_length}_p{args.population_size}_g{args.generations}_k{args.keep_percentile}_m{args.mutation_rate}_l{args.loss_function}.png"
    )

    cv.imwrite(str(output_path), best_dna.get_image_with_lines())
    print(f"Output image saved at {output_path}")


if __name__ == "__main__":
    main()
