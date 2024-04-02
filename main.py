import argparse
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import time
from dna import DNA
from train import train
from utils import (
    draw_points_on_image,
    get_points_on_image,
    read_binary_image,
    save_train_results,
    visualize_fitness,
)

# Default values for the arguments
default_sequence_length = 50
default_population_size = 100
default_mutation_rate = 0.01
default_number_of_generations = 1000
default_fitness_function = "cosine"
default_keep_percentile = 50


def get_args() -> argparse.Namespace:
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
        "-o",
        "--output",
        type=str,
        default="outputs",
        help="Path to save the output files. Default is 'outputs' folder.",
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
        "-f",
        "--fitness_function",
        type=str,
        default=default_fitness_function,
        help="The fitness function to use. "
        f"Options are: {', '.join([f'{k} ({v})' for k, v in DNA.fitness_function_long_names.items()])}. "
        f"Default is {default_fitness_function}.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    print(args)
    print("Reading the image...")
    # Read the image as binary black and white
    image = read_binary_image(args.image_path)

    # get the points on the image
    points = get_points_on_image(image, args.radius)

    if args.verbose:
        # draw the points on the image
        image_with_points = draw_points_on_image(image, points)

        # show the image with points
        plt.imshow(image_with_points, cmap="gray")
        plt.title("Image with Points")
        plt.show()
        plt.close()

    print("Training the model...")
    # set the fitness function and mutation rate
    DNA.init_params(args.fitness_function, args.mutation_rate, points)

    # train the model
    start_time = time.time()
    best_dnas, fitness_over_time = train(
        points,
        args.population_size,
        args.generations,
        args.keep_percentile,
        args.sequence_length,
        image,
        args.verbose,
    )
    end_time = time.time()
    train_time = end_time - start_time
    print(f"Training time: {train_time:.2f} seconds")

    # get the best DNA
    best_dna: DNA = best_dnas[-1]

    # print details of the best DNA
    print(f"Best sequence found (fitness: {best_dna.fitness()}): {best_dna.sequence}")
    if args.verbose:
        # show the best DNA
        best_dna.visualize(f"Best Sequence (Fitness: {best_dna.fitness()})", wait=0)

    # visualize the fitness over generations
    fitness_plot = visualize_fitness(
        f"Fitness over {args.generations} generations.",
        fitness_over_time,
        args.fitness_function,
    )

    if args.verbose:
        plt.show()
        plt.close()

    # save the output image
    if args.output:
        save_train_results(
            fitness_plot,
            best_dna,
            train_time,
            base_output_path=Path(args.output),
            image_name=Path(args.image_path).stem,
            r=args.radius,
            f=args.fitness_function,
            p=args.population_size,
            m=args.mutation_rate,
            k=args.keep_percentile,
            s=args.sequence_length,
            g=args.generations,
        )


if __name__ == "__main__":
    np.random.seed(42)
    args = get_args()

    main(args)
