from typing import Optional
import numpy as np
from tqdm import tqdm
from dna import DNA


def get_initial_population(
    points, population_size, sequence_length, target_image
) -> list[DNA]:
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
    verbose: bool = True,
    initial_population: Optional[list[DNA]] = None,
):
    assert (
        len(points) >= sequence_length
    ), "Number of points should be greater than or equal to the sequence length"

    if initial_population is None:
        # create the initial population
        population = get_initial_population(
            points, population_size, sequence_length, target_image
        )
    else:
        # use the provided initial population
        population = initial_population
        assert (
            len(population) == population_size
        ), f"The initial population size is not correct: {len(population)} != {population_size}"

    # create the static probabilities for selection
    selection_probabilities = np.linspace(1, 0, population_size) / np.sum(
        np.linspace(1, 0, population_size)
    )

    # calculate the number of DNAs to keep and dropout
    keep_index = int(population_size * keep_percentile / 100)
    dropout_index = population_size - keep_index
    crossover_count = dropout_index // 2 + 1

    # store the best DNA objects and fitness over time
    best_dnas = []
    fitness_over_time = []

    # train the model
    for generation in tqdm(range(generations), desc=f"Training", unit="generation"):
        # sort the population based on fitness
        population.sort(key=lambda dna: dna.fitness(), reverse=True)

        # create the next generation
        next_generation = []
        # create the next generation
        for _ in range(crossover_count):
            # select two parents
            parent1: DNA = np.random.choice(population, p=selection_probabilities)  # type: ignore
            parent2: DNA = np.random.choice(population, p=selection_probabilities)  # type: ignore

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
        if verbose and generation % 100 == 0:
            print(f"\nGeneration: {generation}, Fitness: {best_dna.fitness()}")
            best_dna.visualize(
                f"Generation {generation}, Fitness: {best_dna.fitness()}", wait=0.5
            )

    return best_dnas, fitness_over_time
