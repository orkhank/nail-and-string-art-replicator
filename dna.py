from typing import Callable, Union, Optional

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from fitness_functions import (
    cosine_similarity,
    dice_similarity,
    simple_matching_coefficient,
)


class DNA:
    fitness_function_dict: dict[str, Callable] = {
        "smc": simple_matching_coefficient,
        "dice": dice_similarity,
        "cosine": cosine_similarity,
    }
    fitness_function_long_names: dict[str, str] = {
        "smc": "Simple Matching Coefficient",
        "dice": "Dice Similarity",
        "cosine": "Cosine Similarity",
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
        """
        Initialize the parameters for the DNA class

        Args:
            fitness_function_name (str): Name of the fitness function
            mutation_rate (float): Mutation rate
            points (dict[int, tuple[int, int]]): Points on the image
        """
        cls.set_fitness_function(fitness_function_name)
        cls.set_mutation_rate(mutation_rate)
        cls.set_points(points)

    def __init__(
        self,
        sequence: np.ndarray,
        target_image: np.ndarray,
    ):
        """
        Create a DNA object

        Args:
            sequence (np.ndarray): Sequence of points
            target_image (np.ndarray): Target image
        """
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

    def get_image_with_lines(self) -> np.ndarray:
        """
        Create an image with lines connecting the points in the sequence

        Returns:
            np.ndarray: Image with lines connecting the points
        """
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
        """
        Calculate the fitness of the DNA object

        Returns:
            float: Fitness of the DNA object
        """
        dna_image = self.get_image_with_lines()

        # calculate the fitness
        return self.fitness_func(dna_image, self.target_image)

    def mutate(self, mutation_rate: Optional[float] = None):
        """
        Mutate the DNA sequence

        Args:
            mutation_rate (float, optional): Mutation rate. Defaults to DNA.mutation_rate.
        """
        possible_points = DNA.get_possible_point_names()
        mutation_rate = mutation_rate or DNA.mutation_rate

        assert mutation_rate is not None, "Mutation rate is not set"
        # mutate the sequence
        mutated_sequence = [
            (
                # randomly select a point from the possible points
                np.random.choice(possible_points)
                if np.random.rand() < mutation_rate
                # otherwise keep the point as is
                else point
            )
            for point in self.sequence
        ]

        # update the sequence
        self.sequence = np.array(mutated_sequence)

    @property
    def fitness_func(self) -> Callable:
        assert DNA.fitness_function_name is not None, "Fitness function is not set"
        return DNA.fitness_function_dict[DNA.fitness_function_name]

    @staticmethod
    def crossover(parent1: "DNA", parent2: "DNA") -> tuple["DNA", "DNA"]:
        assert len(parent1) == len(parent2), "Parents should have the same length"

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
