import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import mathgenerator


class MathCurriculum:
    """
    A curriculum manager for the mathgenerator library.

    This class organizes math problems by difficulty and provides methods
    to generate problems of appropriate difficulty based on the learner's
    performance.
    """

    # Define difficulty levels and map generator IDs to each level
    DIFFICULTY_LEVELS = {
        # Level 1: Basic arithmetic operations
        1: [
            0,
            1,
            2,
            3,
            8,
            31,
            71,
            80,
            90,
        ],  # Addition, Subtraction, Multiplication, Division, Square, Factorial, Absolute difference, Percentage, IsPrime
        # Level 2: Basic operations with fractions and pre-algebra
        2: [
            6,
            11,
            13,
            16,
            28,
            44,
            47,
            53,
            97,
            118,
            119,
            124,
        ],  # Square Root, Basic Algebra, Fraction to Decimal, Fraction Division, Fraction Multiplication, Compare Fractions, Cube Root, Exponentiation, Power of Powers, Percentage difference/error, Is Composite
        # Level 3: Basic geometry and more algebra
        3: [
            18,
            19,
            22,
            24,
            25,
            49,
            58,
            75,
            96,
            104,
            108,
            112,
            115,
        ],  # Area of Triangle, Triangle exists check, Third Angle of Triangle, Distance between 2 points, Pythagorean Theorem, Fourth Angle of Quadrilateral, Sum of Angles of Polygon, Area of a Sector, Perimeter of Polygons, Circumference, Arc length, Area of Circle
        # Level 4: More advanced algebra and basic statistics
        4: [
            9,
            10,
            20,
            21,
            23,
            26,
            40,
            41,
            45,
            50,
            76,
            78,
            105,
        ],  # LCM, GCD, Midpoint, Factoring Quadratic, System of Equations, Linear Equations, Common Factors, Intersection of Two Lines, Simple Interest, Quadratic Equation, Mean and Median, Compound Interest, Combine Like terms
        # Level 5: Vectors, matrices, and solid geometry
        5: [
            17,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            43,
            46,
            60,
            61,
            70,
            72,
            77,
            95,
            113,
            117,
            122,
            123,
        ],  # Matrix Multiplication, Surface Areas, Volumes, Vector operations, etc.
        # Level 6: Advanced topics (calculus, statistics, computer science)
        6: [
            4,
            5,
            7,
            12,
            14,
            15,
            27,
            30,
            42,
            48,
            52,
            54,
            55,
            56,
            59,
            62,
            64,
            73,
            79,
            84,
            88,
            89,
            91,
            103,
            107,
            110,
        ],  # Binary operations, Calculus, Combinatorics, Probability, etc.
        # Level 7: Most complex topics
        7: [
            65,
            66,
            67,
            68,
            69,
            74,
            85,
            92,
            93,
            94,
            98,
            99,
            100,
            101,
            106,
            109,
            111,
            121,
        ],  # Complex numbers, Advanced operations, etc.
    }

    def __init__(
        self,
        starting_level: int = 1,
        progress_threshold: float = 0.8,
        min_evaluations: int = 5,
    ):
        """
        Initialize the curriculum manager.

        Args:
            starting_level: The difficulty level to start with (default: 1)
            progress_threshold: The success rate required to advance to the next level (default: 0.8)
            min_evaluations: Minimum number of evaluations needed before considering level advancement (default: 5)
        """
        self.current_level = starting_level
        self.progress_threshold = progress_threshold
        self.min_evaluations = min_evaluations

        # Performance tracking
        self.performance_history = {
            level: [] for level in self.DIFFICULTY_LEVELS.keys()
        }

        # Ensure starting level is valid
        if starting_level not in self.DIFFICULTY_LEVELS:
            raise ValueError(
                f"Invalid starting level: {starting_level}. Available levels: {list(self.DIFFICULTY_LEVELS.keys())}"
            )

    def get_problem(self) -> Tuple[str, str, int]:
        """
        Generate a math problem at the current difficulty level.

        Returns:
            Tuple containing (problem_text, solution_text, generator_id)
        """
        # Get the available generator IDs for the current level
        available_generators = self.DIFFICULTY_LEVELS[self.current_level]

        # Try generators until one works
        max_attempts = 5  # Limit the number of attempts to avoid infinite loops
        attempts = 0

        while attempts < max_attempts:
            # Get a random generator ID from the current level
            generator_id = random.choice(available_generators)

            try:
                # Generate the problem
                problem, solution = mathgenerator.genById(generator_id)
                return problem, solution, generator_id
            except Exception as e:
                # Log the error and try another generator
                print(f"Error with generator {generator_id}: {str(e)}")
                attempts += 1

                # Remove the problematic generator from the available list for this session
                if generator_id in available_generators:
                    available_generators.remove(generator_id)

                # If we've exhausted all generators in this level, move to an adjacent level
                if not available_generators:
                    fallback_level = max(
                        1, min(7, self.current_level + random.choice([-1, 1]))
                    )
                    available_generators = self.DIFFICULTY_LEVELS[fallback_level].copy()

        # If all attempts fail, return a simple addition problem as fallback
        return "What is $2 + 2$?", "4", 0

    def record_performance(self, generator_id: int, is_correct: bool) -> None:
        """
        Record the performance on a specific problem.

        Args:
            generator_id: The ID of the generator used
            is_correct: Whether the answer was correct
        """
        # Find which level this generator belongs to
        level = None
        for lvl, generator_ids in self.DIFFICULTY_LEVELS.items():
            if generator_id in generator_ids:
                level = lvl
                break

        if level is not None:
            # Add the result to the performance history
            self.performance_history[level].append(is_correct)

    def get_success_rate(self, level: int) -> Optional[float]:
        """
        Calculate the success rate for a specific level.

        Args:
            level: The difficulty level

        Returns:
            Success rate as a float between 0 and 1, or None if not enough data
        """
        history = self.performance_history[level]

        if len(history) < self.min_evaluations:
            return None

        # Calculate success rate from recent evaluations
        recent_history = history[-self.min_evaluations :]
        return sum(recent_history) / len(recent_history)

    def should_advance(self) -> bool:
        """
        Determine if the learner should advance to the next level.

        Returns:
            Boolean indicating whether to advance
        """
        success_rate = self.get_success_rate(self.current_level)

        # If not enough data or below threshold, don't advance
        if success_rate is None or success_rate < self.progress_threshold:
            return False

        # Check if there's a next level to advance to
        return self.current_level < max(self.DIFFICULTY_LEVELS.keys())

    def advance_difficulty(self) -> bool:
        """
        Advance to the next difficulty level if appropriate.

        Returns:
            Boolean indicating whether advancement occurred
        """
        if self.should_advance():
            self.current_level += 1
            return True
        return False

    def get_current_level(self) -> int:
        """
        Get the current difficulty level.

        Returns:
            Current level as an integer
        """
        return self.current_level

    def get_num_levels(self) -> int:
        """
        Get the total number of difficulty levels.

        Returns:
            Total number of levels
        """
        return len(self.DIFFICULTY_LEVELS)

    def get_level_description(self, level: Optional[int] = None) -> str:
        """
        Get a description of the specified difficulty level.

        Args:
            level: The level to describe (default: current level)

        Returns:
            String description of the level
        """
        if level is None:
            level = self.current_level

        level_descriptions = {
            1: "Basic arithmetic operations (addition, subtraction, multiplication, division)",
            2: "Basic operations with fractions and pre-algebra",
            3: "Basic geometry and more algebra",
            4: "More advanced algebra and basic statistics",
            5: "Vectors, matrices, and solid geometry",
            6: "Advanced topics (calculus, statistics, computer science)",
            7: "Most complex topics (complex numbers, advanced operations)",
        }

        return level_descriptions.get(
            level, f"Custom level with IDs: {self.DIFFICULTY_LEVELS.get(level, [])}"
        )

    def reset(self, level: int = 1) -> None:
        """
        Reset the curriculum to a specific level and clear performance history.

        Args:
            level: The level to reset to (default: 1)
        """
        if level not in self.DIFFICULTY_LEVELS:
            raise ValueError(
                f"Invalid level: {level}. Available levels: {list(self.DIFFICULTY_LEVELS.keys())}"
            )

        self.current_level = level
        self.performance_history = {lvl: [] for lvl in self.DIFFICULTY_LEVELS.keys()}

    def get_generator_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all available generators.

        Returns:
            List of dictionaries containing generator information
        """
        generators = []
        gen_list = mathgenerator.getGenList()

        for gen in gen_list:
            # Find which level this generator belongs to
            level = None
            for lvl, generator_ids in self.DIFFICULTY_LEVELS.items():
                if gen[0] in generator_ids:
                    level = lvl
                    break

            generators.append(
                {
                    "id": gen[0],
                    "name": gen[1],
                    "function": gen[3],
                    "subject": gen[4],
                    "params": gen[5],
                    "difficulty_level": level,
                }
            )

        return generators
