#!/usr/bin/env python3
"""
Test script for the Infinite Math curriculum manager.
This script tests the core functionality of both the MathCurriculum and InfiniteMath classes.
"""

import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

# Use relative imports
from .curriculum import MathCurriculum
from .infinimath import InfiniteMath


def test_curriculum_initialization():
    """Test that the curriculum initializes correctly with different levels."""
    print("\n=== Testing Curriculum Initialization ===")

    # Test default initialization
    curriculum = MathCurriculum()
    assert curriculum.get_current_level() == 1, "Default level should be 1"
    print("✓ Default initialization successful")

    # Test initialization with specific level
    curriculum = MathCurriculum(starting_level=3)
    assert curriculum.get_current_level() == 3, "Starting level should be 3"
    print("✓ Custom level initialization successful")

    # Test initialization with invalid level
    try:
        curriculum = MathCurriculum(starting_level=10)
        print("✗ Invalid level initialization should fail")
    except ValueError:
        print("✓ Invalid level initialization correctly raises ValueError")


def test_problem_generation():
    """Test that problems are generated correctly at different difficulty levels."""
    print("\n=== Testing Problem Generation ===")

    # Test problem generation at different levels
    for level in range(1, 8):
        curriculum = MathCurriculum(starting_level=level)
        problem, solution, generator_id = curriculum.get_problem()

        # Verify we got a problem and solution
        assert (
            isinstance(problem, str) and len(problem) > 0
        ), f"Problem at level {level} should be a non-empty string"
        assert solution is not None, f"Solution at level {level} should not be None"

        # Verify the generator ID belongs to the correct level
        assert (
            generator_id in curriculum.DIFFICULTY_LEVELS[level]
        ), f"Generator ID {generator_id} should be in level {level}"

        print(
            f"✓ Level {level} problem generated: {problem[:50]}{'...' if len(problem) > 50 else ''}"
        )
        print(f"  Solution: {solution}")
        print(f"  Generator ID: {generator_id}")


def test_performance_tracking():
    """Test performance tracking and level advancement."""
    print("\n=== Testing Performance Tracking and Advancement ===")

    # Create curriculum with test parameters
    curriculum = MathCurriculum(
        starting_level=1, progress_threshold=0.7, min_evaluations=3
    )

    # Record some correct answers (not enough to advance)
    generator_id = curriculum.DIFFICULTY_LEVELS[1][0]  # Get a generator from level 1
    curriculum.record_performance(generator_id, True)
    curriculum.record_performance(generator_id, True)

    # Check if we advance (should not)
    did_advance = curriculum.advance_difficulty()
    assert not did_advance, "Should not advance with only 2 evaluations"
    assert curriculum.get_current_level() == 1, "Level should still be 1"
    print("✓ Correctly did not advance with insufficient evaluations")

    # Add one more correct answer (now should advance)
    curriculum.record_performance(generator_id, True)
    did_advance = curriculum.advance_difficulty()
    assert did_advance, "Should advance with 3 correct evaluations (100% success rate)"
    assert curriculum.get_current_level() == 2, "Level should be 2 after advancement"
    print("✓ Correctly advanced to level 2 after sufficient success")

    # Test with too low success rate
    curriculum = MathCurriculum(
        starting_level=1, progress_threshold=0.7, min_evaluations=3
    )
    generator_id = curriculum.DIFFICULTY_LEVELS[1][0]
    curriculum.record_performance(generator_id, True)  # 1 correct
    curriculum.record_performance(generator_id, False)  # 1 incorrect
    curriculum.record_performance(generator_id, False)  # 1 incorrect

    did_advance = curriculum.advance_difficulty()
    assert (
        not did_advance
    ), "Should not advance with 33% success rate when threshold is 70%"
    print("✓ Correctly did not advance with insufficient success rate")

    # Test advancement at the highest level
    curriculum = MathCurriculum(
        starting_level=7, progress_threshold=0.7, min_evaluations=3
    )
    generator_id = curriculum.DIFFICULTY_LEVELS[7][0]
    curriculum.record_performance(generator_id, True)
    curriculum.record_performance(generator_id, True)
    curriculum.record_performance(generator_id, True)

    did_advance = curriculum.advance_difficulty()
    assert not did_advance, "Should not advance beyond the highest level"
    print("✓ Correctly did not advance beyond the highest level")


def test_level_descriptions():
    """Test that level descriptions are correct."""
    print("\n=== Testing Level Descriptions ===")

    curriculum = MathCurriculum()

    for level in range(1, 8):
        description = curriculum.get_level_description(level)
        assert (
            isinstance(description, str) and len(description) > 0
        ), f"Description for level {level} should be a non-empty string"
        print(f"✓ Level {level}: {description}")


def test_infinite_math_environment():
    """Test the InfiniteMath environment functionality."""
    print("\n=== Testing InfiniteMath Environment ===")

    # Initialize the environment
    env = InfiniteMath(starting_level=1, progress_threshold=0.7, min_evaluations=3)

    # Get the initial state
    state = env.get_state()
    assert "problem" in state, "State should include a problem"
    assert "current_level" in state, "State should include current level"
    print(
        f"✓ Initial state: Level {state['current_level']}, Problem: {state['problem'][:50]}{'...' if len(state['problem']) > 50 else ''}"
    )

    # Test answering a problem incorrectly
    result = env.submit_answer("wrong answer")
    assert "is_correct" in result, "Result should indicate correctness"
    assert not result["is_correct"], "Result should be incorrect"
    print("✓ Incorrect answer handled correctly")

    # Test answering a problem correctly
    # Note: We can't predict the correct answer, so we'll get it from the environment
    correct_solution = env.current_solution
    result = env.submit_answer(correct_solution)
    assert result["is_correct"], "Result should be correct"
    print("✓ Correct answer handled successfully")

    # Test resetting the environment
    env.reset(level=3)
    state = env.get_state()
    assert state["current_level"] == 3, "After reset, level should be 3"
    print(f"✓ Reset to level 3 successful")

    # Test getting difficulty stats
    stats = env.get_difficulty_stats()
    assert len(stats) == 7, "Should have stats for all 7 difficulty levels"
    print("✓ Difficulty statistics retrieved successfully")


def simulate_learning():
    """Simulate a learning agent improving performance over time."""
    print("\n=== Simulating Learning Process ===")

    env = InfiniteMath(starting_level=1, progress_threshold=0.7, min_evaluations=5)
    episodes = 30

    print(f"Starting simulation with {episodes} episodes")
    print(f"Initial level: {env.get_state()['current_level']}")

    for i in range(episodes):
        state = env.get_state()
        current_level = state["current_level"]

        # Simulated agent gradually improves - higher chance of correct answer over time
        # and higher chance at lower levels
        success_probability = min(0.5 + (i / episodes) + (1 / current_level), 0.95)

        # Simulate an answer
        is_correct = random.random() < success_probability

        # If we decide to be correct, use the actual solution
        if is_correct:
            answer = env.current_solution
        else:
            # Otherwise provide a wrong answer
            answer = "wrong answer"

        # Submit the answer
        result = env.submit_answer(answer)

        # Check for level advancement
        if result.get("did_advance_level", False):
            new_level = result["new_level"]
            print(
                f"Episode {i+1}: Advanced to level {new_level}! (success probability: {success_probability:.2f})"
            )
        elif i % 5 == 0:  # Print status occasionally
            print(
                f"Episode {i+1}: Still at level {current_level} (success probability: {success_probability:.2f})"
            )

    # Print final stats
    final_state = env.get_state()
    print(f"\nFinal level: {final_state['current_level']}")
    print(
        f"Overall accuracy: {final_state['correct_problems'] / final_state['total_problems']:.2%}"
    )

    # Print level-by-level stats
    stats = env.get_difficulty_stats()
    print("\nPerformance by level:")
    for level, level_stats in stats.items():
        if level_stats["problems_attempted"] > 0:
            success_rate = level_stats["success_rate"]
            if success_rate is not None:
                print(
                    f"Level {level}: {success_rate:.2%} success rate ({level_stats['problems_attempted']} problems)"
                )
            else:
                print(
                    f"Level {level}: Not enough data ({level_stats['problems_attempted']} problems)"
                )


def main():
    """Run all tests."""
    print("=== Starting Curriculum Manager Tests ===")

    try:
        test_curriculum_initialization()
        test_problem_generation()
        test_performance_tracking()
        test_level_descriptions()
        test_infinite_math_environment()
        simulate_learning()

        print("\n=== All tests completed successfully! ===")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
