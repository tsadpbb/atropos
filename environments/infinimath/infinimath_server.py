#!/usr/bin/env python3
import asyncio
import logging
import os
import argparse

from dotenv import load_dotenv
from openai import OpenAI

from environments.infinimath.infinimath_env import (
    InfiniteMathEnv,
    InfiniteMathEnvConfig,
)
from atroposlib.envs.base import OpenaiConfig
from atroposlib.utils.config_handler import ConfigHandler

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="InfiniteMath environment server")
    parser.add_argument(
        "--config",
        type=str,
        default="infinimath",
        help="Configuration file name (without .yaml extension or path for configs/envs/ directory, or full path)",
    )
    return parser.parse_args()


async def main():
    logger.info("Starting InfiniteMath environment server")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize config handler and load configuration
    config_handler = ConfigHandler()
    
    # Determine config path
    if os.path.isabs(args.config) or "/" in args.config or args.config.endswith(".yaml"):
        config_path = args.config
    else:
        # short form that defaults to the envs directory
        config_path = os.path.join(
            config_handler.config_dir, f"envs/{args.config}.yaml"
        )
    
    logger.info(f"Loading configuration from: {config_path}")
    
    try:
        with open(config_path, "r") as f:
            import yaml
            raw_config = yaml.safe_load(f)
            logger.info(f"Loaded configuration successfully")
    except Exception as e:
        logger.error(f"Error loading config directly: {e}")
        logger.info("Falling back to default config handler")
        raw_config = config_handler.load_config(args)

    # Configure the InfiniteMath environment with values from config
    config = InfiniteMathEnvConfig(
        # Base environment parameters
        tokenizer_name=raw_config.get("tokenizer_name", "NousResearch/DeepHermes-3-Llama-3-8B-Preview"),
        group_size=raw_config.get("group_size", 1),
        use_wandb=raw_config.get("use_wandb", False),
        max_num_workers=raw_config.get("max_num_workers", 1),
        rollout_server_url=raw_config.get("rollout_server_url", "http://localhost:8000"),
        total_steps=raw_config.get("total_steps", 1),
        batch_size=raw_config.get("batch_size", 1),
        steps_per_eval=raw_config.get("steps_per_eval", 2),
        max_token_length=raw_config.get("max_token_length", 4096),
        wandb_name=raw_config.get("wandb_name", "infinite_math_test"),
        ensure_scores_are_not_same=raw_config.get("ensure_scores_are_not_same", False),
        
        # InfiniteMath specific parameters
        starting_level=raw_config.get("infinimath", {}).get("starting_level", 1),
        progress_threshold=raw_config.get("infinimath", {}).get("progress_threshold", 0.7),
        min_evaluations=raw_config.get("infinimath", {}).get("min_evaluations", 3),
        correct_reward=raw_config.get("infinimath", {}).get("correct_reward", 1.0),
        incorrect_reward=raw_config.get("infinimath", {}).get("incorrect_reward", -0.5),
        apply_length_penalty=raw_config.get("infinimath", {}).get("apply_length_penalty", True),
        length_threshold_ratio=raw_config.get("infinimath", {}).get("length_threshold_ratio", 0.6),
        temperature=raw_config.get("infinimath", {}).get("temperature", 0.7),
        top_p=raw_config.get("infinimath", {}).get("top_p", 0.9),
        reward_functions=raw_config.get("infinimath", {}).get("reward_functions", ["accuracy", "format", "boxed"]),
        accuracy_reward_weight=raw_config.get("infinimath", {}).get("accuracy_reward_weight", 1.0),
        format_reward_weight=raw_config.get("infinimath", {}).get("format_reward_weight", 0.2),
        boxed_reward_weight=raw_config.get("infinimath", {}).get("boxed_reward_weight", 0.3),
    )

    # Server configuration from config file or defaults
    server_configs = []
    
    if "server_configs" in raw_config:
        for server_config in raw_config["server_configs"]:
            api_key = server_config.get("api_key", os.environ.get("OPENAI_API_KEY"))
            # Handle environment variable references like ${OPENAI_API_KEY}
            if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
                env_var = api_key[2:-1]
                api_key = os.environ.get(env_var, "")
            
            server_configs.append(
                OpenaiConfig(
                    model_name=server_config.get("model_name", "gpt-4.1-nano"),
                    base_url=server_config.get("base_url", None),
                    api_key=api_key,
                    num_requests_for_eval=server_config.get("num_requests_for_eval", 70),
                )
            )
    else:
        # Default configuration if not specified in config file
        server_configs.append(
            OpenaiConfig(
                model_name="gpt-4.1-nano",
                base_url=None,
                api_key=os.environ.get("OPENAI_API_KEY"),
                num_requests_for_eval=70,
            )
        )

    # Create the environment
    env = InfiniteMathEnv(
        config=config,
        server_configs=server_configs,
        slurm=False,
    )

    # Setup the environment
    await env.setup()
    logger.info("Environment setup complete")

    # Log the number of evaluation problems
    total_problems = sum(len(probs) for probs in env.eval_problems.values())
    logger.info(
        f"Using {total_problems} evaluation problems across {len(env.eval_problems)} difficulty levels"
    )

    # Get a math problem
    item = await env.get_next_item()
    problem_prompt, solution, generator_id = item

    logger.info(f"Problem: {dict(problem_prompt[0])['content']}")
    logger.info(f"Solution: {solution}")

    # Collect trajectories
    logger.info("Collecting trajectories...")
    trajectories_data, backlog = await env.collect_trajectories(item)
    
    # Score the collected trajectories
    logger.info("Scoring trajectories...")
    scored_data = await env.score(trajectories_data)
    
    input("Press Enter to continue...")
    # Print scores
    logger.info(f"Scores: {scored_data['scores']}")

    # Log the correct/incorrect counts
    correct_count = sum(1 for score in scored_data["scores"] if score > 0)
    logger.info(f"Correct answers: {correct_count}/{len(scored_data['scores'])}")

    # Test evaluation function specifically
    logger.info("\n=== Testing Evaluation Function ===")

    # Record the current level
    initial_level = env.curriculum.get_current_level()
    logger.info(f"Current level before evaluation: {initial_level}")
    logger.info(f"Level description: {env.curriculum.get_level_description()}")
    logger.info(f"Progress threshold: {env.curriculum.progress_threshold}")
    logger.info(f"Min evaluations needed: {env.curriculum.min_evaluations}")

    # Run the evaluate method
    eval_metrics = await env.evaluate()

    # Display evaluation results
    logger.info("Evaluation metrics:")
    for metric_name, metric_value in eval_metrics:
        logger.info(f"  - {metric_name}: {metric_value}")

    # Check if the level advanced
    new_level = env.curriculum.get_current_level()
    if new_level > initial_level:
        logger.info(f"Successfully advanced to level {new_level}!")
        logger.info(f"New level description: {env.curriculum.get_level_description()}")
    else:
        # Show current progress toward advancement
        current_level = env.curriculum.get_current_level()
        if current_level in env.curriculum.performance_history:
            history = env.curriculum.performance_history[current_level]
            if len(history) >= env.curriculum.min_evaluations:
                recent_history = history[-env.curriculum.min_evaluations :]
                success_rate = sum(recent_history) / len(recent_history)
                logger.info(
                    f"Current success rate: {success_rate:.2f} (need {env.curriculum.progress_threshold} to advance)"
                )
            else:
                logger.info(
                    f"Need more evaluations: {len(history)}/{env.curriculum.min_evaluations}"
                )

    # Show all levels and their performance history
    logger.info("\nPerformance history by level:")
    for level in sorted(env.curriculum.performance_history.keys()):
        history = env.curriculum.performance_history[level]
        if history:
            success_rate = sum(history) / len(history)
            logger.info(
                f"  Level {level}: {success_rate:.2f} ({sum(history)}/{len(history)} correct)"
            )
        else:
            logger.info(f"  Level {level}: No data")

    # Test curriculum advancement with simulated performance history
    logger.info("\n=== Testing Curriculum Advancement ===")

    # Simulate good performance at current level
    for _ in range(env.config.min_evaluations):
        # Get a problem from current level
        item = await env.get_next_item()
        _, _, generator_id = item

        # Record positive performance
        env.curriculum.record_performance(generator_id, True)

    # Try to advance difficulty
    did_advance = env.curriculum.advance_difficulty()
    new_level = env.curriculum.get_current_level()

    logger.info(f"Curriculum advancement test:")
    logger.info(f"  - Starting level: {initial_level}")
    logger.info(f"  - Recorded {env.config.min_evaluations} correct answers")
    logger.info(f"  - Did advance: {did_advance}")
    logger.info(f"  - New level: {new_level}")

    logger.info("Test server completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
