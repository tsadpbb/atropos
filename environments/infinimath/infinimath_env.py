import asyncio
import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from trajectoryhandler.envs.base import (
    BaseEnv,
    BaseEnvConfig,
    OpenaiConfig,
    ScoredDataGroup,
)
from trajectoryhandler.envs.reward_fns import registry
from trajectoryhandler.envs.reward_fns.combined_reward import CombinedReward
from trajectoryhandler.utils.tokenize_for_trainer import tokenize_for_trainer

from .curriculum import MathCurriculum

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

system_prompt = """You are an expert mathematician. You need to solve the given math problem step-by-step, showing your reasoning clearly. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your final answer in a LaTeX format using \\boxed{your answer here}.

The problems will be given in a LaTeX format, so be sure to follow the LaTeX syntax when writing your answer (although no $ delimiters are necessary).

Follow these steps:
1. Understand the problem carefully
2. Plan your approach
3. Execute the calculations step-by-step
4. Verify your solution
5. Express the final answer as \\boxed{your answer here}

You may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering.

Your answer format should be:
<think>
[Your detailed step-by-step reasoning process here]
</think>

\\boxed{your final answer here}

Remember to format your final answer correctly as this is important for evaluation."""


class InfiniteMathEnvConfig(BaseEnvConfig):
    """Configuration for the InfiniteMath environment."""

    # Curriculum parameters
    starting_level: int = 1
    progress_threshold: float = 0.8
    min_evaluations: int = 5

    # Environment parameters
    max_attempts_per_problem: int = 3
    correct_reward: float = 1.0
    incorrect_reward: float = -1.0

    # Length penalty parameters
    apply_length_penalty: bool = True
    length_threshold_ratio: float = (
        0.5  # Percentage of max_token_length before penalties apply
    )

    # Completion parameters
    temperature: float = 0.7
    top_p: float = 0.9

    # Reward functions
    reward_functions: List[Union[str, Dict[str, Any]]] = ["accuracy", "format", "boxed"]
    accuracy_reward_weight: float = 1.0  # Weight for the accuracy reward
    format_reward_weight: float = (
        0.2  # Weight for the format reward relative to correctness
    )
    boxed_reward_weight: float = (
        0.3  # Weight for the boxed answer reward relative to correctness
    )


class InfiniteMathEnv(BaseEnv):
    """Environment for procedurally generated math problems with curriculum advancement."""

    def __init__(
        self,
        config: InfiniteMathEnvConfig,
        server_configs: Union[List[OpenaiConfig], OpenaiConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config = config  # Override with our specific config class

        # Initialize tracking metrics
        self.percent_correct_buffer = []
        self.level_correct_buffer = {
            i: [] for i in range(1, 8)
        }  # Track correctness for each level
        self.eval_metrics = []

        # Curriculum will be initialized in setup()
        self.curriculum = None

        # Set the system prompt
        self.system_prompt = system_prompt

        # Initialize reward function
        self.reward_function = self._initialize_reward_function()

    def _initialize_reward_function(self):
        """Initialize the combined reward function for scoring."""
        if hasattr(self.config, "reward_functions") and self.config.reward_functions:
            # Configure parameters for specific reward functions
            reward_configs = []
            
            for reward_func in self.config.reward_functions:
                if isinstance(reward_func, str):
                    # String name case - handle known rewards with custom params
                    if reward_func == "accuracy":
                        # Configure accuracy reward
                        accuracy_config = {
                            "type": "accuracy", 
                            "weight": self.config.accuracy_reward_weight,
                            "params": {
                                "split_on_think_tag": True,  # Only look at what's after </think>
                                "tolerance": 1e-6,  # Tolerance for number comparisons
                            },
                        }
                        logger.info(f"Adding accuracy reward with config: {accuracy_config}")
                        reward_configs.append(accuracy_config)
                    elif reward_func == "format":
                        # Configure format reward with think tags and explicit weight
                        format_config = {
                            "type": "format", 
                            "weight": self.config.format_reward_weight,
                            "params": {
                                "preferred_tags": ["think"],
                            },
                        }
                        logger.info(f"Adding format reward with config: {format_config}")
                        reward_configs.append(format_config)
                    elif reward_func == "boxed":
                        # Configure boxed reward with proper parameters and explicit weight
                        boxed_config = {
                            "type": "boxed",
                            "weight": self.config.boxed_reward_weight,
                            "params": {
                                "require_outside_think": True,
                            },
                        }
                        logger.info(f"Adding boxed reward with config: {boxed_config}")
                        reward_configs.append(boxed_config)
                    else:
                        # Pass through other reward functions as is
                        logger.info(f"Adding generic reward function: {reward_func}")
                        reward_configs.append(reward_func)
                else:
                    # Dict case - pass through as is
                    logger.info(f"Adding reward config: {reward_func}")
                    reward_configs.append(reward_func)
            
            # Create the reward function(s)
            if len(reward_configs) == 1:
                logger.info(f"Creating single reward function: {reward_configs[0]}")
                return registry.create(reward_configs[0])
            else:
                logger.info(f"Creating combined reward function with {len(reward_configs)} rewards")
                # Add explicit normalization to sum to 1.0
                return CombinedReward(rewards=reward_configs, normalization="none")

    async def setup(self):
        """Initialize the environment and curriculum."""
        logger.info("Setting up InfiniteMathEnv")

        # Initialize curriculum
        self.curriculum = MathCurriculum(
            starting_level=self.config.starting_level,
            progress_threshold=self.config.progress_threshold,
            min_evaluations=self.config.min_evaluations,
        )

        # Generate some test problems for each level for evaluation
        self.eval_problems = {}
        for level in range(1, 8):
            self.eval_problems[level] = []
            temp_curriculum = MathCurriculum(starting_level=level)
            # Generate 10 test problems for each level
            attempts = 0
            max_attempts_per_level = 20  # Try at most 20 problems to get 10 valid ones

            while (
                len(self.eval_problems[level]) < 10
                and attempts < max_attempts_per_level
            ):
                try:
                    problem, solution, generator_id = temp_curriculum.get_problem()
                    # Strip LaTeX delimiters
                    problem = self.strip_latex_delimiters(problem)
                    solution = self.strip_latex_delimiters(solution)
                    self.eval_problems[level].append((problem, solution, generator_id))
                except Exception as e:
                    logger.warning(
                        f"Error generating evaluation problem for level {level}: {e}"
                    )
                attempts += 1

            logger.info(
                f"Generated {len(self.eval_problems[level])} evaluation problems for level {level}"
            )

        # If any levels have no problems, add a simple fallback
        for level in range(1, 8):
            if not self.eval_problems[level]:
                logger.warning(
                    f"No valid evaluation problems for level {level}, adding fallback"
                )
                if level == 1:
                    self.eval_problems[level].append(("What is 2 + 3?", "5", 0))
                elif level == 2:
                    self.eval_problems[level].append(
                        ("What is the square root of 16?", "4", 6)
                    )
                elif level == 3:
                    self.eval_problems[level].append(
                        (
                            "What is the area of a triangle with base 6 and height 8?",
                            "24",
                            18,
                        )
                    )
                elif level == 4:
                    self.eval_problems[level].append(
                        ("What is the solution to x + 5 = 12?", "7", 26)
                    )
                elif level == 5:
                    self.eval_problems[level].append(
                        ("What is the volume of a cube with side length 3?", "27", 33)
                    )
                elif level == 6:
                    self.eval_problems[level].append(
                        ("What is 5 factorial?", "120", 31)
                    )
                else:
                    self.eval_problems[level].append(("What is |3 - 10|?", "7", 71))

    def strip_latex_delimiters(self, text: str) -> str:
        """Strip LaTeX delimiters ($...$) from text."""
        # Handle both inline expressions $...$ and expressions that make up the entire string
        return re.sub(r"\$(.*?)\$", r"\1", text)

    def save_checkpoint(self, step, data=None):
        """Save curriculum state in checkpoint."""
        if data is None:
            data = {}

        # Save curriculum state
        data["curriculum_level"] = self.curriculum.get_current_level()
        data["performance_history"] = {
            str(k): v for k, v in self.curriculum.performance_history.items()
        }

        super().save_checkpoint(step, data)

    def load_checkpoint(self):
        """Load curriculum state from checkpoint."""
        super().load_checkpoint()

        # Check if we have curriculum data in the checkpoint
        checkpoint_path = f"{self.checkpoint_dir}/env_checkpoints/{self.wandb_prepend}/step-{self.curr_step}.json"
        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)

            # Restore curriculum state if available
            if "curriculum_level" in data:
                level = data["curriculum_level"]
                self.curriculum.current_level = level

            if "performance_history" in data:
                # Convert string keys back to integers
                self.curriculum.performance_history = {
                    int(k): v for k, v in data["performance_history"].items()
                }
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Log overall correct percentage
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / max(1, len(self.percent_correct_buffer))
        except ZeroDivisionError:
            pass

        # Log per-level metrics
        for level, buffer in self.level_correct_buffer.items():
            if buffer:
                wandb_metrics[f"train/level_{level}_correct"] = sum(buffer) / len(
                    buffer
                )
                wandb_metrics[f"train/level_{level}_count"] = len(buffer)

        # Log current level and curriculum progress
        if self.curriculum:
            current_level = self.curriculum.get_current_level()
            max_level = max(self.curriculum.DIFFICULTY_LEVELS.keys())

            wandb_metrics["curriculum/current_level"] = current_level
            wandb_metrics["curriculum/max_level"] = max_level
            wandb_metrics["curriculum/progress_percent"] = (
                current_level / max_level
            ) * 100

            # Log level description
            wandb_metrics["curriculum/level_description"] = (
                self.curriculum.get_level_description()
            )

            # Log performance history for current level
            if current_level in self.curriculum.performance_history:
                history = self.curriculum.performance_history[current_level]
                if history:
                    recent_history = history[
                        -min(len(history), self.curriculum.min_evaluations) :
                    ]
                    if recent_history:
                        success_rate = sum(recent_history) / len(recent_history)
                        wandb_metrics["curriculum/current_level_success_rate"] = (
                            success_rate
                        )
                        wandb_metrics["curriculum/threshold_to_advance"] = (
                            self.curriculum.progress_threshold
                        )
                        wandb_metrics["curriculum/remaining_to_threshold"] = max(
                            0, self.curriculum.progress_threshold - success_rate
                        )

        # Log reward function metrics
        if hasattr(self, "reward_function") and self.wandb:
            if hasattr(self.reward_function, "set_wandb_logger"):
                self.reward_function.set_wandb_logger(self.wandb)
            
            # Log the reward configurations
            if isinstance(self.config.reward_functions, list) and self.config.reward_functions:
                # Log the reward configuration
                wandb_metrics["reward/format_reward_enabled"] = "format" in self.config.reward_functions
                wandb_metrics["reward/boxed_reward_enabled"] = "boxed" in self.config.reward_functions
                
                if hasattr(self.config, "format_reward_weight"):
                    wandb_metrics["reward/format_reward_weight"] = self.config.format_reward_weight
                
                if hasattr(self.config, "boxed_reward_weight"):
                    wandb_metrics["reward/boxed_reward_weight"] = self.config.boxed_reward_weight

        # Add eval metrics
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]

        # Reset buffers
        self.percent_correct_buffer = []
        for level in self.level_correct_buffer:
            self.level_correct_buffer[level] = []
        self.eval_metrics = []

        # Call the parent method to handle remaining metrics
        await super().wandb_log(wandb_metrics)

    async def get_next_item(self):
        """Get the next problem based on current curriculum level."""
        problem, solution, generator_id = self.curriculum.get_problem()

        # Strip LaTeX delimiters from problem and solution
        problem = self.strip_latex_delimiters(problem)
        solution = self.strip_latex_delimiters(solution)

        # Create a message with the problem
        prompt = tuple([frozenset({"role": "user", "content": problem}.items())])

        # Return the problem with metadata
        return (prompt, solution, generator_id)

    async def evaluate(self, *args, **kwargs):
        """Evaluate the model on test problems at the current curriculum level."""
        current_level = self.curriculum.get_current_level()
        logger.info(f"Starting evaluation for curriculum level {current_level}")

        # Only evaluate problems at the current level
        eval_tasks = []
        eval_generator_ids = []
        if current_level in self.eval_problems:
            for problem, solution, generator_id in self.eval_problems[current_level]:
                eval_tasks.append(
                    self.evaluate_single_problem(problem, solution, current_level)
                )
                eval_generator_ids.append(generator_id)

        if not eval_tasks:
            logger.warning(
                f"No evaluation problems available for level {current_level}"
            )
            return []

        # Run evaluation tasks
        logger.info(f"Evaluating {len(eval_tasks)} problems at level {current_level}")
        results = await asyncio.gather(*eval_tasks)

        # Calculate accuracy for the current level
        correct_count = sum(1 for _, is_correct in results if is_correct)
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0

        logger.info(
            f"Level {current_level} accuracy: {accuracy:.2f} ({correct_count}/{total_count})"
        )

        # Record metrics for the current level
        self.eval_metrics.append((f"eval/level_{current_level}_accuracy", accuracy))
        self.eval_metrics.append(("eval/current_level", current_level))

        # Record the actual evaluation results in the curriculum's performance history
        for i, (_, is_correct) in enumerate(results):
            if i < len(eval_generator_ids):
                # Record the actual result
                self.curriculum.record_performance(eval_generator_ids[i], is_correct)
            else:
                # Fallback if somehow the lists are different lengths
                sample_generator_id = random.choice(
                    self.curriculum.DIFFICULTY_LEVELS[current_level]
                )
                self.curriculum.record_performance(sample_generator_id, is_correct)

        # Try to advance to the next level
        advanced = self.curriculum.advance_difficulty()
        new_level = self.curriculum.get_current_level()

        if advanced:
            logger.info(f"Advanced from level {current_level} to level {new_level}!")
            self.eval_metrics.append(("eval/advanced_level", 1))
        else:
            logger.info(f"Remaining at level {current_level}")
            self.eval_metrics.append(("eval/advanced_level", 0))

        return self.eval_metrics

    async def evaluate_single_problem(
        self, problem: str, solution: str, level: int
    ) -> Tuple[int, bool]:
        """Evaluate a single problem."""
        try:
            logger.debug(f"Evaluating level {level} problem: {problem[:30]}...")

            # Convert messages to a single prompt using the tokenizer
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": problem},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

            # Add prefilled thinking starter
            prefill = "\n<think>\n"
            prefilled_prompt = prompt + prefill

            # Generate completion using the prompt
            logger.debug(f"Requesting completion for problem: {problem[:30]}...")
            completion = await self.server.completion(
                prompt=prefilled_prompt,
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=0.0,  # Use 0 temperature for deterministic results
                top_p=1.0,
                split="eval",
            )

            # Extract the completion text and prepend the thinking starter
            model_answer = prefill + (
                completion.choices[0].text
                if hasattr(completion.choices[0], "text")
                else completion.choices[0].message.content
            )

            # Check if the answer is correct
            is_correct = self.check_answer(model_answer, solution)
            logger.debug(f"Problem evaluated: level={level}, correct={is_correct}")

            return level, is_correct
        except Exception as e:
            logger.error(f"Error evaluating problem: {e}")
            # Return a failed result in case of error
            return level, False

    def check_answer(self, model_answer: str, solution: str) -> bool:
        """Check if the model's answer matches the solution."""
        # Extract the part after the thinking block
        after_think_part = (
            model_answer.split("</think>")[-1].strip()
            if "</think>" in model_answer
            else model_answer
        )

        # Extract the boxed answer if present
        boxed_answer = self.extract_boxed_answer(after_think_part)
        if not boxed_answer:
            # Try to find the answer in the last line
            lines = after_think_part.strip().split("\n")
            if lines:
                boxed_answer = lines[-1].strip()

        # Clean up answers for comparison (remove spaces, convert to lowercase)
        model_clean = self.clean_for_comparison(
            boxed_answer if boxed_answer else after_think_part
        )
        solution_clean = self.clean_for_comparison(solution)

        # Check if they match
        return model_clean == solution_clean

    def extract_boxed_answer(self, text: str) -> Optional[str]:
        """Extract answer from a LaTeX boxed expression."""
        # Try to find boxed content
        boxed_match = re.search(r"\\boxed{([^}]*)}", text)
        if boxed_match:
            return boxed_match.group(1)
        return None

    def clean_for_comparison(self, text: str) -> str:
        """Clean text for comparison."""
        # Remove LaTeX commands, spaces, commas, and convert to lowercase
        cleaned = re.sub(r"\\[a-zA-Z]+", "", text)
        cleaned = re.sub(r"[,\s]", "", cleaned)
        cleaned = cleaned.lower()
        return cleaned

    async def collect_trajectories(self, item) -> Tuple[List, List]:
        """Collect trajectories for the current item."""
        # Extract information from the item
        problem_prompt, solution, generator_id = item

        # Create prompt using tokenizer's chat template
        # Add prefilled thinking starter
        prefill = "\n<think>\n"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": dict(problem_prompt[0])["content"]},
            {"role": "assistant", "content": prefill},
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        # Generate completions using completion API
        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        # Prepare data for scoring
        to_score = []
        
        # Track level for metrics
        level = None
        for lvl, generator_ids in self.curriculum.DIFFICULTY_LEVELS.items():
            if generator_id in generator_ids:
                level = lvl
                break

        # Process each completion
        for i, completion in enumerate(completions.choices):
            # Get the completion text and prepend the thinking starter
            model_answer = prefill + (
                completion.text
                if hasattr(completion, "text")
                else completion.message.content
            )
            print("model_answer", model_answer)

            # Build complete message sequence
            full_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": dict(problem_prompt[0])["content"]},
                {"role": "assistant", "content": model_answer},
            ]
            
            # Add to scoring list
            to_score.append((full_messages, solution, generator_id, level))
            
        # Record performance in curriculum for each item we're scoring
        # This will be called again after scoring, but that's fine

        # No additional items for backlog
        backlog = []

        return to_score, backlog
        
    async def score(self, rollout_group_data) -> ScoredDataGroup:
        """Score the collected trajectories."""
        scored_data = ScoredDataGroup()
        scored_data["tokens"] = []
        scored_data["masks"] = []
        scored_data["scores"] = []
        scored_data["messages"] = []
        
        # Format completions for reward function evaluation
        format_completions = []
        
        # Process each item in the rollout data
        for messages, solution, generator_id, level in rollout_group_data:
            # Extract the model's answer
            model_answer = messages[-1]["content"]
            
            # Add to format completions list for reward function
            format_completions.append([{"role": "assistant", "content": model_answer}])
            
            # Record performance in curriculum based on the answer and solution
            # This will be updated after the reward functions are applied
        
        # Apply all reward functions
        reward_scores = []
        unweighted_scores = []
        if hasattr(self, "reward_function") and self.reward_function:
            try:
                # Apply the reward function (which may be a combined reward)
                reward_scores = self.reward_function(format_completions, solution=solution)
                logger.info(f"Reward scores: {reward_scores}")
                
                # Debug individual rewards if it's a combined reward
                if hasattr(self.reward_function, "rewards"):
                    logger.info(f"Combined reward with {len(self.reward_function.rewards)} components")
                    for i, reward in enumerate(self.reward_function.rewards):
                        if hasattr(reward, "compute"):
                            # Get raw unweighted scores
                            raw_scores = reward.compute(format_completions, solution=solution)
                            if hasattr(reward, "weight"):
                                logger.info(f"Reward {i} ({type(reward).__name__}): raw={raw_scores}, weight={reward.weight}")
                            else:
                                logger.info(f"Reward {i} ({type(reward).__name__}): raw={raw_scores}")
                else:
                    logger.info(f"Using single reward: {type(self.reward_function).__name__}")
                
            except Exception as e:
                logger.error(f"Error applying reward functions: {e}")
                logger.exception(e)
                reward_scores = [0.0] * len(format_completions)
        
        # Now update curriculum based on accuracy reward results
        for i, (messages, solution, generator_id, level) in enumerate(rollout_group_data):
            # Extract accuracy from the combined reward if available
            is_correct = False
            if reward_scores and hasattr(self.reward_function, "rewards"):
                for reward in self.reward_function.rewards:
                    if type(reward).__name__ == "AccuracyReward":
                        # Get raw scores from accuracy reward
                        accuracy_scores = reward.compute(format_completions, solution=solution)
                        is_correct = accuracy_scores[i] > 0
                        break
            
            # Record answer correctness for tracking
            self.percent_correct_buffer.append(1 if is_correct else 0)
            if level is not None:
                self.level_correct_buffer[level].append(1 if is_correct else 0)
            
            # Record performance in curriculum
            self.curriculum.record_performance(generator_id, is_correct)
        
        # Combine scores and add to scored data
        for i, (messages, _, _, _) in enumerate(rollout_group_data):
            # Use the reward score directly (all weights are applied)
            combined_score = reward_scores[i] if reward_scores else 0.0
            
            logger.info(f"Final score for item {i}: {combined_score}")
            
            # Tokenize for the trainer
            tokens_dict = tokenize_for_trainer(
                self.tokenizer,
                messages,
                None,
            )
            
            # Add to scored data
            scored_data["tokens"].append(tokens_dict["tokens"])
            scored_data["masks"].append(tokens_dict["masks"])
            scored_data["scores"].append(combined_score)
            scored_data["messages"].append(messages)
        
        # Advance difficulty if criteria met
        self.curriculum.advance_difficulty()
        
        return scored_data


if __name__ == "__main__":
    import asyncio

    async def main():
        config = InfiniteMathEnvConfig(
            tokenizer_name="NousResearch/Nous-Hermes-2-Yi-34B",
            group_size=8,
            use_wandb=True,
            max_num_workers=64,
            rollout_server_url="http://localhost:8000",
            total_steps=10000,
            batch_size=1024,
            steps_per_eval=25,
            max_token_length=4096,
            inference_weight=1.0,
            wandb_name="infinite_math",
            data_path_to_save_groups="data/infinite_math_groups.jsonl",
            # InfiniteMath specific config
            starting_level=1,
            progress_threshold=0.8,
            min_evaluations=10,
            correct_reward=1.0,
            incorrect_reward=-0.5,
            apply_length_penalty=True,
            length_threshold_ratio=0.6,
            # Completion parameters
            temperature=0.7,
            top_p=0.9,
            # Reward function configuration - use name directly
            reward_functions=["accuracy", "format", "boxed"],
            accuracy_reward_weight=1.0,
            format_reward_weight=0.2,
            boxed_reward_weight=0.3,
        )

        openai_config = OpenaiConfig(
            model_name="NousResearch/Nous-Hermes-2-Yi-34B",
            base_url="http://localhost:9004/v1",
            api_key="x",
            num_requests_for_eval=64,
        )

        env = InfiniteMathEnv(
            config=config,
            server_configs=[openai_config],
            slurm=False,
        )

        await env.env_manager()

    asyncio.run(main())
