"""
Pydantic Schema Following Environment

This environment trains models to generate JSON that adheres to Pydantic schemas.
It loads schemas dynamically from a HuggingFace dataset and validates model outputs.

Recent improvements (2025-06-01):
1. Fixed Pydantic ValidationError compatibility issues with proper exception handling
2. Enhanced JSON extraction with fallback methods for responses missing proper tags
3. Added comprehensive input validation for tokenization to prevent 'list' object errors
4. Improved system prompt with explicit formatting requirements and examples
5. Reduced max_token_length to prevent overly verbose responses
6. Added robust error handling throughout the scoring pipeline
7. Enhanced debug logging for better troubleshooting
8. MAJOR: Added strict thinking tag validation (similar to MCQA environment)
9. Enforces exactly one <think></think> section followed by <json_output></json_output>
10. Added detailed validation method with comprehensive error reporting

Key Features:
- Dynamic Pydantic model creation from dataset configurations
- Comprehensive data dumping for analysis
- Strict thinking tag validation for consistent response format
- Fallback JSON extraction for improved success rates
- Detailed debug logging and error handling
- Robust validation with detailed error messages

Response Format Requirements:
- Must use exactly ONE <think> opening tag and ONE </think> closing tag
- All reasoning must be inside the thinking tags
- JSON output must be in <json_output></json_output> tags after </think>
- No additional <think> tags allowed after the first </think> closing tag
"""

import asyncio
import json
import logging
import os
import random
import re
import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from uuid import UUID

import wandb
from datasets import load_dataset
from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    HttpUrl,
    ValidationError,
    field_validator,
    model_validator,
)
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# System prompt for the LLM
system_prompt = (
    "You are an AI assistant that generates JSON objects according to Pydantic schemas.\n"
    "You may use extremely long chains of thought to deeply consider the problem and deliberate "
    "with yourself via systematic reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> tags.\n\n"
    "CRITICAL: Your final JSON output MUST be enclosed within <json_output> </json_output> tags.\n"
    "The JSON must be valid and complete. Do not include any text after the closing </json_output> tag.\n"
    "Example format:\n"
    "<think>\nMy reasoning here...\n</think>\n\n"
    '<json_output>\n{"field1": "value1", "field2": "value2"}\n</json_output>\n\n'
    "Ensure the generated JSON strictly adheres to the Pydantic model schema and any specific field "
    "requirements provided in the user prompt. Generate all required fields for the model, and "
    "include optional fields if they make sense in the context or are specified."
)


class PydanticEnvConfig(BaseEnvConfig):
    """Custom config class for PydanticSchemaFollowingEnv with additional parameters."""

    dataset_name: str = Field(
        default="justus27/pydantic-adherance-test",
        description="Name of the HuggingFace dataset to load",
    )
    dataset_split: str = Field(
        default="train", description="Dataset split to use (train, test, validation)"
    )
    debug_logging: bool = Field(
        default=True, description="Enable detailed debug logging"
    )
    dump_rollouts: bool = Field(
        default=False,
        description="Whether to dump rollouts to JSONL files for analysis",
    )
    include_messages: bool = Field(
        default=True,
        description="Whether to include messages in the dataset for SFT data generation",
    )


class PydanticSchemaFollowingEnv(BaseEnv):
    env_config_cls = PydanticEnvConfig

    def __init__(
        self,
        config: PydanticEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()  # Tracks 1.0 scores
        self.eval_metrics = list()
        self.rollouts_for_wandb = []
        self.dataset_items: List[Dict[str, Any]] = []
        self.model_cache: Dict[str, Type[BaseModel]] = (
            {}
        )  # Cache for dynamically created models

        # Set up debug logging
        self.debug_logging = getattr(config, "debug_logging", True)
        if self.debug_logging:
            self.logger = logging.getLogger(f"{self.__class__.__name__}")
            self.logger.setLevel(logging.DEBUG)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            self.logger.info("Debug logging enabled for PydanticSchemaFollowingEnv")
        else:
            self.logger = logging.getLogger(f"{self.__class__.__name__}")
            self.logger.addHandler(logging.NullHandler())

        # Data dumping setup
        self.run_uuid = str(uuid.uuid4())

        # Buffer for saving rollouts - each item group contains rollouts for one dataset item
        # RolloutDetail: conversation, score, expected_json, model_name, problem_id
        RolloutDetail = Dict[str, Union[List[Dict[str, str]], float, str]]
        ItemGroup = Dict[str, Union[str, List[RolloutDetail]]]
        self.rollouts_to_save_buffer: List[ItemGroup] = []
        self.processed_item_count = 0

        # Create datadumps directory relative to this file
        self.datadumps_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "datadumps"
        )
        self.save_file_batch_num = 0

        if self.debug_logging:
            self.logger.info(
                f"Data dumping {'enabled' if config.dump_rollouts else 'disabled'}"
            )
            if config.dump_rollouts:
                self.logger.info(f"Rollouts will be saved to: {self.datadumps_dir}")

    @classmethod
    def config_init(cls) -> Tuple[PydanticEnvConfig, List[APIServerConfig]]:
        """Initialize configuration for the environment."""
        env_config = PydanticEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=20,
            max_token_length=1024 * 12,
            inference_weight=1.0,
            wandb_name="pydantic_schema_following",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            dataset_name="justus27/pydantic-adherance-test",
            dataset_split="train",
            debug_logging=True,  # Enable debug logging by default
            dump_rollouts=True,  # Enable data dumping by default
            include_messages=True,  # Ensure messages are included for SFT data generation
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            ),
        ]
        return env_config, server_configs

    def _create_pydantic_model_from_code(
        self, pydantic_config: str, model_name: str
    ) -> Type[BaseModel]:
        """
        Dynamically create a Pydantic model from the provided configuration code.
        This executes the pydantic_config string and extracts the target model.
        """
        if self.debug_logging:
            self.logger.debug(f"Creating Pydantic model '{model_name}' from config")
            self.logger.debug(f"Config length: {len(pydantic_config)} characters")

        # Check cache first
        cache_key = f"{model_name}_{hash(pydantic_config)}"
        if cache_key in self.model_cache:
            if self.debug_logging:
                self.logger.debug(f"Model '{model_name}' found in cache")
            return self.model_cache[cache_key]

        if self.debug_logging:
            self.logger.debug(
                f"Model '{model_name}' not in cache, creating new instance"
            )

        # Create a namespace for executing the pydantic config
        namespace = {
            "BaseModel": BaseModel,
            "model_validator": model_validator,
            "ConfigDict": ConfigDict,
            "ValidationError": ValidationError,
            "HttpUrl": HttpUrl,
            "EmailStr": EmailStr,
            "Field": Field,
            "field_validator": field_validator,
            "List": List,
            "Dict": Dict,
            "Optional": Optional,
            "Union": Union,
            "Any": Any,
            "Literal": getattr(__import__("typing"), "Literal", None),
            "datetime": datetime,
            "date": date,
            "time": getattr(__import__("datetime"), "time"),
            "timedelta": timedelta,
            "Enum": Enum,
            "Decimal": Decimal,
            "UUID": UUID,
            # Add common imports that might be needed
            "typing": __import__("typing"),
            "json": json,
            "re": re,
        }

        try:
            # Execute the pydantic configuration code
            if self.debug_logging:
                self.logger.debug(f"Executing pydantic config for model '{model_name}'")
            exec(pydantic_config, namespace)

            # Extract the target model class
            if model_name in namespace:
                model_class = namespace[model_name]
                self.model_cache[cache_key] = model_class
                if self.debug_logging:
                    self.logger.debug(
                        f"Successfully created and cached model '{model_name}'"
                    )
                    self.logger.debug(
                        f"Model fields: {list(model_class.model_fields.keys())}"
                    )
                return model_class
            else:
                error_msg = (
                    f"Model '{model_name}' not found in the executed pydantic config"
                )
                if self.debug_logging:
                    self.logger.error(error_msg)
                    self.logger.debug(
                        f"Available classes in namespace: {[k for k in namespace.keys() if isinstance(namespace[k], type)]}"  # noqa: E501
                    )
                raise ValueError(error_msg)

        except Exception as e:
            error_msg = f"Error creating Pydantic model '{model_name}': {e}"
            if self.debug_logging:
                self.logger.error(error_msg)
                self.logger.debug(f"Pydantic config that failed:\n{pydantic_config}")
            print(error_msg)
            raise

    async def setup(self):
        """Load the dataset and prepare tasks."""
        if self.debug_logging:
            self.logger.info("Starting environment setup")

        try:
            # Load the dataset - you'll need to specify the correct dataset name/path
            dataset_name = getattr(
                self.config, "dataset_name", "justus27/pydantic-adherance-test"
            )
            dataset_split = getattr(self.config, "dataset_split", "train")

            if self.debug_logging:
                self.logger.info(
                    f"Loading dataset: {dataset_name}, split: {dataset_split}"
                )

            # Load your dataset
            dataset = load_dataset(dataset_name, split=dataset_split)

            if self.debug_logging:
                self.logger.info(
                    f"Dataset loaded successfully. Total items: {len(dataset)}"
                )

            # Convert to list for easier handling
            self.dataset_items = list(dataset)

            if self.debug_logging:
                self.logger.debug(
                    f"Sample dataset item keys: {list(self.dataset_items[0].keys()) if self.dataset_items else 'No items'}"  # noqa: E501
                )
                if self.dataset_items:
                    sample_item = self.dataset_items[0]
                    self.logger.debug(
                        f"Sample problem_id: {sample_item.get('problem_id', 'N/A')}"
                    )
                    self.logger.debug(
                        f"Sample task_type: {sample_item.get('task_type', 'N/A')}"
                    )
                    self.logger.debug(
                        f"Sample prompt length: {len(sample_item.get('prompt', ''))}"
                    )

            # Shuffle the dataset
            random.shuffle(self.dataset_items)
            if self.debug_logging:
                self.logger.debug("Dataset shuffled")

            # Split into train and test
            split_idx = int(len(self.dataset_items) * 0.90)  # 90% train, 10% test
            self.train_items = self.dataset_items[:split_idx]
            self.test_items = self.dataset_items[split_idx:]

            self.iter = 0

            if self.debug_logging:
                self.logger.info(
                    f"Dataset split complete: {len(self.train_items)} training items, {len(self.test_items)} test items"
                )
                self.logger.info("Environment setup complete")

            print(
                f"PydanticSchemaFollowingEnv setup complete. {len(self.train_items)} training items, {len(self.test_items)} test items."  # noqa: E501
            )

        except Exception as e:
            error_msg = f"Error during setup: {e}"
            if self.debug_logging:
                self.logger.error(error_msg)
            print(error_msg)
            # Fallback to empty lists if dataset loading fails
            self.train_items = []
            self.test_items = []
            self.iter = 0

    async def get_next_item(self) -> Tuple[Tuple[frozenset, ...], Dict[str, Any]]:
        """Get the next training item from the dataset."""
        if not self.train_items:
            error_msg = "No training items available. Setup might have failed or dataset is empty."
            if self.debug_logging:
                self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Get the next item cyclically
        dataset_item = self.train_items[self.iter % len(self.train_items)]

        if self.debug_logging:
            self.logger.debug(
                f"Getting item {self.iter % len(self.train_items)} (iteration {self.iter})"
            )
            self.logger.debug(
                f"Item problem_id: {dataset_item.get('problem_id', 'N/A')}"
            )

            # Parse verification info to get model name
            try:
                verification_info = dataset_item.get("verification_info", "{}")
                verification_data = json.loads(verification_info)
                model_name = verification_data.get("model_name", "Unknown")
                self.logger.debug(f"Target model: {model_name}")
            except (json.JSONDecodeError, KeyError):
                self.logger.debug("Could not parse model name from verification_info")

        self.iter += 1

        # Extract the prompt from the dataset item
        user_content = dataset_item["prompt"]

        if self.debug_logging:
            self.logger.debug(f"Prompt length: {len(user_content)} characters")

        # Create the message structure
        prompt_messages = [
            frozenset({"role": "system", "content": system_prompt}.items()),
            frozenset({"role": "user", "content": user_content}.items()),
        ]

        # Return the prompt and the full dataset item for scoring
        return tuple(prompt_messages), dataset_item

    def _extract_json_response(self, text: str) -> Optional[str]:
        """Extracts JSON content from <json_output> tags, with strict thinking tag validation."""
        if self.debug_logging:
            self.logger.debug(f"Extracting JSON from response (length: {len(text)})")

        # Ensure text is a string
        if not isinstance(text, str):
            if self.debug_logging:
                self.logger.warning(
                    f"Expected string but got {type(text)}, converting to string"
                )
            text = str(text)

        # First, validate thinking tags (similar to MCQA environment)
        think_tags = re.findall(r"<think>", text, re.IGNORECASE)
        think_close_tags = re.findall(r"</think>", text, re.IGNORECASE)

        # Check for proper thinking tag structure
        if len(think_tags) != 1 or len(think_close_tags) != 1:
            if self.debug_logging:
                self.logger.warning(
                    f"Invalid thinking tag structure: {len(think_tags)} open tags, {len(think_close_tags)} close tags"
                )
            return None

        # Split the text into thinking and response sections
        parts = re.split(r"</think>", text, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) != 2:
            if self.debug_logging:
                self.logger.warning(
                    "Could not split text into thinking and response sections"
                )
            return None

        thinking_section, response_section = parts

        # Validate thinking section contains opening tag
        if "<think>" not in thinking_section.lower():
            if self.debug_logging:
                self.logger.warning("Thinking section missing opening <think> tag")
            return None

        # Check if there are any thinking tags in the response section (after </think>)
        if "<think>" in response_section.lower():
            if self.debug_logging:
                self.logger.warning(
                    "Found <think> tags in response section after </think>"
                )
            return None

        # Now extract JSON from the response section only
        match = re.search(
            r"<json_output>\s*(.*?)\s*</json_output>",
            response_section,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            json_str = match.group(1).strip()
            if self.debug_logging:
                self.logger.debug(
                    f"Found JSON output tags in response section, extracted {len(json_str)} characters"
                )

            # Handle empty extraction
            if not json_str:
                if self.debug_logging:
                    self.logger.warning("JSON output tags found but content is empty")
                return None

            # Validate JSON
            try:
                json.loads(json_str)
                if self.debug_logging:
                    self.logger.debug("Extracted JSON is valid")
                return json_str
            except json.JSONDecodeError as e:
                if self.debug_logging:
                    self.logger.warning(f"Extracted text is not valid JSON: {e}")
                return None

        # Fallback: Look for JSON in response section only (no thinking tag validation)
        if self.debug_logging:
            self.logger.debug(
                "No <json_output> tags found in response section, trying fallback extraction"
            )

        # Look for JSON objects that start with { and end with } in response section only
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, response_section, re.DOTALL)

        for potential_json in matches:
            try:
                json.loads(potential_json.strip())
                if self.debug_logging:
                    self.logger.debug(
                        f"Fallback extraction successful from response section: {len(potential_json)} characters"
                    )
                return potential_json.strip()
            except json.JSONDecodeError:
                continue

        if self.debug_logging:
            self.logger.warning("No valid JSON found with any extraction method")
        return None

    async def score(
        self,
        rollout_group_data: List[Tuple[Tuple[Dict[str, str], ...], Dict[str, Any]]],
    ) -> Optional[ScoredDataGroup]:
        """Score the rollouts based on Pydantic validation."""
        if self.debug_logging:
            self.logger.debug(f"Scoring {len(rollout_group_data)} rollouts")

        scores_obj = ScoredDataGroup()
        scores_obj["tokens"] = list()
        scores_obj["masks"] = list()
        scores_obj["scores"] = list()
        scores_obj["messages"] = list()  # Add messages for data dumping

        # All items in rollout_group_data share the same dataset_item (item[1])
        if not rollout_group_data:
            if self.debug_logging:
                self.logger.warning("No rollout data to score")
            return None

        dataset_item = rollout_group_data[0][1]

        if self.debug_logging:
            self.logger.debug(
                f"Scoring for problem_id: {dataset_item.get('problem_id', 'N/A')}"
            )

        # Extract verification info (pydantic config) and model name
        verification_info = dataset_item["verification_info"]

        # Parse the verification info to get pydantic config and model name
        try:
            verification_data = json.loads(verification_info)
            pydantic_config = verification_data["pydantic_config"]
            model_name = verification_data["model_name"]

            if self.debug_logging:
                self.logger.debug(f"Target model for scoring: {model_name}")
                self.logger.debug(
                    f"Pydantic config length: {len(pydantic_config)} characters"
                )
        except (json.JSONDecodeError, KeyError) as e:
            error_msg = f"Error parsing verification_info: {e}"
            if self.debug_logging:
                self.logger.error(error_msg)
            print(error_msg)
            return None

        # Create the Pydantic model dynamically
        try:
            target_model_cls = self._create_pydantic_model_from_code(
                pydantic_config, model_name
            )
        except Exception as e:
            error_msg = f"Error creating Pydantic model: {e}"
            if self.debug_logging:
                self.logger.error(error_msg)
            print(error_msg)
            return None

        # Score each rollout
        valid_count = 0
        invalid_count = 0
        extraction_failures = 0

        # Shuffle to avoid bias in selection
        random.shuffle(rollout_group_data)

        for i, (item_messages, _) in enumerate(rollout_group_data):
            if self.debug_logging and i == 0:
                self.logger.debug(f"Scoring rollout {i+1}/{len(rollout_group_data)}")

            # Convert frozensets to dictionaries for easier access
            messages_as_dicts = [dict(fs_message) for fs_message in item_messages]
            model_response_text = messages_as_dicts[-1]["content"]  # LLM full response

            if self.debug_logging and i == 0:
                self.logger.debug(
                    f"Response length: {len(model_response_text)} characters"
                )

            json_str = self._extract_json_response(model_response_text)

            reward = 0.0  # Default score

            if json_str:
                # Validate JSON against the Pydantic model
                is_valid, error_msg = self._validate_json_against_model(
                    json_str, target_model_cls, dataset_item.get("problem_id", "N/A")
                )

                if is_valid:
                    reward = 1.0  # Valid schema - full score
                    valid_count += 1
                    if self.debug_logging and i < 3:  # Log first few successes
                        self.logger.debug(f"Rollout {i}: Validation successful")
                else:
                    reward = 0.0  # Validation failed
                    invalid_count += 1
                    if self.debug_logging and i < 3:  # Log first few validation errors
                        self.logger.debug(f"Rollout {i}: {error_msg}")
            else:
                reward = 0.0  # No JSON output found or extraction failed
                extraction_failures += 1
                if self.debug_logging and i < 3:  # Log first few failures
                    self.logger.debug(f"Rollout {i}: JSON extraction failed")

            # Tokenize for training - convert frozensets to dicts for tokenizer
            try:
                # Validate that messages_as_dicts is properly formatted
                if not isinstance(messages_as_dicts, list):
                    if self.debug_logging:
                        self.logger.error(
                            f"Expected list for tokenization, got {type(messages_as_dicts)}"
                        )
                    continue

                # Validate each message has required keys
                for msg_idx, msg in enumerate(messages_as_dicts):
                    if not isinstance(msg, dict):
                        if self.debug_logging:
                            self.logger.error(
                                f"Message {msg_idx} is not a dict: {type(msg)}"
                            )
                        continue
                    if "role" not in msg or "content" not in msg:
                        if self.debug_logging:
                            self.logger.error(
                                f"Message {msg_idx} missing required keys: {msg.keys()}"
                            )
                        continue
                    # Ensure content is a string
                    if not isinstance(msg["content"], str):
                        if self.debug_logging:
                            self.logger.warning(
                                f"Converting content to string for message {msg_idx}"
                            )
                        msg["content"] = str(msg["content"])

                out_dict = tokenize_for_trainer(
                    self.tokenizer, messages_as_dicts, include_messages=True
                )
                tokens = out_dict["tokens"]
                masks = out_dict["masks"]

            except Exception as e:
                if self.debug_logging:
                    self.logger.error(f"Tokenization failed for rollout {i}: {e}")
                    self.logger.debug(
                        f"Messages format: {[type(m) for m in messages_as_dicts]}"
                    )
                continue

            if len([1 for i in masks if i != -100]) < 10:  # Min context length
                if self.debug_logging:
                    self.logger.debug(
                        "Skipping rollout due to insufficient context length"
                    )
                continue

            scores_obj["tokens"].append(tokens)
            scores_obj["masks"].append(masks)
            scores_obj["scores"].append(reward)
            scores_obj["messages"].append(
                out_dict.get("messages", messages_as_dicts)
            )  # Store converted messages for dumping

            # Track perfect scores for wandb
            self.percent_correct_buffer.append(1.0 if reward == 1.0 else 0.0)

            if len(scores_obj["tokens"]) >= self.config.group_size:
                break

        if self.debug_logging:
            self.logger.info(
                f"Scoring complete: {valid_count} valid, {invalid_count} invalid, {extraction_failures} extraction failures"  # noqa: E501
            )
            if scores_obj["scores"]:
                avg_score = sum(scores_obj["scores"]) / len(scores_obj["scores"])
                self.logger.info(f"Average score for this batch: {avg_score:.3f}")

        if not scores_obj["tokens"]:  # No valid examples processed
            if self.debug_logging:
                self.logger.warning("No valid examples processed in this batch")
            return None

        if (
            all(scores_obj["scores"][0] == score for score in scores_obj["scores"])
            and scores_obj["scores"][0] != 1.0
        ):
            if self.debug_logging:
                self.logger.debug(
                    "All scores are identical and not perfect, returning None for learning signal"
                )
            return None

        # Apply length penalty if average response length is too high and all scores are 1.0
        if all(s == 1.0 for s in scores_obj["scores"]):
            avg_len = sum(len(t) for t in scores_obj["tokens"]) / len(
                scores_obj["tokens"]
            )
            if (
                avg_len > self.config.max_token_length * 0.75
            ):  # Penalize if too verbose even when correct
                scores_obj["scores"] = [s * 0.9 for s in scores_obj["scores"]]
                if self.debug_logging:
                    self.logger.debug(
                        f"Applied length penalty: avg_len={avg_len}, penalty_threshold={self.config.max_token_length * 0.75}"  # noqa: E501
                    )

        return scores_obj

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List]:
        """Collect trajectories for a given item."""
        prompt_messages_tuple, dataset_item = item

        if self.debug_logging:
            self.logger.debug(
                f"Collecting trajectories for problem_id: {dataset_item.get('problem_id', 'N/A')}"
            )

        # Convert frozensets to dicts for the API call
        messages_for_api = [dict(fs_message) for fs_message in prompt_messages_tuple]

        prompt_str = self.tokenizer.apply_chat_template(
            messages_for_api, add_generation_prompt=True, tokenize=False
        )

        if self.debug_logging:
            self.logger.debug(f"Generated prompt length: {len(prompt_str)} characters")
            self.logger.debug(
                f"Requesting {self.config.group_size} completions with max_tokens={self.config.max_token_length}, temperature=0.9"  # noqa: E501
            )

        completions = await self.server.completion(
            prompt=prompt_str,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
            temperature=0.9,
        )

        if self.debug_logging:
            self.logger.debug(
                f"Received {len(completions.choices)} completions from server"
            )

        to_score_list = []
        for i, choice in enumerate(completions.choices):
            if self.debug_logging and i < 3:  # Log first few completions
                self.logger.debug(
                    f"Completion {i} length: {len(choice.text)} characters"
                )

            # Create a full message list for this choice
            current_trajectory_messages = list(prompt_messages_tuple)
            current_trajectory_messages.append(
                frozenset({"role": "assistant", "content": choice.text}.items())
            )
            to_score_list.append((tuple(current_trajectory_messages), dataset_item))

        scored_data = await self.score(to_score_list)

        if self.debug_logging:
            if scored_data:
                self.logger.debug(
                    f"Scoring successful: {len(scored_data['scores'])} scored items"
                )
            else:
                self.logger.warning("Scoring returned None")

        # Log batch progress for data dumping
        current_batch_progress = self.processed_item_count % 100
        log_message_group_processed = (
            f"GROUP_PROC - Item Iter: {self.iter-1}, Scored Data Present: {bool(scored_data)}, "
            f"Dump Rollouts Cfg: {self.config.dump_rollouts}, "
            f"Total Items Processed (for save): {self.processed_item_count}, Batch Counter: {current_batch_progress}/99"
        )
        if self.debug_logging:
            self.logger.info(log_message_group_processed)

        # Data dumping logic
        if self.debug_logging:
            self.logger.info(
                f"COLLECT_TRAJ - dump_rollouts: {self.config.dump_rollouts}, "
                f"processed_item_count: {self.processed_item_count}, "
                f"current_buffer_size: {len(self.rollouts_to_save_buffer)}"
            )

        if scored_data and self.config.dump_rollouts:
            rollouts_for_current_item = []

            num_scored_rollouts = len(scored_data.get("scores", []))
            conversation_messages_batch = scored_data.get("messages", [])

            # Extract model info from dataset item
            try:
                verification_info = dataset_item.get("verification_info", "{}")
                verification_data = json.loads(verification_info)
                model_name = verification_data.get("model_name", "Unknown")
            except (json.JSONDecodeError, KeyError):
                model_name = "Unknown"

            for i in range(num_scored_rollouts):
                conversation_messages = (
                    conversation_messages_batch[i]
                    if i < len(conversation_messages_batch)
                    else []
                )
                score_for_rollout = scored_data["scores"][i]

                # Extract the generated JSON from the assistant's response
                if conversation_messages:
                    assistant_response = conversation_messages[-1].get("content", "")
                    generated_json = self._extract_json_response(assistant_response)
                else:
                    generated_json = None

                rollouts_for_current_item.append(
                    {
                        "conversation": conversation_messages,
                        "score": score_for_rollout,
                        "expected_json": dataset_item.get("verification_info", ""),
                        "generated_json": generated_json,
                        "model_name": model_name,
                        "problem_id": dataset_item.get("problem_id", "N/A"),
                        "task_type": dataset_item.get("task_type", "N/A"),
                    }
                )

            if rollouts_for_current_item:
                # Use problem_id as the source item ID
                source_item_id = dataset_item.get("problem_id", f"item_{self.iter-1}")

                item_data_to_save = {
                    "item_id": source_item_id,
                    "rollouts": rollouts_for_current_item,
                }
                self.rollouts_to_save_buffer.append(item_data_to_save)
                self.processed_item_count += 1

                if self.debug_logging:
                    self.logger.debug(
                        f"Added {len(rollouts_for_current_item)} rollouts for item {source_item_id}"
                    )

                # Save batch every 100 processed items
                if (
                    self.config.dump_rollouts
                    and self.processed_item_count > 0
                    and self.processed_item_count % 100 == 0
                ):
                    log_msg = (
                        f"Reached {self.processed_item_count} processed items. "
                        f"Triggering save for {len(self.rollouts_to_save_buffer)} item groups."
                    )
                    if self.debug_logging:
                        self.logger.info(log_msg)
                    await self._save_rollouts_to_jsonl()

        return scored_data, []

    async def _save_rollouts_to_jsonl(self):
        """Saves the buffered rollouts to a JSONL file in the datadumps directory."""
        if not self.rollouts_to_save_buffer:
            if self.debug_logging:
                self.logger.info("No rollouts in buffer to save.")
            return

        try:
            if not os.path.exists(self.datadumps_dir):
                os.makedirs(self.datadumps_dir)
                if self.debug_logging:
                    self.logger.info(f"Created directory: {self.datadumps_dir}")
        except OSError as e:
            error_msg = f"Error creating directory {self.datadumps_dir}: {e}"
            if self.debug_logging:
                self.logger.error(error_msg)
            print(error_msg)
            return

        file_path = os.path.join(
            self.datadumps_dir,
            f"pydantic_rollouts_{self.run_uuid}_{self.save_file_batch_num:04d}.jsonl",
        )

        try:
            with open(file_path, "w") as f:
                for rollout_dict in self.rollouts_to_save_buffer:
                    json.dump(rollout_dict, f)
                    f.write("\n")

            success_msg = f"Successfully saved {len(self.rollouts_to_save_buffer)} rollouts to {file_path}"
            if self.debug_logging:
                self.logger.info(success_msg)
            print(success_msg)

            self.rollouts_to_save_buffer.clear()
            self.save_file_batch_num += 1

        except IOError as e:
            error_msg = f"Error writing rollouts to {file_path}: {e}"
            if self.debug_logging:
                self.logger.error(error_msg)
            print(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error saving rollouts to {file_path}: {e}"
            if self.debug_logging:
                self.logger.error(error_msg)
            print(error_msg)

    async def rollout_and_score_eval(self, dataset_item: Dict[str, Any]) -> float:
        """Evaluate a single item from the test set."""
        if self.debug_logging:
            self.logger.debug(
                f"Evaluating item: {dataset_item.get('problem_id', 'N/A')}"
            )

        user_content = dataset_item["prompt"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        if self.debug_logging:
            self.logger.debug(f"Eval prompt length: {len(prompt)} characters")

        completion = await self.server.completion(
            prompt=prompt,
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.1,  # Lower temperature for eval
            split="eval",
        )

        model_response_text = completion.choices[0].text

        if self.debug_logging:
            self.logger.debug(
                f"Eval response length: {len(model_response_text)} characters"
            )

        json_str = self._extract_json_response(model_response_text)

        score = 0.0

        # Extract verification info and create model
        try:
            verification_info = dataset_item["verification_info"]
            verification_data = json.loads(verification_info)
            pydantic_config = verification_data["pydantic_config"]
            model_name = verification_data["model_name"]

            if self.debug_logging:
                self.logger.debug(f"Eval target model: {model_name}")

            target_model_cls = self._create_pydantic_model_from_code(
                pydantic_config, model_name
            )

            if json_str:
                # Validate JSON against the Pydantic model
                is_valid, error_msg = self._validate_json_against_model(
                    json_str, target_model_cls, dataset_item.get("problem_id", "N/A")
                )

                if is_valid:
                    score = 1.0  # Valid schema
                    if self.debug_logging:
                        self.logger.debug(
                            f"Eval validation successful for {dataset_item.get('problem_id', 'N/A')}"
                        )
                else:
                    score = 0.0  # Validation failed
                    if self.debug_logging:
                        self.logger.debug(
                            f"Eval validation failed for {dataset_item.get('problem_id', 'N/A')}: {error_msg}"
                        )
            else:
                score = 0.0  # No valid JSON extracted
                if self.debug_logging:
                    self.logger.debug(
                        f"Eval JSON extraction failed for {dataset_item.get('problem_id', 'N/A')}"
                    )

        except Exception as e:
            score = 0.0  # Any error in model creation or setup
            if self.debug_logging:
                self.logger.error(
                    f"Error in eval setup for {dataset_item.get('problem_id', 'N/A')}: {e}"
                )

        return score

    async def evaluate(self, *args, **kwargs):
        """Run evaluation on the test set."""
        if self.debug_logging:
            self.logger.info("Starting evaluation")

        if not self.test_items:
            warning_msg = "No test items available for evaluation."
            if self.debug_logging:
                self.logger.warning(warning_msg)
            print(warning_msg)
            self.eval_metrics.append(("eval/percent_correct", 0.0))
            return

        # Use a subset for faster evaluation
        items_to_eval = self.test_items[: min(len(self.test_items), 50)]

        if self.debug_logging:
            self.logger.info(f"Evaluating {len(items_to_eval)} items from test set")

        eval_results = await tqdm_asyncio.gather(
            *[self.rollout_and_score_eval(item) for item in items_to_eval]
        )

        # Calculate metrics
        perfect_scores = sum(1 for score in eval_results if score == 1.0)
        if eval_results:
            avg_score = sum(eval_results) / len(eval_results)
            percent_perfect = perfect_scores / len(eval_results)
        else:
            avg_score = 0.0
            percent_perfect = 0.0

        self.eval_metrics.append(("eval/avg_score", avg_score))
        self.eval_metrics.append(("eval/percent_perfect", percent_perfect))

        if self.debug_logging:
            self.logger.info(
                f"Evaluation complete: avg_score={avg_score:.3f}, percent_perfect={percent_perfect:.3f}"
            )
            self.logger.info(f"Perfect scores: {perfect_scores}/{len(eval_results)}")

        print(
            f"Evaluation complete. Avg Score: {avg_score:.3f}, Percent Perfect (1.0): {percent_perfect:.3f}"
        )

    async def add_rollouts_for_wandb(
        self,
        scored_data: Optional[ScoredDataGroup],
        item: Item = None,
    ):
        """Add rollouts to wandb logging."""
        if self.debug_logging:
            self.logger.debug("Adding rollouts for wandb logging")

        if scored_data is None or not scored_data["tokens"]:
            if self.debug_logging:
                self.logger.debug("No scored data to log to wandb")
            return

        dataset_item = item[1] if item else {}

        if self.debug_logging:
            self.logger.debug(
                f"Logging rollouts for problem_id: {dataset_item.get('problem_id', 'N/A')}"
            )

        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:  # Log all from the group
            num_keep = len(scored_data["tokens"])
        else:
            num_keep = min(num_keep, len(scored_data["tokens"]))

        if self.debug_logging:
            self.logger.debug(f"Keeping {num_keep} rollouts for logging")

        rollout_batch = []
        for i in range(num_keep):
            # Decode tokens to text for logging
            full_convo_text = self.tokenizer.decode(
                scored_data["tokens"][i], skip_special_tokens=True
            )

            extracted_json = self._extract_json_response(full_convo_text)

            # Extract model info from dataset item
            try:
                verification_info = dataset_item.get("verification_info", "{}")
                verification_data = json.loads(verification_info)
                model_name = verification_data.get("model_name", "N/A")
                expected_json = verification_data.get("pydantic_config", "N/A")
            except (json.JSONDecodeError, KeyError):
                model_name = "N/A"
                expected_json = "N/A"
                if self.debug_logging:
                    self.logger.debug("Could not extract model name for wandb logging")

            rollout_batch.append(
                (
                    full_convo_text,  # Full conversation
                    scored_data["scores"][i],
                    model_name,
                    dataset_item.get("problem_id", "N/A"),
                    dataset_item.get("task_type", "N/A"),
                    (
                        extracted_json
                        if extracted_json
                        else "Extraction failed or no JSON"
                    ),
                    (
                        expected_json[:200] + "..."
                        if len(expected_json) > 200
                        else expected_json
                    ),  # Truncate for display
                )
            )

        if rollout_batch:
            self.rollouts_for_wandb.append(rollout_batch)
            if self.debug_logging:
                self.logger.debug(f"Added {len(rollout_batch)} rollouts to wandb queue")

        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            removed = self.rollouts_for_wandb.pop(0)
            if self.debug_logging:
                self.logger.debug(
                    f"Removed oldest rollout batch ({len(removed)} items) from wandb queue"
                )

    async def create_rollout_table(self, wandb_metrics: Dict) -> Dict:
        """Create wandb table for rollout visualization."""
        if self.debug_logging:
            self.logger.debug(
                f"Creating wandb rollout table with {len(self.rollouts_for_wandb)} batches"
            )

        if self.rollouts_for_wandb:
            table = wandb.Table(
                columns=[
                    "full_conversation",
                    "score",
                    "model_name",
                    "problem_id",
                    "task_type",
                    "extracted_json",
                    "expected_schema",
                ]
            )
            total_entries = 0
            for group in self.rollouts_for_wandb:
                for entry in group:
                    table.add_data(*entry)
                    total_entries += 1
            wandb_metrics["train/rollouts"] = table

            if self.debug_logging:
                self.logger.debug(
                    f"Created wandb table with {total_entries} total entries"
                )

        self.rollouts_for_wandb = []  # Clear after logging
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb."""
        if self.debug_logging:
            self.logger.debug("Logging metrics to wandb")

        if wandb_metrics is None:
            wandb_metrics = {}

        if self.percent_correct_buffer:
            percent_perfect = sum(self.percent_correct_buffer) / len(
                self.percent_correct_buffer
            )
            wandb_metrics["train/percent_perfect"] = percent_perfect
            if self.debug_logging:
                self.logger.debug(
                    f"Train percent perfect: {percent_perfect:.3f} (from {len(self.percent_correct_buffer)} samples)"
                )
        else:
            wandb_metrics["train/percent_perfect"] = 0.0
            if self.debug_logging:
                self.logger.debug("No percent_correct_buffer data available")

        self.percent_correct_buffer = list()

        # Add eval metrics
        if self.eval_metrics:
            if self.debug_logging:
                self.logger.debug(
                    f"Adding {len(self.eval_metrics)} eval metrics to wandb"
                )
            for key, value in self.eval_metrics:
                wandb_metrics[key] = value
                if self.debug_logging:
                    self.logger.debug(f"Eval metric: {key} = {value}")

        self.eval_metrics = list()

        # Create rollout table (if any rollouts were collected)
        wandb_metrics = await self.create_rollout_table(wandb_metrics)

        if self.debug_logging:
            self.logger.debug(f"Final wandb metrics: {list(wandb_metrics.keys())}")

        await super().wandb_log(wandb_metrics)

    async def close(self):
        """Clean up and save any remaining rollouts before exiting."""
        if self.debug_logging:
            self.logger.info(
                "Closing PydanticSchemaFollowingEnv. Attempting to save any remaining rollouts..."
            )

        if self.config.dump_rollouts and self.rollouts_to_save_buffer:
            if self.debug_logging:
                self.logger.info(
                    f"Found {len(self.rollouts_to_save_buffer)} rollouts in buffer. Saving now."
                )
            await self._save_rollouts_to_jsonl()
        else:
            if self.debug_logging:
                self.logger.info("No rollouts in buffer to save upon closing.")

        # Call the superclass's close method more robustly
        base_close_method = getattr(super(), "close", None)
        if base_close_method and callable(base_close_method):
            try:
                if asyncio.iscoroutinefunction(base_close_method):
                    await base_close_method()
                else:
                    base_close_method()
            except Exception as e_super_close:
                if self.debug_logging:
                    self.logger.error(f"Error during super().close(): {e_super_close}")
        elif self.debug_logging:
            self.logger.debug(
                "No callable super().close() method found or it does not exist."
            )

        if self.debug_logging:
            self.logger.info("PydanticSchemaFollowingEnv closed.")

    def _validate_json_against_model(
        self, json_str: str, model_cls: Type[BaseModel], problem_id: str = "N/A"
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate JSON string against a Pydantic model and return detailed error info.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            model_cls.model_validate_json(json_str)
            return True, None
        except ValidationError as ve:
            try:
                # Attempt to get structured error data, which is generally more robust
                error_details = json.dumps(ve.errors(), indent=2)
                # Truncate potentially long error details
                max_detail_len = 250
                if len(error_details) > max_detail_len:
                    error_details_str = f"{error_details[:max_detail_len]}..."
                else:
                    error_details_str = error_details
                error_msg = f"Pydantic validation failed for {problem_id} with {len(ve.errors())} error(s):\n{error_details_str}"  # noqa: E501
            except Exception as format_exc:
                # Fallback if formatting ve.errors() fails for some reason
                error_msg = f"Pydantic validation failed for {problem_id}. Error: {str(ve)[:250]}. (Additionally, formatting error details failed: {str(format_exc)[:50]})"  # noqa: E501

            if self.debug_logging:
                self.logger.debug(error_msg)
            return False, error_msg
        except json.JSONDecodeError as je:
            error_msg = f"JSON decode failed for {problem_id}: {str(je)[:100]}"
            if self.debug_logging:
                self.logger.debug(error_msg)
            return False, error_msg
        except TypeError as te:
            # Specifically check for the "model" keyword argument issue in ValidationError.__new__
            if (
                "ValidationError.__new__() got an unexpected keyword argument 'model'"
                in str(te)
            ):
                error_msg = f"Pydantic internal TypeError for {problem_id}: {str(te)[:200]}. This may indicate a Pydantic V1/V2 compatibility issue within the dynamic schema definition from the dataset."  # noqa: E501
                if self.debug_logging:
                    self.logger.error(
                        error_msg
                    )  # Log as error due to its specific nature
                    # Optionally log the problematic JSON for further inspection if not too large
                    # self.logger.debug(f"Problematic JSON string for {problem_id} (first 500 chars): {json_str[:500]}")
            else:
                # Handle other TypeErrors normally
                error_msg = f"Unexpected TypeError during validation for {problem_id}: {type(te).__name__}: {str(te)[:100]}"  # noqa: E501
                if self.debug_logging:
                    self.logger.debug(error_msg)
            return False, error_msg
        except Exception as e:
            # Catch any other unexpected exceptions during validation
            error_msg = f"Generic unexpected validation error for {problem_id}: {type(e).__name__}: {str(e)[:100]}"
            if self.debug_logging:
                self.logger.debug(error_msg)
            return False, error_msg


if __name__ == "__main__":
    PydanticSchemaFollowingEnv.cli()
