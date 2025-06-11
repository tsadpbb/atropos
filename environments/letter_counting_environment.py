import json
import logging
import os
import random
import re
import uuid
from typing import Dict, List, Optional, Tuple, Union

import wandb
from pydantic import Field
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

# Import NLTK words corpus for large-scale word list
try:
    import nltk
    from nltk.corpus import words
    from nltk.tokenize import sent_tokenize
    # Download required NLTK data
    try:
        words.words()
    except LookupError:
        nltk.download('words')
    try:
        sent_tokenize("Test sentence.")
    except LookupError:
        nltk.download('punkt')
except ImportError:
    print("Warning: NLTK not available. Please install with: pip install nltk")
    words = None
    sent_tokenize = None

# Import datasets for OpenWebText
try:
    from datasets import load_dataset
except ImportError:
    print("Warning: datasets library not available. Please install with: pip install datasets")
    load_dataset = None

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)


class LetterCountingConfig(BaseEnvConfig):
    """Configuration class for Letter Counting Environment with custom parameters."""
    
    # Word dataset configuration
    min_word_length: int = Field(3, description="Minimum word length to include")
    max_word_length: int = Field(15, description="Maximum word length to include")
    train_test_split: float = Field(0.95, description="Ratio for train/test split")
    
    # Letter selection configuration
    use_all_letters: bool = Field(True, description="Whether to use all 26 letters or custom set")
    custom_letters: str = Field("aeiou", description="Custom letter set if use_all_letters=False")
    
    # Generation configuration
    generation_temperature: float = Field(1.0, description="Temperature for training generation")
    eval_temperature: float = Field(0.2, description="Temperature for evaluation generation")
    max_generation_tokens: int = Field(1024 * 15, description="Maximum tokens for model generation")
    
    # Evaluation configuration
    eval_sample_size: int = Field(1000, description="Number of test words to evaluate on")
    
    # Reproducibility configuration
    random_seed: Optional[int] = Field(42, description="Seed for reproducibility, None for random")
    
    # Random string generation configuration
    random_string_percentage: float = Field(0.0, description="Percentage of dataset to be random strings (0.0-1.0)")
    random_string_min_length: int = Field(3, description="Minimum length for random strings")
    random_string_max_length: int = Field(15, description="Maximum length for random strings")
    
    # Word capitalization configuration
    uppercase_word_percentage: float = Field(0.0, description="Percentage of real words to make uppercase (0.0-1.0)")
    capitalized_word_percentage: float = Field(0.0, description="Percentage of real words to capitalize first letter (0.0-1.0)")
    
    # Text/passage configuration
    use_text_passages: bool = Field(False, description="Include text passages from OpenWebText in addition to words")
    text_passage_percentage: float = Field(0.5, description="Percentage of dataset to be text passages when use_text_passages=True (0.0-1.0)")
    min_text_length: int = Field(50, description="Minimum character length for text passages")
    max_text_length: int = Field(500, description="Maximum character length for text passages")
    include_punctuation_in_count: bool = Field(True, description="Include punctuation in letter counting")
    include_spaces_in_count: bool = Field(False, description="Include spaces in letter counting")
    
    # Multi-letter counting configuration
    max_letters_to_count: int = Field(1, description="Maximum number of different letters to count simultaneously (1 for single letter)")
    multi_letter_probability: float = Field(0.0, description="Probability of asking for multiple letters (0.0-1.0)")
    
    # Difficulty and training thresholds
    max_group_average_for_training: float = Field(1.0, description="Maximum group average to use for training (skip groups that are too easy)")
    calculate_text_difficulty: bool = Field(True, description="Calculate and log text difficulty scores")
    
    # Logging and data dumping configuration
    debug_logging: bool = Field(False, description="Enable debug-level logging for more verbose output")
    suppress_base_env_logs: bool = Field(True, description="Suppress verbose base environment logs")
    dump_rollouts: bool = Field(False, description="Whether to dump successful rollouts to JSONL files")
    dump_failed_rollouts: bool = Field(False, description="Whether to dump failed rollouts for debugging")
    rollout_save_score_threshold: float = Field(0.7, description="Minimum score threshold for saving rollouts to data dumps. Only groups with at least one rollout above this threshold will be saved.")

class LetterCountingEnv(BaseEnv):
    """
    Letter Counting Environment for training models to count letters in words and sentences.
    
    This environment presents the model with questions like "How many 'a's are in the word 'banana'?"
    or "Count the occurrences of the letters 'e', 'o', and 't' in the following text: 'The quick brown fox jumps over the lazy dog'"
    and expects responses in the format <answer>3</answer> for single letters or <answer>{"e": 4, "o": 4, "t": 2}</answer> 
    for multiple letters. The model should use <think></think> tags for reasoning before providing the final answer.
    
    Features:
    - **Word Mode**: Uses NLTK's words corpus (236k+ English words)
    - **Mixed Mode**: Combines words and text passages from OpenWebText-10k dataset
    - **Text Passage Mode**: Uses OpenWebText-10k dataset with character-based text extraction
    - Optional random string generation (80% alphabetical) mixed with real words
    - Configurable word/string/passage length ranges and letter sets
    - Optional word capitalization (uppercase, title case)
    - **Multi-letter counting**: Configurable simultaneous counting of multiple letters with JSON responses
    - Text difficulty scoring and training thresholds
    - Configurable punctuation and space handling for letter counting
    - Comprehensive logging and data dumping capabilities
    - Detailed metrics tracking (letter distribution, text lengths, error rates, difficulty scores)
    - Support for saving successful and failed rollouts for analysis
    
    Data Dumping:
    - Set dump_rollouts=True to save successful rollouts (score=1) to JSONL files
    - Set dump_failed_rollouts=True to save failed rollouts for debugging
    - Files saved to data_dumps/ directory with unique UUIDs
    - Rollouts include full conversations, scores, metadata, and difficulty scores
    
    Mixed Mode Configuration:
    - Set use_text_passages=True to enable mixed mode with both words and text passages
    - Configure text_passage_percentage to control the ratio (e.g., 0.3 = 30% passages, 70% words)
    - Configure min/max text passage character lengths (more reliable than word counts)
    - Set max_group_average_for_training to skip groups that are too easy
    - Enable difficulty calculation for text complexity scoring
    
    Logging:
    - Set debug_logging=True for verbose per-item scoring details
    - Comprehensive WandB metrics including letter distribution entropy and difficulty scores
    - Progress tracking for data dumps and evaluation
    """ # noqa
    
    name = "letter_counting"
    env_config_cls = LetterCountingConfig
    
    def __init__(
        self,
        config: LetterCountingConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        """
        Initialize the Letter Counting environment.

        Args:
            config: Configuration for the base environment
            server_configs: List of server configurations for OpenAI API
            slurm: Whether to use Slurm for distributed training
            testing: Whether in testing mode
        """ # noqa: E501
        super().__init__(config, server_configs, slurm, testing)
        
        # Initialize data dumping infrastructure first (needed for logging)
        self.run_uuid = str(uuid.uuid4())
        self.rollouts_to_save_buffer: List[Dict[str, Union[str, List[Dict[str, Union[List[Dict[str, str]], float]]]]]] = []
        self.processed_item_count = 0
        self.datadumps_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data_dumps"
        )
        self.save_file_batch_num = 0

        # For saving failed rollouts (all 0 scores) for debugging
        self.failed_rollouts_to_save_buffer: List[Dict[str, Union[str, List[Dict[str, Union[List[Dict[str, str]], float]]]]]] = []
        self.failed_processed_item_count = 0
        self.failed_save_file_batch_num = 0
        
        # Additional metrics tracking
        self.letter_distribution_stats: Dict[str, int] = {}
        self.word_length_stats: Dict[int, int] = {}
        self.answer_format_errors = 0
        self.think_format_errors = 0
        
        # Initialize the logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            # Add a basic stream handler if no handlers are configured
            _handler = logging.StreamHandler()
            _formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            _handler.setFormatter(_formatter)
            self.logger.addHandler(_handler)
            # Set logging level based on config
            log_level = logging.DEBUG if self.config.debug_logging else logging.INFO
            self.logger.setLevel(log_level)
        # Ensure the logger itself is enabled
        self.logger.disabled = False

        # Suppress base environment logs if requested
        if self.config.suppress_base_env_logs:
            # Set the base environment logger to WARNING level to suppress INFO logs
            base_logger = logging.getLogger("atroposlib.envs.base")
            base_logger.setLevel(logging.WARNING)
            
        # Log initialization completion
        self.logger.info(f"LetterCountingEnv initialized with run UUID: {self.run_uuid}")
        self.logger.info(f"Debug logging: {'enabled' if self.config.debug_logging else 'disabled'}")
        self.logger.info(f"Data dumping: rollouts={'enabled' if self.config.dump_rollouts else 'disabled'}, failed={'enabled' if self.config.dump_failed_rollouts else 'disabled'}")
        
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.rollouts_for_wandb: List[List[Tuple[str, float, str, str, str]]] = []

    @classmethod
    def config_init(self) -> Tuple[LetterCountingConfig, List[APIServerConfig]]:
        env_config = LetterCountingConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=32,
            use_wandb=True,
            max_num_workers=128,
            rollout_server_url="http://localhost:8000",
            total_steps=250,
            batch_size=1024,
            steps_per_eval=20,
            max_token_length=1024 * 15,
            inference_weight=1.0,
            wandb_name="letter_counting_deep_thinking",
            data_path_to_save_groups=None,
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            # Letter counting specific configs
            min_word_length=3,
            max_word_length=30,
            train_test_split=0.95,
            eval_sample_size=1000,
            generation_temperature=1.0,
            eval_temperature=0.5,
            random_seed=42,
            use_all_letters=True,
            custom_letters="aeiou",
            max_generation_tokens=1024 * 15,
            # Random string generation
            random_string_percentage=0.0,
            random_string_min_length=3,
            random_string_max_length=15,
            # Word capitalization
            uppercase_word_percentage=0.01,
            capitalized_word_percentage=0.005,
            # Text passage configuration
            use_text_passages=True,
            text_passage_percentage=0.3,
            min_text_length=3,
            max_text_length=2000,
            include_punctuation_in_count=True,
            include_spaces_in_count=True,
            max_group_average_for_training=0.7,
            calculate_text_difficulty=True,
            # Multi-letter counting
            max_letters_to_count=4,
            multi_letter_probability=0.2,
            debug_logging=True,
            dump_rollouts=True,
            rollout_save_score_threshold=0.7,
            dump_failed_rollouts=False,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            )
        ]

        return env_config, server_configs

    async def setup(self):
        """
        Set up the environment by loading and preparing the word/text dataset.
        """
        if self.config.use_text_passages:
            await self._setup_mixed_dataset()
        else:
            await self._setup_word_dataset()
        
        # Initialize iteration counter
        self.iter = 0

    async def _setup_mixed_dataset(self):
        """
        Set up the environment using both words and text passages from OpenWebText dataset.
        """
        if load_dataset is None:
            raise ImportError("datasets library is required for text passage mode. Please install with: pip install datasets")
        if words is None:
            raise ImportError("NLTK is required for this environment. Please install with: pip install nltk")
        
        # Set random seed for reproducibility if configured
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
        
        # Validate configuration
        await self._validate_config()
        
        self.logger.info("Setting up mixed dataset with words and text passages...")
        
        # First, set up words (same as word-only mode)
        all_words = words.words()
        filtered_words = [
            word.lower() for word in all_words 
            if word.isalpha() and self.config.min_word_length <= len(word) <= self.config.max_word_length
        ]
        
        # Apply capitalization to real words if configured
        filtered_words = self._apply_word_capitalization(filtered_words)
        
        # Generate random strings if configured
        if self.config.random_string_percentage > 0.0:
            total_filtered_words = len(filtered_words)
            if self.config.random_string_percentage >= 1.0:
                num_random_strings = total_filtered_words if total_filtered_words > 0 else 1000
                filtered_words = []
            else:
                num_random_strings = int(
                    (self.config.random_string_percentage * total_filtered_words) / 
                    (1.0 - self.config.random_string_percentage)
                )
            
            random_strings = self._generate_random_strings(num_random_strings)
            all_word_strings = filtered_words + random_strings
            self.logger.info(f"Generated {num_random_strings} random strings ({self.config.random_string_percentage:.1%} of word dataset)")
        else:
            all_word_strings = filtered_words
            random_strings = []
        
        self.logger.info(f"Prepared {len(all_word_strings)} word/string items ({len(filtered_words)} real words + {len(random_strings)} random strings)")
        
        # Now load and process text passages
        self.logger.info("Loading OpenWebText-10k dataset for text passages...")
        dataset = load_dataset("stas/openwebtext-10k", split="train")
        self.logger.info(f"Loaded {len(dataset)} text samples from OpenWebText")
        
        # Extract and filter text passages
        all_passages = []
        processed_texts = 0
        
        for item in dataset:
            text = item["text"]
            
            # Skip texts that are too long initially
            if len(text) > self.config.max_text_length * 3:  # Allow some overhead for chunking
                continue
                
            # Extract passages from this text
            passages = self._extract_text_passages(text)
            all_passages.extend(passages)
            processed_texts += 1
            
            # Log progress periodically
            if processed_texts % 1000 == 0:
                self.logger.info(f"Processed {processed_texts} texts, extracted {len(all_passages)} passages so far...")
        
        self.logger.info(f"Extracted {len(all_passages)} text passages from {processed_texts} texts")
        
        # Now mix words and passages according to text_passage_percentage
        if self.config.text_passage_percentage >= 1.0:
            # Special case: 100% text passages, no words
            all_mixed_items = all_passages
            final_word_count = 0
            final_passage_count = len(all_passages)
        elif self.config.text_passage_percentage <= 0.0:
            # Special case: 0% text passages, only words
            all_mixed_items = all_word_strings
            final_word_count = len(all_word_strings)
            final_passage_count = 0
        else:
            # Calculate how many passages to include
            # If we want X% passages, then passages / (words + passages) = X
            # Solving: passages_to_use = X * words / (1 - X)
            total_words = len(all_word_strings)
            num_passages_to_use = int(
                (self.config.text_passage_percentage * total_words) / 
                (1.0 - self.config.text_passage_percentage)
            )
            
            # Sample passages if we have more than needed
            if num_passages_to_use > len(all_passages):
                passages_to_use = all_passages
                self.logger.warning(f"Requested {num_passages_to_use} passages but only have {len(all_passages)}. Using all available passages.")
            else:
                passages_to_use = random.sample(all_passages, num_passages_to_use)
            
            # Combine words and passages
            all_mixed_items = all_word_strings + passages_to_use
            final_word_count = len(all_word_strings)
            final_passage_count = len(passages_to_use)
        
        # Shuffle the mixed dataset
        random.shuffle(all_mixed_items)
        
        # Create train/test split
        split_point = int(self.config.train_test_split * len(all_mixed_items))
        self.train_words = all_mixed_items[:split_point]  # Reusing train_words for mixed items
        self.test_words = all_mixed_items[split_point:]
        
        # Calculate actual percentages
        actual_passage_percentage = final_passage_count / len(all_mixed_items) if len(all_mixed_items) > 0 else 0.0
        
        # Log dataset statistics
        self.logger.info(f"Mixed dataset created:")
        self.logger.info(f"  Total items: {len(all_mixed_items)} ({final_word_count} words/strings + {final_passage_count} passages)")
        self.logger.info(f"  Actual passage percentage: {actual_passage_percentage:.1%}")
        self.logger.info(f"  Training items: {len(self.train_words)}")
        self.logger.info(f"  Test items: {len(self.test_words)}")
        
        # Show examples of both types
        word_examples = [item for item in self.train_words[:10] if len(item) <= 50][:3]
        passage_examples = [item[:100] + "..." for item in self.train_words[:20] if len(item) > 50][:3]
        
        if word_examples:
            self.logger.info(f"  Example words: {word_examples}")
        if passage_examples:
            self.logger.info(f"  Example passages: {passage_examples}")
        
        # Log configuration details
        self.logger.info(f"Word length range: {self.config.min_word_length}-{self.config.max_word_length}")
        self.logger.info(f"Passage length range: {self.config.min_text_length}-{self.config.max_text_length} characters")
        self.logger.info(f"Include punctuation: {self.config.include_punctuation_in_count}")
        self.logger.info(f"Include spaces: {self.config.include_spaces_in_count}")
        self.logger.info(f"Training threshold: {self.config.max_group_average_for_training}")
        
        self.logger.info("Mixed dataset setup complete")

    async def _setup_text_passage_dataset(self):
        """
        Set up the environment using text passages from OpenWebText dataset.
        """
        if load_dataset is None:
            raise ImportError("datasets library is required for text passage mode. Please install with: pip install datasets")
        
        # Set random seed for reproducibility if configured
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
        
        # Validate configuration
        await self._validate_config()
        
        self.logger.info("Loading OpenWebText-10k dataset...")
        
        # Load the dataset
        dataset = load_dataset("stas/openwebtext-10k", split="train")
        
        self.logger.info(f"Loaded {len(dataset)} text samples from OpenWebText")
        
        # Extract and filter text passages
        all_passages = []
        processed_texts = 0
        
        for item in dataset:
            text = item["text"]
            
            # Skip texts that are too long initially
            if len(text) > self.config.max_text_length * 3:  # Allow some overhead for chunking
                continue
                
            # Extract passages from this text
            passages = self._extract_text_passages(text)
            all_passages.extend(passages)
            processed_texts += 1
            
            # Log progress periodically
            if processed_texts % 1000 == 0:
                self.logger.info(f"Processed {processed_texts} texts, extracted {len(all_passages)} passages so far...")
        
        self.logger.info(f"Extracted {len(all_passages)} text passages from {processed_texts} texts")
        
        # Shuffle passages for randomness
        random.shuffle(all_passages)
        
        # Create train/test split
        split_point = int(self.config.train_test_split * len(all_passages))
        self.train_words = all_passages[:split_point]  # Reusing train_words for passages
        self.test_words = all_passages[split_point:]
        
        # Log dataset statistics
        self.logger.info(f"Training passages: {len(self.train_words)}")
        self.logger.info(f"Test passages: {len(self.test_words)}")
        self.logger.info(f"Example passages: {[p[:100] + '...' for p in self.train_words[:3]]}")
        
        # Log configuration details
        self.logger.info(f"Passage length range: {self.config.min_text_length}-{self.config.max_text_length} characters")
        self.logger.info(f"Include punctuation: {self.config.include_punctuation_in_count}")
        self.logger.info(f"Include spaces: {self.config.include_spaces_in_count}")
        self.logger.info(f"Training threshold: {self.config.max_group_average_for_training}")
        
        self.logger.info("Text passage dataset setup complete")

    async def _setup_word_dataset(self):
        """
        Set up the environment using single words (original functionality).
        """
        # Load the NLTK words corpus (contains 236,736 English words)
        if words is None:
            raise ImportError("NLTK is required for this environment. Please install with: pip install nltk")
        
        # Set random seed for reproducibility if configured
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
        
        # Validate configuration
        await self._validate_config()
        
        # Get all English words from NLTK
        all_words = words.words()
        
        # Filter words to ensure they contain only alphabetic characters
        # and are within the configured length range for reasonable difficulty
        filtered_words = [
            word.lower() for word in all_words 
            if word.isalpha() and self.config.min_word_length <= len(word) <= self.config.max_word_length
        ]
        
        # Apply capitalization to real words if configured
        filtered_words = self._apply_word_capitalization(filtered_words)
        
        # Generate random strings if configured
        if self.config.random_string_percentage > 0.0:
            # Calculate how many random strings to generate
            total_filtered_words = len(filtered_words)
            # If we want X% random strings, then random_strings / (words + random_strings) = X
            # Solving: random_strings = X * words / (1 - X)
            if self.config.random_string_percentage >= 1.0:
                # Special case: 100% random strings, no real words
                num_random_strings = total_filtered_words if total_filtered_words > 0 else 1000
                filtered_words = []
            else:
                num_random_strings = int(
                    (self.config.random_string_percentage * total_filtered_words) / 
                    (1.0 - self.config.random_string_percentage)
                )
            
            random_strings = self._generate_random_strings(num_random_strings)
            
            # Combine real words and random strings
            all_strings = filtered_words + random_strings
            self.logger.info(f"Generated {num_random_strings} random strings ({self.config.random_string_percentage:.1%} of dataset)")
        else:
            # No random strings, use only real words
            all_strings = filtered_words
            random_strings = []
        
        # Shuffle all strings for randomness
        random.shuffle(all_strings)
        
        # Create train/test split using configured ratio
        split_point = int(self.config.train_test_split * len(all_strings))
        self.train_words = all_strings[:split_point]
        self.test_words = all_strings[split_point:]
        
        # Log dataset statistics
        if self.config.random_string_percentage > 0.0:
            self.logger.info(f"Total dataset: {len(all_strings)} strings ({len(filtered_words)} real words + {len(random_strings)} random strings)")
        else:
            self.logger.info(f"Loaded {len(all_strings)} words total")
        self.logger.info(f"Training strings: {len(self.train_words)}")
        self.logger.info(f"Test strings: {len(self.test_words)}")
        self.logger.info(f"Example strings: {self.train_words[:10]}")
        
        # Log configuration details
        self.logger.info(f"Word length range: {self.config.min_word_length}-{self.config.max_word_length}")
        if self.config.random_string_percentage > 0.0:
            self.logger.info(f"Random string length range: {self.config.random_string_min_length}-{self.config.random_string_max_length}")
        self.logger.info(f"Train/test split: {self.config.train_test_split:.2%}")
        self.logger.info(f"Letter set: {'all 26 letters' if self.config.use_all_letters else f'custom ({self.config.custom_letters})'}")
        self.logger.info(f"Random strings: {self.config.random_string_percentage:.1%} of dataset")
        
        # Log capitalization settings
        if self.config.uppercase_word_percentage > 0.0 or self.config.capitalized_word_percentage > 0.0:
            self.logger.info(f"Word capitalization: {self.config.uppercase_word_percentage:.1%} uppercase, {self.config.capitalized_word_percentage:.1%} title case")
        
        self.logger.info(f"Random seed: {self.config.random_seed}")
        
        # Log data dumping configuration
        if self.config.dump_rollouts:
            self.logger.info("Data dumping enabled - saving groups with at least one successful rollout")
            self.logger.info(f"Maximum group average threshold: {self.config.max_group_average_threshold}")
            self.logger.info(f"Data dumps directory: {self.datadumps_dir}")
        if self.config.dump_failed_rollouts:
            self.logger.info("Failed rollout dumping enabled for debugging")
            
        self.logger.info("Letter counting environment setup complete")

    async def _save_rollouts_to_jsonl(self):
        """Saves the buffered rollouts to a JSONL file in the datadumps directory."""
        if not self.rollouts_to_save_buffer:
            self.logger.info("No rollouts in buffer to save.")
            return

        try:
            if not os.path.exists(self.datadumps_dir):
                os.makedirs(self.datadumps_dir)
                self.logger.info(f"Created directory: {self.datadumps_dir}")
        except OSError as e:
            self.logger.error(f"Error creating directory {self.datadumps_dir}: {e}")
            return

        file_path = os.path.join(
            self.datadumps_dir,
            f"letter_counting_environment_rollouts_{self.run_uuid}_{self.save_file_batch_num:04d}.jsonl",
        )

        try:
            with open(file_path, "w") as f:
                for rollout_dict in self.rollouts_to_save_buffer:
                    json.dump(rollout_dict, f)
                    f.write("\n")
            self.logger.info(
                f"Successfully saved {len(self.rollouts_to_save_buffer)} rollouts to {file_path}"
            )
            self.rollouts_to_save_buffer.clear()
            self.save_file_batch_num += 1
        except IOError as e:
            self.logger.error(f"Error writing rollouts to {file_path}: {e}")
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while saving rollouts to {file_path}: {e}"
            )

    async def _save_failed_rollouts_to_jsonl(self):
        """Saves the buffered failed rollouts (all 0 scores) to a JSONL file for debugging."""
        if not self.failed_rollouts_to_save_buffer:
            self.logger.info("No failed rollouts in buffer to save.")
            return

        try:
            if not os.path.exists(self.datadumps_dir):
                os.makedirs(self.datadumps_dir)
                self.logger.info(f"Created directory: {self.datadumps_dir}")
        except OSError as e:
            self.logger.error(f"Error creating directory {self.datadumps_dir}: {e}")
            return

        file_path = os.path.join(
            self.datadumps_dir,
            f"letter_counting_environment_FAILED_rollouts_{self.run_uuid}_{self.failed_save_file_batch_num:04d}.jsonl",
        )

        try:
            with open(file_path, "w") as f:
                for rollout_dict in self.failed_rollouts_to_save_buffer:
                    json.dump(rollout_dict, f)
                    f.write("\n")
            self.logger.info(
                f"Successfully saved {len(self.failed_rollouts_to_save_buffer)} FAILED rollouts to {file_path}"
            )
            self.failed_rollouts_to_save_buffer.clear()
            self.failed_save_file_batch_num += 1
        except IOError as e:
            self.logger.error(f"Error writing failed rollouts to {file_path}: {e}")
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while saving failed rollouts to {file_path}: {e}"
            )

    async def _save_failed_group_for_debugging(self, rollout_group_data, scores_container):
        """Helper method to save failed groups (all 0 scores) for debugging analysis."""
        failed_rollouts_with_scores_to_save = []

        # Build the failed rollouts data structure
        for i, rollout_data in enumerate(rollout_group_data):
            if i < len(scores_container["scores"]):
                trajectory_messages = rollout_data[0]
                correct_counts = rollout_data[1]
                text = rollout_data[2]
                target_letters = rollout_data[3]
                stop_reason = rollout_data[4]
                difficulty_score = rollout_data[5] if len(rollout_data) > 5 else 0.0
                
                score_for_rollout = scores_container["scores"][i]
                failed_rollouts_with_scores_to_save.append(
                    {
                        "conversation": trajectory_messages,  # Full conversation history
                        "score": score_for_rollout,
                        "expected_counts": correct_counts,
                        "text": text,
                        "target_letters": target_letters,
                        "stop_reason": stop_reason,
                        "difficulty_score": difficulty_score,
                    }
                )

        if failed_rollouts_with_scores_to_save:
            # Extract item info for logging
            first_rollout = rollout_group_data[0]
            correct_counts = first_rollout[1]
            text = first_rollout[2]
            target_letters = first_rollout[3]
            text_preview = text[:30].replace(' ', '_') if len(text) > 30 else text.replace(' ', '_')
            letters_str = "_".join(target_letters)
            counts_str = "_".join(str(correct_counts.get(letter, 0)) for letter in target_letters)
            item_id = f"{text_preview}_{letters_str}_{counts_str}"

            failed_item_data_to_save = {
                "item_id": item_id,
                "rollouts": failed_rollouts_with_scores_to_save,
            }
            self.failed_rollouts_to_save_buffer.append(failed_item_data_to_save)
            self.failed_processed_item_count += 1

            # Calculate progress toward next failed save
            failed_batch_progress = self.failed_processed_item_count % 50
            if failed_batch_progress == 0:
                failed_batch_progress = 50

            # Log progress every 10 failed items or when we hit the save threshold
            if failed_batch_progress % 10 == 0 or failed_batch_progress == 50:
                self.logger.info(
                    f"Failed rollouts progress: {failed_batch_progress}/50 items buffered "
                    f"(Total failed processed: {self.failed_processed_item_count}, Failed buffer size: {len(self.failed_rollouts_to_save_buffer)})"
                )

            # Check if it's time to save a batch of failed rollouts (every 50)
            if (
                self.config.dump_failed_rollouts
                and self.failed_processed_item_count % 50 == 0
                and self.failed_processed_item_count > 0
            ):
                failed_log_msg = (
                    f"Reached {self.failed_processed_item_count} failed items. "
                    f"Triggering save for {len(self.failed_rollouts_to_save_buffer)} failed items "
                    f"(each with multiple failed rollouts)."
                )
                self.logger.info(failed_log_msg)
                await self._save_failed_rollouts_to_jsonl()

    def _get_letter_set(self):
        """
        Get the set of letters to choose from based on configuration.
        
        Returns:
            String containing letters to choose from (lowercase only for consistency)
        """
        if not self.config.use_all_letters:
            return self.config.custom_letters.lower()
        else:
            return 'abcdefghijklmnopqrstuvwxyz'

    def _generate_random_string(self, length: int) -> str:
        """
        Generate a random string of specified length with at least 80% alphabetical characters.
        
        Args:
            length: Length of the string to generate
            
        Returns:
            Random string with mix of uppercase, lowercase, and some non-alphabetical chars
        """
        # Ensure at least 80% alphabetical characters
        min_alpha_chars = max(1, int(length * 0.8))
        
        # Character sets
        letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        non_alpha = '0123456789!@#$%^&*()-_=+[]{}|;:,.<>?'
        
        result = []
        
        # First, add the required alphabetical characters
        for _ in range(min_alpha_chars):
            result.append(random.choice(letters))
        
        # Fill the rest with either alphabetical or non-alphabetical characters
        remaining_length = length - min_alpha_chars
        for _ in range(remaining_length):
            # 90% chance of alphabetical even for remaining chars (to exceed 80% minimum)
            if random.random() < 0.9:
                result.append(random.choice(letters))
            else:
                result.append(random.choice(non_alpha))
        
        # Shuffle to avoid alphabetical chars being clustered at the beginning
        random.shuffle(result)
        return ''.join(result)

    def _generate_random_strings(self, count: int) -> List[str]:
        """
        Generate a list of random strings with lengths within the configured random string range.
        
        Args:
            count: Number of random strings to generate
            
        Returns:
            List of random strings
        """
        random_strings = []
        for _ in range(count):
            # Generate random length within configured random string range
            length = random.randint(self.config.random_string_min_length, self.config.random_string_max_length)
            random_string = self._generate_random_string(length)
            random_strings.append(random_string)
        return random_strings

    def _apply_word_capitalization(self, words: List[str]) -> List[str]:
        """
        Apply capitalization transformations to real words based on configuration.
        
        Args:
            words: List of lowercase words
            
        Returns:
            List of words with applied capitalization
        """
        if self.config.uppercase_word_percentage == 0.0 and self.config.capitalized_word_percentage == 0.0:
            return words
        
        result = []
        for word in words:
            rand_val = random.random()
            
            if rand_val < self.config.uppercase_word_percentage:
                # Make uppercase
                result.append(word.upper())
            elif rand_val < self.config.uppercase_word_percentage + self.config.capitalized_word_percentage:
                # Capitalize first letter
                result.append(word.capitalize())
            else:
                # Keep lowercase
                result.append(word)
        
        return result

    def _calculate_text_difficulty(self, text: str, target_letters: List[str]) -> float:
        """
        Calculate a difficulty score for counting letters in the given text.
        
        Args:
            text: The text to analyze
            target_letters: List of letters being counted
            
        Returns:
            Difficulty score (higher = more difficult)
        """
        if not self.config.calculate_text_difficulty:
            return 0.0
            
        # Basic difficulty factors
        text_length = len(text)
        word_count = len(text.split())
        
        # Calculate total letter occurrences and max occurrences for any single letter
        total_letter_count = 0
        max_single_letter_count = 0
        for letter in target_letters:
            count = text.lower().count(letter.lower())
            total_letter_count += count
            max_single_letter_count = max(max_single_letter_count, count)
        
        # Multi-letter complexity
        num_letters = len(target_letters)
        multi_letter_factor = min(num_letters / 2.0, 2.0)  # Cap at 2x for multiple letters
        
        # Complexity factors
        unique_chars = len(set(text.lower()))
        punctuation_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        
        # Calculate difficulty score
        # Longer texts are harder
        length_factor = min(text_length / 100.0, 5.0)  # Cap at 5x
        
        # More words = harder
        word_factor = min(word_count / 10.0, 3.0)  # Cap at 3x
        
        # More letter occurrences = harder to count accurately
        letter_factor = min(total_letter_count / 10.0, 2.5)  # Cap at 2.5x
        
        # More unique characters = more distracting
        char_diversity_factor = min(unique_chars / 20.0, 1.5)  # Cap at 1.5x
        
        # Punctuation adds complexity
        punct_factor = min(punctuation_count / 10.0, 1.5)  # Cap at 1.5x
        
        # Combine factors (weighted average)
        difficulty = (
            length_factor * 0.25 +
            word_factor * 0.2 +
            letter_factor * 0.2 +
            multi_letter_factor * 0.15 +
            char_diversity_factor * 0.1 +
            punct_factor * 0.1
        )
        
        return difficulty

    def _extract_text_passages(self, text: str) -> List[str]:
        """
        Extract text passages from raw text based on character length.
        
        Args:
            text: Raw text to extract passages from
            
        Returns:
            List of filtered text passages
        """
        # Clean the text - remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # If text is shorter than min length, skip it
        if len(text) < self.config.min_text_length:
            return []
        
        # If text is within range, use it as-is
        if len(text) <= self.config.max_text_length:
            return [text]
        
        # For longer texts, create overlapping chunks
        passages = []
        chunk_size = self.config.max_text_length
        overlap = min(50, chunk_size // 4)  # 25% overlap, max 50 chars
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            
            # If this would be the last chunk and it's too small, extend the previous chunk
            if end >= len(text):
                if len(text) - start >= self.config.min_text_length:
                    passages.append(text[start:])
                break
            
            # Try to break at a natural boundary (space, punctuation)
            chunk = text[start:end]
            
            # Look for a good break point in the last 20% of the chunk
            break_start = int(len(chunk) * 0.8)
            break_candidates = []
            
            # Find sentence endings first
            for i in range(len(chunk) - 1, break_start - 1, -1):
                if chunk[i] in '.!?':
                    break_candidates.append(i + 1)
                    break
            
            # If no sentence ending, look for other punctuation
            if not break_candidates:
                for i in range(len(chunk) - 1, break_start - 1, -1):
                    if chunk[i] in ',;:':
                        break_candidates.append(i + 1)
                        break
            
            # If no punctuation, look for spaces
            if not break_candidates:
                for i in range(len(chunk) - 1, break_start - 1, -1):
                    if chunk[i] == ' ':
                        break_candidates.append(i)
                        break
            
            # Use the break point if found, otherwise use the full chunk
            if break_candidates:
                actual_end = start + break_candidates[0]
                passage = text[start:actual_end].strip()
            else:
                passage = chunk.strip()
                actual_end = end
            
            # Only add if it meets minimum length
            if len(passage) >= self.config.min_text_length:
                passages.append(passage)
            
            # Move start position with overlap
            start = actual_end - overlap
            
            # Avoid infinite loops
            if start >= end - overlap:
                start = end
        
        return passages

    def _prepare_text_for_counting(self, text: str) -> str:
        """
        Prepare text for letter counting based on configuration.
        
        Args:
            text: Original text
            
        Returns:
            Processed text for counting
        """
        if not self.config.include_punctuation_in_count:
            # Remove punctuation but keep spaces and alphanumeric
            text = ''.join(c for c in text if c.isalnum() or c.isspace())
        
        if not self.config.include_spaces_in_count:
            # Remove spaces
            text = text.replace(' ', '')
        
        return text

    async def _validate_config(self):
        """Validate configuration parameters."""
        if not (0.0 <= self.config.random_string_percentage <= 1.0):
            raise ValueError(f"random_string_percentage must be between 0.0 and 1.0, got {self.config.random_string_percentage}") # noqa
        if not (0.0 <= self.config.uppercase_word_percentage <= 1.0):
            raise ValueError(f"uppercase_word_percentage must be between 0.0 and 1.0, got {self.config.uppercase_word_percentage}") # noqa
        if not (0.0 <= self.config.capitalized_word_percentage <= 1.0):
            raise ValueError(f"capitalized_word_percentage must be between 0.0 and 1.0, got {self.config.capitalized_word_percentage}") # noqa
        if self.config.uppercase_word_percentage + self.config.capitalized_word_percentage > 1.0:
            raise ValueError(f"Sum of uppercase_word_percentage ({self.config.uppercase_word_percentage}) and capitalized_word_percentage ({self.config.capitalized_word_percentage}) cannot exceed 1.0") # noqa
        if self.config.random_string_min_length < 1:
            raise ValueError(f"random_string_min_length must be at least 1, got {self.config.random_string_min_length}") # noqa
        if self.config.random_string_max_length < self.config.random_string_min_length:
            raise ValueError(f"random_string_max_length ({self.config.random_string_max_length}) must be >= random_string_min_length ({self.config.random_string_min_length})") # noqa
        if self.config.use_text_passages:
            if not (0.0 <= self.config.text_passage_percentage <= 1.0):
                raise ValueError(f"text_passage_percentage must be between 0.0 and 1.0, got {self.config.text_passage_percentage}")
            if self.config.min_text_length < 1:
                raise ValueError(f"min_text_length must be at least 1, got {self.config.min_text_length}")
            if self.config.max_text_length < self.config.min_text_length:
                raise ValueError(f"max_text_length ({self.config.max_text_length}) must be >= min_text_length ({self.config.min_text_length})")
            if self.config.max_text_length < 10:
                raise ValueError(f"max_text_length must be at least 10 characters, got {self.config.max_text_length}")
        if self.config.max_letters_to_count < 1:
            raise ValueError(f"max_letters_to_count must be at least 1, got {self.config.max_letters_to_count}")
        if not (0.0 <= self.config.multi_letter_probability <= 1.0):
            raise ValueError(f"multi_letter_probability must be between 0.0 and 1.0, got {self.config.multi_letter_probability}")
        if self.config.max_letters_to_count > 26:
            raise ValueError(f"max_letters_to_count cannot exceed 26 (total letters), got {self.config.max_letters_to_count}")

    def save_checkpoint(self, step, data=None):
        """Save checkpoint including current iteration number, statistics, and data dumping state."""
        if data is None:
            data = {}
        data["iter"] = self.iter
        data["processed_item_count"] = self.processed_item_count
        data["save_file_batch_num"] = self.save_file_batch_num
        data["failed_processed_item_count"] = self.failed_processed_item_count
        data["failed_save_file_batch_num"] = self.failed_save_file_batch_num
        data["letter_distribution_stats"] = self.letter_distribution_stats
        data["word_length_stats"] = self.word_length_stats
        data["answer_format_errors"] = self.answer_format_errors
        data["think_format_errors"] = self.think_format_errors
        super().save_checkpoint(step, data)

    def load_checkpoint(self):
        """Load checkpoint including iteration number, statistics, and data dumping state."""
        # Call the base class method first to load the data
        super().load_checkpoint()

        # The base class loads data into attributes, so we can access them directly
        # if they were saved in save_checkpoint
        if hasattr(self, "iter"):
            # Restore statistics if available
            if hasattr(self, "letter_distribution_stats") and self.letter_distribution_stats:
                total_letters = sum(self.letter_distribution_stats.values())
                self.logger.info(f"Restored letter distribution stats with {total_letters} total letters")
            if hasattr(self, "word_length_stats") and self.word_length_stats:
                total_words = sum(self.word_length_stats.values())
                self.logger.info(f"Restored word length stats with {total_words} total words")
            if hasattr(self, "answer_format_errors"):
                self.logger.info(f"Restored error counts: {self.answer_format_errors} answer format errors, {self.think_format_errors} think format errors") # noqa

    async def close(self):
        """Clean up and save any remaining rollouts before exiting."""
        self.logger.info(
            "Closing LetterCountingEnv. Attempting to save any remaining rollouts..."
        )
        if self.config.dump_rollouts and self.rollouts_to_save_buffer:
            self.logger.info(
                f"Found {len(self.rollouts_to_save_buffer)} rollouts in buffer. Saving now."
            )
            await self._save_rollouts_to_jsonl()
        else:
            self.logger.info("No rollouts in buffer to save upon closing.")

        # Also save any remaining failed rollouts
        if self.config.dump_failed_rollouts and self.failed_rollouts_to_save_buffer:
            self.logger.info(
                f"Found {len(self.failed_rollouts_to_save_buffer)} failed rollouts in buffer. Saving now."
            )
            await self._save_failed_rollouts_to_jsonl()
        else:
            self.logger.info("No failed rollouts in buffer to save upon closing.")

        # Call the superclass's close method if it exists
        if hasattr(super(), "close"):
            await super().close()
        self.logger.info("LetterCountingEnv closed.")

    async def get_next_item(self):
        """
        Get the next training item from the dataset.

        Returns:
            A tuple containing prompt and expected answer
        """
        # Get the next text from training set (could be a word, sentence, or random string)
        text = self.train_words[self.iter % len(self.train_words)]
        
        # Determine how many letters to count
        letters = self._get_letter_set()
        
        # Decide whether to use multiple letters
        use_multiple = (
            self.config.max_letters_to_count > 1 and 
            random.random() < self.config.multi_letter_probability
        )
        
        if use_multiple:
            # Choose 2 to max_letters_to_count different letters
            num_letters = random.randint(2, self.config.max_letters_to_count)
            target_letters = random.sample(list(letters), num_letters)
        else:
            # Single letter counting
            target_letters = [random.choice(letters)]
        
        # Prepare text for counting (handle punctuation/spaces based on config)
        text_for_counting = self._prepare_text_for_counting(text)
        
        # Count occurrences for each target letter (case-insensitive)
        correct_counts = {}
        for letter in target_letters:
            correct_counts[letter] = text_for_counting.lower().count(letter.lower())
        
        # Determine if this is a text passage or word/string based on length and content
        is_text_passage = len(text) > 50 or ' ' in text or any(c in text for c in '.,!?;:')
        
        # Log item selection details for every item
        text_type = "passage" if is_text_passage else "word/string"
        text_preview = text[:50] + "..." if len(text) > 50 else text
        letters_str = ", ".join(target_letters)
        counts_str = ", ".join(f"{letter}:{correct_counts[letter]}" for letter in target_letters)
        difficulty_info = ""
        if is_text_passage and self.config.calculate_text_difficulty:
            difficulty_score = self._calculate_text_difficulty(text, target_letters)
            difficulty_info = f" (difficulty: {difficulty_score:.2f})"
        
        self.logger.info(
            f"Selected {text_type}: '{text_preview}' | Letters: [{letters_str}] | Counts: [{counts_str}]{difficulty_info} (iteration {self.iter})"
        )
        
        self.iter += 1
        
        # Create the question based on whether this item is a text passage or word/string and single/multiple letters
        if len(target_letters) == 1:
            # Single letter question
            target_letter = target_letters[0]
            if is_text_passage:
                question_text = f"How many {target_letter}s are in the following text: \"{text}\"?"
            else:
                question_text = f"How many {target_letter}s are in the string {text}?"
            
            # Add instruction for single letter answer format
            question_with_instruction = f'{question_text}\n\nProvide your answer in the format: <answer>{{number}}</answer>'
        else:
            # Multiple letters question
            letters_str = ", ".join(f"'{letter}'" for letter in target_letters[:-1]) + f", and '{target_letters[-1]}'"
            if is_text_passage:
                question_text = f"Count the occurrences of the letters {letters_str} in the following text: \"{text}\""
            else:
                question_text = f"Count the occurrences of the letters {letters_str} in the string {text}"
            
            # Add instruction for multiple letter JSON answer format
            example_json = "{" + ", ".join(f'"{letter}": 0' for letter in target_letters) + "}"
            question_with_instruction = f'{question_text}\n\nProvide your answer as JSON in the format: <answer>{example_json}</answer>'
        
        # Create prompt tuple using frozensets as required
        prompt = []
        
        # Add system prompt
        prompt.append(frozenset({"role": "system", "content": system_prompt}.items()))
        
        # Add user message with the question
        prompt.append(
            frozenset(
                {"role": "user", "content": question_with_instruction}.items()
            )
        )
        
        # Calculate difficulty score if enabled (for text passages)
        difficulty_score = self._calculate_text_difficulty(text, target_letters) if is_text_passage else 0.0
        
        # Return the prompt, correct counts, text, target letters, and difficulty score
        return (tuple(prompt), correct_counts, text, target_letters, difficulty_score)

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, List]:
        """
        Generate and collect model responses for scoring.

        Args:
            item: Input item containing prompt and expected answer

        Returns:
            Tuple of lists containing scored data groups and backlog
        """
        # Extract messages from the item
        messages = []
        for role_dict in item[0]:
            messages.append(dict(role_dict))

        # Apply chat template to convert messages to a single string
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Get completions from the model
        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=self.config.max_generation_tokens,
            temperature=self.config.generation_temperature,
        )

        to_score = list()

        for i, completion_choice in enumerate(completions.choices):
            # Create a copy of the prompt messages
            trajectory_messages = []
            for role_dict in item[0]:
                trajectory_messages.append(dict(role_dict))

            # Add the model's response
            trajectory_messages.append(
                {"role": "assistant", "content": completion_choice.text}
            )

            # Add to scoring queue with expected answer and metadata
            to_score.append(
                (
                    tuple(trajectory_messages),
                    item[1],  # correct_counts (dict)
                    item[2],  # text (word or sentence)
                    item[3],  # target_letters (list)
                    completion_choice.finish_reason,  # stop reason
                    item[4] if len(item) > 4 else 0.0,  # difficulty_score
                )
            )

        # Call score to get the scored data
        scored_data = await self.score(to_score)

        # If rollouts were generated and scored, and data dumping is enabled, prepare them for saving
        if scored_data and self.config.dump_rollouts:
            # Only save groups that have at least one rollout with score > threshold
            group_scores = scored_data.get("scores", [])
            threshold = self.config.rollout_save_score_threshold
            if any(score > threshold for score in group_scores):
                self.logger.debug(
                    f"Saving group with scores: {[f'{s:.3f}' for s in group_scores]} (has high-quality rollout, threshold: {threshold})"
                )
                rollouts_with_scores_to_save = []

                num_scored_rollouts = len(group_scores)
                for i in range(num_scored_rollouts):
                    conversation_messages = to_score[i][0]
                    score_for_rollout = group_scores[i]
                    correct_counts = to_score[i][1]
                    text = to_score[i][2]
                    target_letters = to_score[i][3]
                    stop_reason = to_score[i][4]
                    difficulty_score = to_score[i][5] if len(to_score[i]) > 5 else 0.0
                    
                    rollouts_with_scores_to_save.append(
                        {
                            "conversation": conversation_messages,
                            "score": score_for_rollout,
                            "expected_counts": correct_counts,
                            "text": text,
                            "target_letters": target_letters,
                            "stop_reason": stop_reason,
                            "difficulty_score": difficulty_score,
                        }
                    )

                if rollouts_with_scores_to_save:
                    # Extract item info for logging
                    _, correct_counts, text, target_letters = item[1], item[2], item[3]
                    text_preview = text[:30].replace(' ', '_') if len(text) > 30 else text.replace(' ', '_')
                    letters_str = "_".join(target_letters)
                    counts_str = "_".join(str(correct_counts.get(letter, 0)) for letter in target_letters)
                    item_id = f"{text_preview}_{letters_str}_{counts_str}"

                    item_data_to_save = {
                        "item_id": item_id,
                        "rollouts": rollouts_with_scores_to_save,
                    }
                    self.rollouts_to_save_buffer.append(item_data_to_save)
                    self.processed_item_count += 1

                # Calculate progress toward next save
                current_batch_progress = self.processed_item_count % 100
                if current_batch_progress == 0:
                    current_batch_progress = 100

                # Log progress every 10 items or when we hit the save threshold
                if current_batch_progress % 10 == 0 or current_batch_progress == 100:
                    self.logger.info(
                        f"Data dump progress: {current_batch_progress}/100 items buffered "
                        f"(Total processed: {self.processed_item_count}, Buffer size: {len(self.rollouts_to_save_buffer)})"
                    )

                # Check if it's time to save a batch of rollouts
                if (
                    self.config.dump_rollouts
                    and self.processed_item_count % 100 == 0
                    and self.processed_item_count > 0
                ):
                    log_msg = (
                        f"Reached {self.processed_item_count} processed items. "
                        f"Triggering save for {len(self.rollouts_to_save_buffer)} items "
                        f"(each with multiple scored rollouts)."
                    )
                    self.logger.info(log_msg)
                    await self._save_rollouts_to_jsonl()
            else:
                max_score = max(group_scores) if group_scores else 0.0
                self.logger.debug(
                    f"Skipping group save - no high-quality rollouts (max score: {max_score:.3f}, threshold: {threshold})"
                )

        to_backlog = []
        return scored_data, to_backlog

    def _extract_answer(self, text, expected_format="single"):
        """
        Extract the answer from model response (single number or JSON).
        Only allows one valid answer format - multiple answer formats result in a score of 0.

        Args:
            text: Text containing the model's response
            expected_format: "single" for number, "multi" for JSON

        Returns:
            Extracted answer (int for single, dict for multi) or None if invalid
        """
        # Check for multiple <think> tags - score as 0 if found
        think_tags = re.findall(r"<think>", text, re.IGNORECASE)
        if len(think_tags) > 1:
            return None

        # Check if the think tag is properly opened - we need exactly one opening tag
        if len(think_tags) != 1:
            return None

        # Check for </think> closing tags
        think_close_tags = re.findall(r"</think>", text, re.IGNORECASE)
        if len(think_close_tags) != 1:
            return None  # Must have exactly one closing tag

        # Split the text into thinking and answer sections
        parts = re.split(r"</think>", text, flags=re.IGNORECASE, maxsplit=1)

        # If there's no </think> tag or multiple sections, return None
        if len(parts) != 2:
            return None

        thinking_section, answer_section = parts

        # Validate thinking section
        # Make sure thinking section actually contains the opening <think> tag
        if "<think>" not in thinking_section.lower():
            return None  # Malformed thinking section

        # Check if there are any <think> tags in the answer section (after the first </think>)
        if "<think>" in answer_section.lower():
            return None

        # Look for answer tags in the answer section
        if expected_format == "single":
            # Single number format
            answer_pattern = r"<answer>\s*(\d+)\s*</answer>"
            answer_matches = re.findall(answer_pattern, answer_section, re.IGNORECASE)
            
            # If no answers found or multiple answers found, return None
            if len(answer_matches) != 1:
                return None
            
            # Return the single found answer as an integer
            try:
                return int(answer_matches[0])
            except ValueError:
                return None
        else:
            # Multi-letter JSON format
            answer_pattern = r"<answer>\s*(\{[^}]+\})\s*</answer>"
            answer_matches = re.findall(answer_pattern, answer_section, re.IGNORECASE)
            
            # If no answers found or multiple answers found, return None
            if len(answer_matches) != 1:
                return None
            
            # Try to parse the JSON
            try:
                import json
                answer_dict = json.loads(answer_matches[0])
                
                # Validate that all values are integers
                if not isinstance(answer_dict, dict):
                    return None
                
                for key, value in answer_dict.items():
                    if not isinstance(key, str) or not isinstance(value, int):
                        return None
                
                return answer_dict
            except (json.JSONDecodeError, ValueError):
                return None

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        """
        Score the generated model responses against expected letter counts.

        Args:
            rollout_group_data: List of generated responses with expected answers

        Returns:
            ScoredDataGroup with tokenized inputs and scores, or None if no valid scores
        """
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        if not rollout_group_data:
            return None

        # Get the expected answer from first item
        expected_counts = rollout_group_data[0][1]  # correct counts (dict for multi, int for single - legacy)
        text = rollout_group_data[0][2]  # text (word or sentence)
        target_letters = rollout_group_data[0][3]  # target letters (list)
        difficulty_score = rollout_group_data[0][5] if len(rollout_group_data[0]) > 5 else 0.0  # difficulty score

        # Handle legacy format (single letter as string, single count as int)
        if isinstance(target_letters, str):
            target_letters = [target_letters]
            expected_counts = {target_letters[0]: expected_counts}
        elif isinstance(expected_counts, int):
            # Legacy format with single letter
            expected_counts = {target_letters[0]: expected_counts}

        # Track statistics for all target letters
        for target_letter in target_letters:
            if target_letter not in self.letter_distribution_stats:
                self.letter_distribution_stats[target_letter] = 0
            self.letter_distribution_stats[target_letter] += 1
        
        text_len = len(text)
        if text_len not in self.word_length_stats:
            self.word_length_stats[text_len] = 0
        self.word_length_stats[text_len] += 1

        # Shuffle to avoid bias in selection
        random.shuffle(rollout_group_data)

        format_errors_in_group = 0
        think_errors_in_group = 0

        for item in rollout_group_data:
            # Extract the model's response
            model_response = item[0][-1]["content"]
            stop_reason = item[4]  # Get the stop reason

            # If the response was cut off due to length, give it a score of 0
            if stop_reason == "length":
                reward = 0
                if self.config.debug_logging:
                    letters_str = ", ".join(target_letters)
                    self.logger.debug(f"Text '{text[:50]}...' letters '{letters_str}': Length cutoff, score=0")
            else:
                # Determine expected format and extract the answer
                expected_format = "single" if len(target_letters) == 1 else "multi"
                model_answer = self._extract_answer(model_response, expected_format)

                # Track metrics based on result
                if model_answer is None:
                    reward = 0  # Invalid format gets 0 reward
                    format_errors_in_group += 1
                    # Check if it's a think format error
                    if "<think>" not in model_response.lower() or "</think>" not in model_response.lower():
                        think_errors_in_group += 1
                    if self.config.debug_logging:
                        letters_str = ", ".join(target_letters)
                        self.logger.debug(f"Text '{text[:50]}...' letters '{letters_str}': Format error, score=0")
                else:
                    # Check if answer matches expected counts
                    if expected_format == "single":
                        # Single letter: compare integer
                        expected_single_count = expected_counts[target_letters[0]]
                        if model_answer == expected_single_count:
                            reward = 1
                            if self.config.debug_logging:
                                self.logger.debug(f"Text '{text[:50]}...' letter '{target_letters[0]}': Correct answer {model_answer}, score=1")
                        else:
                            reward = 0
                            if self.config.debug_logging:
                                self.logger.debug(f"Text '{text[:50]}...' letter '{target_letters[0]}': Wrong answer {model_answer} (expected {expected_single_count}), score=0")
                    else:
                        # Multiple letters: compare dictionaries
                        # Check if all expected letters are present and counts match
                        if (set(model_answer.keys()) == set(target_letters) and 
                            all(model_answer.get(letter, -1) == expected_counts[letter] for letter in target_letters)):
                            reward = 1
                            if self.config.debug_logging:
                                self.logger.debug(f"Text '{text[:50]}...' letters {target_letters}: Correct answer {model_answer}, score=1")
                        else:
                            reward = 0
                            if self.config.debug_logging:
                                self.logger.debug(f"Text '{text[:50]}...' letters {target_letters}: Wrong answer {model_answer} (expected {expected_counts}), score=0")

            # Tokenize the conversation for learning
            out_dict = tokenize_for_trainer(self.tokenizer, item[0])
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Remove examples with insufficient context
            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(1.0 if reward else 0.0)

            # Break once we have enough examples
            if len(scores["tokens"]) >= self.config.group_size:
                break

        if not scores["tokens"]:
            letters_str = ", ".join(target_letters)
            self.logger.warning(
                f"No valid items were scored for text '{text[:50]}...' letters '{letters_str}' - all items had insufficient context"
            )
            return None

        # Update global error counters
        self.answer_format_errors += format_errors_in_group
        self.think_format_errors += think_errors_in_group

        # Record success rate metrics for wandb logging
        for score in scores["scores"]:
            self.percent_correct_buffer.append(score)

        # Calculate and log average score for the current group
        current_scores = scores.get("scores", [])
        if current_scores:
            average_score = sum(current_scores) / len(current_scores)
            
            # Create log message with appropriate text preview
            text_preview = text[:50] + "..." if len(text) > 50 else text
            letters_str = ", ".join(target_letters)
            expected_str = str(expected_counts) if len(target_letters) > 1 else str(expected_counts[target_letters[0]])
            log_message_main = (
                f"Text: '{text_preview}' | Letters: '{letters_str}' | Expected: {expected_str} | "
                f"Group average score: {average_score:.4f}"
            )
            
            # Add difficulty score if available
            if difficulty_score > 0:
                log_message_main += f" | Difficulty: {difficulty_score:.3f}"
            
            if all(s == 1.0 for s in current_scores):
                self.logger.info(f"{log_message_main} (All correct in this group!)")
            elif all(s == 0.0 for s in current_scores):
                self.logger.info(f"{log_message_main} (All failed - format/answer errors!)")
            else:
                self.logger.info(log_message_main)
            
            # Check training threshold - if group is too easy, skip it for training
            if average_score > self.config.max_group_average_for_training:
                self.logger.debug(
                    f"Skipping group for training - too easy (avg: {average_score:.3f} > threshold: {self.config.max_group_average_for_training})"
                )
                
                # Before returning None, check if this is a completely failed group (all 0.0 scores) for debugging
                if self.config.dump_failed_rollouts and all(score == 0.0 for score in scores["scores"]):
                    self.logger.debug("Saving failed group (all 0 scores) for debugging analysis")
                    await self._save_failed_group_for_debugging(rollout_group_data, scores)
                
                return None

        # Check if all scores are the same (no learning signal)
        if all(scores["scores"][0] == score for score in scores["scores"]):
            self.logger.debug(
                f"All scores in group are identical ({scores['scores'][0]:.4f}) - no learning signal, skipping group"
            )

            # Before returning None, check if this is a completely failed group (all 0.0 scores) for debugging
            if self.config.dump_failed_rollouts and all(score == 0.0 for score in scores["scores"]):
                self.logger.debug("Saving failed group (all 0 scores) for debugging analysis")
                await self._save_failed_group_for_debugging(rollout_group_data, scores)

            return None

        return scores

    async def rollout_and_score_eval(self, test_text):
        """
        Generate and score model responses for a single test text.

        Args:
            test_text: Test text from dataset (could be word, sentence, or random string)

        Returns:
            Score (1 for correct, 0 for incorrect)
        """
        # Determine how many letters to count (same logic as get_next_item)
        letters = self._get_letter_set()
        
        # Decide whether to use multiple letters
        use_multiple = (
            self.config.max_letters_to_count > 1 and 
            random.random() < self.config.multi_letter_probability
        )
        
        if use_multiple:
            # Choose 2 to max_letters_to_count different letters
            num_letters = random.randint(2, self.config.max_letters_to_count)
            target_letters = random.sample(list(letters), num_letters)
        else:
            # Single letter counting
            target_letters = [random.choice(letters)]
        
        # Prepare text for counting (handle punctuation/spaces based on config)
        text_for_counting = self._prepare_text_for_counting(test_text)
        
        # Count occurrences for each target letter (case-insensitive)
        expected_counts = {}
        for letter in target_letters:
            expected_counts[letter] = text_for_counting.lower().count(letter.lower())
        
        # Determine if this is a text passage or word/string based on length and content
        is_text_passage = len(test_text) > 50 or ' ' in test_text or any(c in test_text for c in '.,!?;:')
        
        # Create the question based on whether this item is a text passage or word/string and single/multiple letters
        if len(target_letters) == 1:
            # Single letter question
            target_letter = target_letters[0]
            if is_text_passage:
                question_text = f"How many {target_letter}s are in the following text: \"{test_text}\"?"
            else:
                question_text = f"How many {target_letter}s are in the string {test_text}?"
            
            # Add instruction for single letter answer format
            question_with_instruction = f'{question_text}\n\nProvide your answer in the format: <answer>{{number}}</answer>'
        else:
            # Multiple letters question
            letters_str = ", ".join(f"'{letter}'" for letter in target_letters[:-1]) + f", and '{target_letters[-1]}'"
            if is_text_passage:
                question_text = f"Count the occurrences of the letters {letters_str} in the following text: \"{test_text}\""
            else:
                question_text = f"Count the occurrences of the letters {letters_str} in the string {test_text}"
            
            # Add instruction for multiple letter JSON answer format
            example_json = "{" + ", ".join(f'"{letter}": 0' for letter in target_letters) + "}"
            question_with_instruction = f'{question_text}\n\nProvide your answer as JSON in the format: <answer>{example_json}</answer>'

        # Create messages for model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_with_instruction},
        ]

        # Apply chat template to convert messages to a single string
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Get model completion
        completion = await self.server.completion(
            prompt=prompt,
            n=1,
            max_tokens=self.config.max_generation_tokens,
            temperature=self.config.eval_temperature,
            split="eval",
        )

        # Extract the model's response from the completion
        model_response = completion.choices[0].text

        # Determine expected format and extract the answer
        expected_format = "single" if len(target_letters) == 1 else "multi"
        model_answer = self._extract_answer(model_response, expected_format)

        # Score 1 if the answers match, 0 otherwise
        if model_answer is None:
            score = 0
        elif expected_format == "single":
            # Single letter: compare integer
            expected_single_count = expected_counts[target_letters[0]]
            score = 1 if model_answer == expected_single_count else 0
        else:
            # Multiple letters: compare dictionaries
            score = 1 if (set(model_answer.keys()) == set(target_letters) and 
                         all(model_answer.get(letter, -1) == expected_counts[letter] for letter in target_letters)) else 0

        return score

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the model on test data.
        """
        self.logger.info("Starting evaluation...")
        if not self.test_words:
            self.logger.warning("No test texts available for evaluation. Skipping.")
            self.eval_metrics.append(("eval/percent_correct", 0.0))
            return

        eval_tasks = []
        # Sample a subset of test texts for evaluation to keep it manageable
        eval_texts = random.sample(self.test_words, min(len(self.test_words), self.config.eval_sample_size))
        
        text_type = "mixed items (words and passages)" if self.config.use_text_passages else "strings"
        self.logger.info(f"Starting evaluation on {len(eval_texts)} test {text_type}...")
        
        for test_text in eval_texts:
            eval_tasks.append(self.rollout_and_score_eval(test_text))

        # Run evaluation
        scores = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating")
        
        if not scores:
            percent_correct = 0.0
        else:
            percent_correct = sum(scores) / len(scores)
            
        self.eval_metrics.append(("eval/percent_correct", percent_correct))
        self.logger.info(f"Evaluation finished. Percent correct: {percent_correct:.4f}")

    async def add_rollouts_for_wandb(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
    ):
        if item is None or scored_data is None or not scored_data.get("tokens"):
            return

        expected_counts = item[1]  # correct counts (dict)
        text = item[2]  # text (word or sentence)
        target_letters = item[3]  # target letters (list)
        difficulty_score = item[4] if len(item) > 4 else 0.0  # difficulty score
        
        # Handle legacy format
        if isinstance(target_letters, str):
            target_letters = [target_letters]
            expected_counts = {target_letters[0]: expected_counts}
        elif isinstance(expected_counts, int):
            expected_counts = {target_letters[0]: expected_counts}

        # save rollout to trajectory
        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = self.config.group_size

        # Make sure there's data to log
        num_keep = min(num_keep, len(scored_data["tokens"]))
        if num_keep == 0:
            return

        current_rollouts = []
        for i in range(num_keep):
            # Ensure tokens and scores have the same length
            if i < len(scored_data["tokens"]) and i < len(scored_data["scores"]):
                # Decode the full trajectory including prompt and model response
                full_text = self.tokenizer.decode(
                    scored_data["tokens"][i], skip_special_tokens=True
                )
                score_val = scored_data["scores"][i]
                expected_str = str(expected_counts) if len(target_letters) > 1 else str(expected_counts[target_letters[0]])
                letters_str = ", ".join(target_letters)
                current_rollouts.append(
                    (full_text, score_val, expected_str, text[:100], letters_str, difficulty_score)
                )
            else:
                self.logger.warning(
                    f"Mismatch in lengths of tokens/scores for wandb logging at index {i}."
                )

        self.rollouts_for_wandb.append(current_rollouts)

        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics):
        if len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(columns=["full_text", "score", "expected_counts", "text", "target_letters", "difficulty_score"])
            for group in self.rollouts_for_wandb:
                for item in group:
                    # Handle both old format (5 items) and new format (6 items)
                    if len(item) >= 6:
                        table.add_data(item[0], item[1], item[2], item[3], item[4], item[5])
                    else:
                        table.add_data(item[0], item[1], item[2], item[3], item[4], 0.0)
            wandb_metrics["train/rollouts"] = table
        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Try to calculate percent_correct, pass if there's a division by zero
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        self.percent_correct_buffer = list()
        
        # Add eval metrics
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        # Add comprehensive letter counting specific metrics
        if self.letter_distribution_stats:
            # Log letter distribution statistics
            total_letters_asked = sum(self.letter_distribution_stats.values())
            wandb_metrics["stats/total_letters_asked"] = total_letters_asked
            
            # Log most and least common letters asked
            most_common_letter = max(self.letter_distribution_stats, key=self.letter_distribution_stats.get)
            least_common_letter = min(self.letter_distribution_stats, key=self.letter_distribution_stats.get)
            wandb_metrics["stats/most_common_letter_count"] = self.letter_distribution_stats[most_common_letter]
            wandb_metrics["stats/least_common_letter_count"] = self.letter_distribution_stats[least_common_letter]
            
            # Log distribution entropy (measure of how evenly distributed the letters are)
            import math
            entropy = -sum(
                (count / total_letters_asked) * math.log2(count / total_letters_asked)
                for count in self.letter_distribution_stats.values()
                if count > 0
            )
            wandb_metrics["stats/letter_distribution_entropy"] = entropy

        if self.word_length_stats:
            # Log word length statistics
            total_words_asked = sum(self.word_length_stats.values())
            wandb_metrics["stats/total_words_asked"] = total_words_asked
            
            # Calculate average word length
            avg_word_length = sum(
                length * count for length, count in self.word_length_stats.items()
            ) / total_words_asked
            wandb_metrics["stats/avg_word_length"] = avg_word_length
            
            # Log min and max word lengths seen
            wandb_metrics["stats/min_word_length"] = min(self.word_length_stats.keys())
            wandb_metrics["stats/max_word_length"] = max(self.word_length_stats.keys())

        # Log error rates
        total_items_processed = self.processed_item_count + self.failed_processed_item_count
        if total_items_processed > 0:
            wandb_metrics["errors/answer_format_error_rate"] = self.answer_format_errors / (total_items_processed * self.config.group_size)
            wandb_metrics["errors/think_format_error_rate"] = self.think_format_errors / (total_items_processed * self.config.group_size)
            wandb_metrics["errors/total_format_errors"] = self.answer_format_errors + self.think_format_errors

        # Log data dumping progress
        if self.config.dump_rollouts:
            wandb_metrics["data_dumps/processed_item_count"] = self.processed_item_count
            wandb_metrics["data_dumps/rollouts_buffer_size"] = len(self.rollouts_to_save_buffer)
            wandb_metrics["data_dumps/save_file_batch_num"] = self.save_file_batch_num

        if self.config.dump_failed_rollouts:
            wandb_metrics["data_dumps/failed_processed_item_count"] = self.failed_processed_item_count
            wandb_metrics["data_dumps/failed_rollouts_buffer_size"] = len(self.failed_rollouts_to_save_buffer)
            wandb_metrics["data_dumps/failed_save_file_batch_num"] = self.failed_save_file_batch_num

        # Add rollout table
        wandb_metrics = await self.create_rollout_table(wandb_metrics)

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    LetterCountingEnv.cli()
