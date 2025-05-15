import json
import random
import re
from typing import Dict, List, Optional, Tuple, Union

import wandb
from datasets import load_dataset, Dataset # Added Dataset for dummy data
from tqdm.asyncio import tqdm_asyncio
from langdetect import detect, LangDetectException

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# System prompt can be reused or adapted for instruction following tasks
system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)


class InstructionFollowingEnv(BaseEnv):
    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.rollouts_for_wandb = []
        # self.completion_lengths = [] # Kept from SingleToolCallingEnv, assess utility

    @classmethod
    def config_init(self) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        # Configuration for the Instruction Following Environment
        env_config = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview", # Or other suitable tokenizer
            group_size=32, # Number of rollouts per group
            use_wandb=True,
            rollout_server_url="http://localhost:8000", # Assuming same rollout server
            total_steps=2000,
            batch_size=1024, # Samples per training batch
            steps_per_eval=20,
            max_token_length=1024 * 2, # Max token length for model generation, adjust as needed for IF
            inference_weight=1.0,
            wandb_name="instruction_following_rlvr", # Specific WandB project name
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            # Add any specific dataset configurations here if needed
            # dataset_name="allenai/ifeval",
            # dataset_config_name="default", # Or specific config for the dataset
        )
        # Server configurations can be similar to SingleToolCallingEnv or adjusted
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            ),
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9005/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            ),
        ]
        return env_config, server_configs

    async def create_rollout_table(self, wandb_metrics):
        # Logs rollouts to a WandB table for visualization
        if len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(columns=["text", "score", "constraint_details"])
            for group in self.rollouts_for_wandb:
                for item in group:
                    # item[0] is model output, item[1] is score, item[2] is constraint info
                    table.add_data(item[0], item[1], json.dumps(item[2]))
            wandb_metrics["train/rollouts"] = table
        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        # Logs metrics to WandB
        if wandb_metrics is None:
            wandb_metrics = dict()

        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            pass # Buffer might be empty

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        """
        Load and preprocess the dataset for instruction following.
        This method expects each data item from the loaded dataset to have at least:
        - 'prompt': The instruction text for the LLM (string).
        - 'func_name': The string name of the verifier function from IF_FUNCTIONS_MAP.
        - 'args_json': A JSON string representing the dictionary of arguments for the verifier function.
        
        Example item structure expected from the dataset loader (after any initial parsing):
        {
            "prompt": "Include the keywords 'apple' and 'banana' in your response.",
            "func_name": "verify_keywords",
            "args_json": "{\\"keyword_list\\": [\\"apple\\", \\"banana\\"]}", // Note: escaped for Python string
            "original_constraints_for_logging": "Include keywords {apple}, {banana} in your response", // Optional
            "expected_response_for_logging": "An apple and a banana are fruits." // Optional
        }

        If your raw dataset (e.g., from Hugging Face) has a natural language constraint string,
        you need to implement a parsing step (either before this environment or at the beginning
        of this setup method) to convert that string into 'func_name' and 'args_json'.
        The verifier functions and IF_FUNCTIONS_MAP included in this file define the available functions
        and their expected argument names.
        """
        dataset_name = getattr(self.config, "dataset_name", "allenai/ifeval") # Example dataset
        dataset_config_name = getattr(self.config, "dataset_config_name", None)

        processed_items = []
        try:
            # Attempt to load the dataset specified in the config
            # This section assumes 'dataset_name' provides items with 'prompt', 'func_name', and 'args_json'
            print(f"Attempting to load dataset: {dataset_name}, config: {dataset_config_name}")
            if dataset_config_name:
                full_dataset_raw = load_dataset(dataset_name, dataset_config_name, split="train", trust_remote_code=True)
            else:
                full_dataset_raw = load_dataset(dataset_name, split="train", trust_remote_code=True)
            print(f"Successfully loaded raw dataset. Number of items: {len(full_dataset_raw)}")

            for i, item in enumerate(full_dataset_raw):
                prompt_text = item.get("prompt")
                func_name_from_item = item.get("func_name")
                args_json_from_item = item.get("args_json") # Expecting a JSON string

                if not prompt_text or not func_name_from_item or args_json_from_item is None: # Check explicitly for None
                    print(f"Warning: Item {i} missing 'prompt', 'func_name', or 'args_json'. Skipping. Item: {item}")
                    continue
                
                if func_name_from_item not in IF_FUNCTIONS_MAP:
                    print(f"Warning: func_name '{func_name_from_item}' in item {i} not in IF_FUNCTIONS_MAP. Skipping. Prompt: {prompt_text[:50]}...")
                    continue

                try:
                    args_dict = json.loads(args_json_from_item)
                    if not isinstance(args_dict, dict):
                        # Allow empty string for args_json to represent empty dict, but json.loads('') fails
                        # However, json.loads('{}') is fine. Assume args_json is valid JSON if not empty.
                        raise ValueError("Parsed args_json is not a dictionary.")
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse 'args_json' for item {i}. Error: {e}. Args JSON: '{args_json_from_item}'. Prompt: {prompt_text[:50]}... Skipping.")
                    continue
                except ValueError as e: # Catches the "not a dictionary" error
                    print(f"Warning: Parsed 'args_json' was not a dictionary for item {i}. Error: {e}. Args JSON: '{args_json_from_item}'. Prompt: {prompt_text[:50]}... Skipping.")
                    continue
                
                processed_items.append({
                    "prompt": prompt_text,
                    "func_name": func_name_from_item,
                    "args": args_dict, # Parsed dictionary of arguments
                    "original_constraints_for_logging": item.get("original_constraints_for_logging", str(item.get("constraints", ""))), # For logging
                    "expected_response_for_logging": item.get("expected_response_for_logging", str(item.get("response", ""))) # For logging
                })

            if not processed_items:
                print("Warning: No items successfully processed from the dataset. Check dataset format/content or parsing logic if any.")
                # Fallback to dummy data if processing yields nothing, to allow environment to initialize.
                # This indicates a problem with the primary dataset source or its assumed structure.
                raise ValueError("Dataset processing resulted in no valid items. Cannot proceed without data or a valid dummy fallback.")
            
            full_dataset = Dataset.from_list(processed_items)
            print(f"Successfully processed {len(full_dataset)} items from dataset.")

        except Exception as e:
            print(f"CRITICAL: Failed to load or process primary dataset '{dataset_name}': {e}. Using a small DUMMY dataset as a fallback.")
            # Fallback to a minimal dummy dataset with the expected structure for 'args' already parsed
            dummy_data_for_fallback = [
                {
                    "prompt": "Dummy Instruction 1: Ensure your response contains the word 'example'.",
                    "func_name": "verify_keywords",
                    "args": {"keyword_list": ["example"]}, 
                    "original_constraints_for_logging": "Contains 'example'",
                    "expected_response_for_logging": "This is an example response."
                },
                 {
                    "prompt": "Dummy Instruction 2: Output a valid JSON with key 'data' and value 'test'.",
                    "func_name": "validate_json_format",
                    "args": {}, 
                    "original_constraints_for_logging": "Output valid JSON.",
                    "expected_response_for_logging": "{\\\"data\\\": \\\"test\\\"}" # Corrected: Escaped for Python string
                }
            ]
            full_dataset = Dataset.from_list(dummy_data_for_fallback)
            print(f"Initialized with DUMMY dataset of {len(full_dataset)} items.")
            
        full_dataset = full_dataset.shuffle(seed=42)
        
        actual_test_size = 0.2
        num_items = len(full_dataset)

        if num_items == 0:
            print("ERROR: Dataset is empty. Cannot create train/test split.")
            self.train = Dataset.from_list([])
            self.test = Dataset.from_list([])
        elif num_items == 1:
            print("Warning: Dataset has only 1 item. Using it for both train and test.")
            self.train = full_dataset
            self.test = full_dataset
        else: # num_items > 1
            # Ensure test_size results in at least 1 item for test set if possible, but not more than train set
            if num_items < 5 : # For 2,3,4 items, make test size 1
                 min_test_items = 1
            else: # For 5+ items, 20% is fine
                 min_test_items = max(1, int(num_items * actual_test_size))

            # Ensure test split is not too large, e.g. not more than 50% unless dataset is very small
            # And ensure train always has at least one item if num_items > 1
            calculated_test_size = min_test_items / num_items
            if calculated_test_size >= 0.5 and num_items > 2: # If test is 50% or more and we have 3+ items
                calculated_test_size = (num_items -1) / num_items # Make train have at least 1

            split_dataset = full_dataset.train_test_split(test_size=calculated_test_size, seed=42)
            self.train = split_dataset["train"]
            self.test = split_dataset["test"]
            # Final check for empty train/test after split, should not happen with logic above if num_items > 0
            if len(self.train) == 0 and len(self.test) > 0:
                print("Warning: Train set empty after split, test set has data. This is unusual. Swapping.")
                self.train = self.test # Fallback, though indicates issue
            elif len(self.test) == 0 and len(self.train) > 0:
                 print("Warning: Test set empty after split, train set has data. Using full train set for test as well.")
                 self.test = self.train


        self.iter = 0
        print(f"Dataset setup complete. Train size: {len(self.train)}, Test size: {len(self.test)}")


    async def _get_score_from_verifier(self, model_response_text: str, func_name: str, args: Dict) -> float:
        """Helper to call verifier function and get a numerical score."""
        if func_name not in IF_FUNCTIONS_MAP:
            print(f"Warning: Verifier function '{func_name}' not found in IF_FUNCTIONS_MAP.")
            return 0.0

        verifier_func = IF_FUNCTIONS_MAP[func_name]
        
        raw_score = None
        try:
            # For validate_response_language, langdetect is now imported at the top.
            # Specific argument handling for functions that don't take generic **args
            # or have special return types should be done before the generic call.
            if func_name == "validate_placeholders":
                # validate_placeholders expects 'text' and 'N', returns (bool, list)
                raw_score = verifier_func(model_response_text, N=args.get("N"))
            elif func_name == "verify_bullet_points":
                # verify_bullet_points expects 'text' and 'N', returns bool (or was (bool,str) in one doc)
                # Assuming it now consistently returns bool as per our integrated version
                raw_score = verifier_func(model_response_text, N=args.get("N"))
            # Add other specific handlers here if necessary, otherwise use generic **args
            else:
                raw_score = verifier_func(model_response_text, **args)

        except LangDetectException: # Specifically catch for language detection issues
            print(f"Warning: langdetect failed for func_name '{func_name}'. Scoring as incorrect.")
            return 0.0
        except ImportError as e:
            # This might happen if a function tries a lazy import that fails (not langdetect now)
            print(f"Warning: ImportError during verifier function '{func_name}': {e}. Check dependencies.")
            return 0.0
        except TypeError as e:
            print(f"TypeError calling {func_name} with args {args}: {e}. Text: '{model_response_text[:100]}...'")
            return 0.0
        except Exception as e: # Catch any other unexpected error from a verifier
            print(f"Unexpected error in verifier function '{func_name}' with args {args}: {e}")
            return 0.0

        # Convert boolean or tuple[boolean, ...] to float score
        if isinstance(raw_score, tuple):
            score_value = float(raw_score[0]) # Assuming the first element is the boolean score
        elif isinstance(raw_score, bool):
            score_value = float(raw_score)
        else:
            print(f"Warning: Verifier '{func_name}' returned unexpected type: {type(raw_score)}. Expected bool or tuple.")
            score_value = 0.0
        
        return score_value

    async def rollout_and_score_eval(self, test_item: Dict):
        # test_item is a dictionary from the test set, processed by setup()
        # It should contain 'prompt', 'func_name', 'args'
        instruction_prompt_text = test_item["prompt"]
        func_name = test_item["func_name"]
        args_for_verifier = test_item["args"]

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": instruction_prompt_text})

        prompt_str = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        completion = await self.server.completion(
            prompt=prompt_str,
            n=1,
            max_tokens=self.config.max_token_length, # Use config for max_tokens
            temperature=0.7, # Temperature for eval, can be 0 for deterministic
            split="eval",
        )

        model_response_text = completion.choices[0].text
        score_value = await self._get_score_from_verifier(model_response_text, func_name, args_for_verifier)
        
        return score_value # Returns 1.0 for correct, 0.0 for incorrect based on verifier

    async def evaluate(self, *args, **kwargs):
        # Evaluates the model on the test set
        if not self.test or len(self.test) == 0:
            print("Warning: Test set is empty. Skipping evaluation.")
            self.eval_metrics.append(("eval/percent_correct", 0.0))
            return

        eval_tasks = []
        for test_item_dict in self.test: # self.test contains dicts after setup
            eval_tasks.append(self.rollout_and_score_eval(test_item_dict))
        
        scores = await tqdm_asyncio.gather(*eval_tasks)
        
        if not scores: # If gather returns empty list
             percent_correct = 0.0
        else:
            percent_correct = sum(scores) / len(scores)
            
        self.eval_metrics.append(("eval/percent_correct", percent_correct))
        print(f"Evaluation percent correct: {percent_correct}")


    async def collect_trajectories(self, item: Item) -> Tuple[Optional[ScoredDataGroup], List]:
        # item = (prompt_messages_tuple, answer_info_dict)
        # answer_info_dict = {"func_name": ..., "args": ...}
        prompt_messages_list = [dict(msg_fset) for msg_fset in item[0]]
        answer_info = item[1]

        prompt_str = self.tokenizer.apply_chat_template(
            prompt_messages_list, add_generation_prompt=True, tokenize=False
        )

        completions = await self.server.completion(
            prompt=prompt_str,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
            temperature=0.8, # Temperature for diverse responses during training rollouts
        )

        to_score_list = []
        for choice in completions.choices:
            trajectory_messages = [dict(msg_fset) for msg_fset in item[0]] # Fresh copy
            trajectory_messages.append({"role": "assistant", "content": choice.text})
            to_score_list.append((tuple(trajectory_messages), answer_info)) # Pass answer_info

        if not to_score_list:
            return None, []
            
        scored_data = await self.score(to_score_list)
        to_backlog = [] # Backlog not currently used but part of signature

        return scored_data, to_backlog

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)
    
    async def score(
        self, rollout_group_data: List[Tuple[tuple, Dict]]
    ) -> Optional[ScoredDataGroup]:
        # rollout_group_data is a list of (trajectory_messages_tuple, answer_info_dict)
        # answer_info_dict = {"func_name": ..., "args": ...}

        scores_container = ScoredDataGroup()
        scores_container["tokens"] = list()
        scores_container["masks"] = list()
        scores_container["scores"] = list()

        if not rollout_group_data:
            return None

        # The 'answer_info' (func_name, args) is consistent for all items in this group,
        # as it comes from the same initial prompt.
        # We can extract it once if needed, but it's passed per item.
        
        random.shuffle(rollout_group_data) # Shuffle to avoid bias

        for trajectory_item in rollout_group_data:
            full_trajectory_messages = trajectory_item[0]
            answer_info = trajectory_item[1] # {"func_name": ..., "args": ...}
            
            model_response_text = full_trajectory_messages[-1]["content"]
            func_name = answer_info["func_name"]
            args_for_verifier = answer_info["args"]

            # Get score (1.0 for correct, 0.0 for incorrect from verifier)
            score_value = await self._get_score_from_verifier(model_response_text, func_name, args_for_verifier)
            
            # Map to reward: 1.0 for correct, -1.0 for incorrect
            reward = 1.0 if score_value == 1.0 else -1.0

            # Tokenize the conversation for PPO training
            # Ensure full_trajectory_messages is a list of dicts
            list_of_dicts_trajectory = [dict(msg) for msg in full_trajectory_messages]
            out_dict = tokenize_for_trainer(self.tokenizer, list_of_dicts_trajectory)
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Filter out examples with insufficient context (too short)
            if sum(1 for m_val in masks if m_val != -100) < 10: # At least 10 non-masked tokens
                continue

            scores_container["tokens"].append(tokens)
            scores_container["masks"].append(masks)
            scores_container["scores"].append(reward)

            # Stop if we have enough examples for the group
            if len(scores_container["tokens"]) >= self.config.group_size:
                break
        
        if not scores_container["tokens"]: # No valid items collected
            return None

        # Record success rate for logging (based on positive rewards)
        for rwd in scores_container["scores"]:
            self.percent_correct_buffer.append(max(0, rwd)) # If reward is 1.0, it's a success

        # Optional: Apply length penalty if all responses are correct (reward 1.0)
        # This logic is from SingleToolCallingEnv, may need adjustment for IF
        if all(s == 1.0 for s in scores_container["scores"]):
            token_lengths = [len(t) for t in scores_container["tokens"]]
            if not token_lengths or max(token_lengths) == 0:
                return scores_container # Avoid division by zero, or if all empty

            max_allowed_length = self.config.max_token_length
            # Threshold can be adjusted, e.g., 50% of max_token_length
            length_threshold = max_allowed_length * 0.5 

            penalized_scores = []
            for i, length in enumerate(token_lengths):
                original_score = scores_container["scores"][i] # Should be 1.0 here
                if length <= length_threshold:
                    penalized_scores.append(original_score)
                else:
                    # Linear penalty for exceeding threshold
                    penalty_factor = (length - length_threshold) / (max_allowed_length - length_threshold)
                    penalty_factor = min(penalty_factor, 1.0) # Cap penalty factor at 1
                    # Penalized score scales from original_score down to original_score * (1-1) = 0
                    penalized_scores.append(original_score * (1.0 - penalty_factor))
            scores_container["scores"] = penalized_scores


        # If all scores are identical after potential penalties, no learning signal
        if len(set(scores_container["scores"])) <= 1 and len(scores_container["scores"]) > 1 :
            return None # Avoid sending data with no variance

        return scores_container

    async def get_next_item(self) -> Item:
        # Fetches the next preprocessed item from the training set
        if not self.train or len(self.train) == 0:
            # This case should be handled by setup, but as a safeguard:
            print("Error: Training data is empty in get_next_item.")
            # Return a dummy item to prevent crashes, though this indicates a setup issue
            dummy_prompt_messages = (
                frozenset({"role": "system", "content": system_prompt}.items()),
                frozenset({"role": "user", "content": "Dummy instruction: say hello."}.items())
            )
            dummy_answer_info = {"func_name": "verify_keywords", "args": {"keyword_list": ["hello"]}}
            return (dummy_prompt_messages, dummy_answer_info)


        raw_item = self.train[self.iter % len(self.train)] # raw_item is a dict
        self.iter += 1

        instruction_prompt_text = raw_item["prompt"]
        
        # Construct messages for the LLM (prompt tuple part of Item)
        # Using frozenset as required by BaseEnv's Item type hint
        prompt_messages_tuple = (
            frozenset({"role": "system", "content": system_prompt}.items()),
            frozenset({"role": "user", "content": instruction_prompt_text}.items()),
        )

        # The "answer" part for scoring purposes (answer_info dict part of Item)
        answer_info = {
            "func_name": raw_item["func_name"],
            "args": raw_item["args"],
            # Optionally include other info for logging/debugging if needed from raw_item
            "original_constraints_for_logging": raw_item.get("original_constraints", ""),
            "expected_response_for_logging": raw_item.get("expected_response_for_logging", "")
        }
        
        return (prompt_messages_tuple, answer_info)

    async def add_rollouts_for_wandb(
        self,
        scored_data: ScoredDataGroup, # Assuming single ScoredDataGroup here
        item: Item = None, # item = (prompt_messages_tuple, answer_info_dict)
    ):
        # Saves rollouts for WandB logging
        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1: # Log all rollouts in the group
            num_keep = len(scored_data["tokens"])
        
        # item[1] is the answer_info_dict containing func_name and args
        constraint_details_for_log = item[1] if item else {}

        rollout_batch = []
        for i in range(min(num_keep, len(scored_data["tokens"]))):
            decoded_text = self.tokenizer.decode(scored_data["tokens"][i], skip_special_tokens=True)
            score = scored_data["scores"][i]
            rollout_batch.append((decoded_text, score, constraint_details_for_log))
        
        self.rollouts_for_wandb.append(rollout_batch)

        # Limit the number of rollout groups stored
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)


# ----- IFEval Verifier Functions and Map -----
# adapted from https://github.com/allenai/open-instruct/blob/main/scripts/eval_constraints/if_functions.py

# Helper function for verify_keyword_frequency, moved import re to top level
def _extract_words(text: str) -> List[str]:
    return re.findall(r"\\b\\w+\\b", text.lower())

# include keywords: Include keywords {keyword1}, {keyword2} in your response
def verify_keywords(text: str, keyword_list: List[str]) -> bool:
    response_lower = text.lower()
    return all(keyword.lower() in response_lower for keyword in keyword_list)

# Keyword Frequency: In your response, the word {word} should appear {N} times.
def verify_keyword_frequency(text: str, word: str, N: int) -> bool:
    text_lower = text.lower()
    keyword_lower = word.lower()
    words = _extract_words(text_lower)
    actual_count = sum(1 for w in words if w == keyword_lower)
    return actual_count == N

# Forbidden Words: Do not include keywords {forbidden words} in the response.
def validate_forbidden_words(text: str, forbidden_words: List[str]) -> bool:
    text_lower = text.lower()
    return not any(word.lower() in text_lower for word in forbidden_words)

# Letter Frequency : In your response, the letter {letter} should appear {N} times.
def verify_letter_frequency(text: str, letter: str, N: int) -> bool:
    if len(letter) != 1:
        # This should ideally raise ValueError, but for RL reward, return False
        return False
    actual_count = text.count(letter)
    return actual_count == N

# Response Language: Your ENTIRE response should be in {language}, no other language is allowed.
def validate_response_language(text: str, language: str) -> bool:
    try:
        detected_language = detect(text)
        return detected_language == language
    except LangDetectException: # Catching specific exception from detect()
        print(f"Warning: langdetect failed to detect language for text: '{text[:50]}...'")
        return False


# Number Paragraphs: Your response should contain {N} paragraphs. You separate paragraphs using the markdown divider:
# * * *
def verify_paragraph_count(text: str, N: int) -> bool:
    def clean_text(txt: str) -> str:
        return "\\n".join(line.strip() for line in txt.splitlines()).strip()
    cleaned_text = clean_text(text)
    # Paragraphs are separated by '* * *'. N dividers mean N+1 paragraphs.
    # If the text IS paragraphs, then N paragraphs will have N-1 dividers.
    # The prompt implies N paragraphs are expected.
    # If N=1, 0 dividers. If N=2, 1 divider. So, count of parts = N.
    paragraphs = cleaned_text.split("* * *") 
    actual_count = len(paragraphs)
    # Verify each split resulted in non-empty content, if text itself is not empty
    if not cleaned_text and N == 0 : return True # 0 paragraphs, empty text
    if not cleaned_text and N > 0 : return False

    valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    # This check might be too strict if empty paragraphs are allowed by the constraint definition
    # If "paragraph" implies non-empty content:
    # return len(valid_paragraphs) == N and actual_count == N
    # If constraint just means N segments separated by dividers:
    return actual_count == N


# Number Words: Answer with at least / around / at most {N} words
def validate_word_constraint(text: str, N: int, quantifier: str) -> bool:
    words = text.strip().split()
    actual_count = len(words)
    tolerance = max(round(N * 0.1), 1) # For 'around'

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "at most":
        return actual_count <= N
    elif quantifier == "around":
        return abs(actual_count - N) <= tolerance
    return False

# Number Sentences: Answer with at least / around / at most {N} sentences.
def verify_sentence_constraint(text: str, N: int, quantifier: str) -> bool:
    # Basic sentence splitting, might need more robust NLP for complex cases
    sentences = re.split(r'(?<!\\w\\.\\w.)(?<![A-Z][a-z]\\.)(?<=\\.|\\?|!)\\s', text.strip())
    # Filter out empty strings that might result from splitting
    sentences = [s for s in sentences if s.strip()]
    actual_count = len(sentences)

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "around":
        # "around" for sentences usually means exact or +/-1
        return abs(actual_count - N) <= 1
    elif quantifier == "at most":
        return actual_count <= N
    return False

# Number Paragraphs + First Word in i-th Paragraph
def validate_paragraphs(text: str, N: int, first_word: str, i: int) -> bool:
    # Paragraphs separated by double line breaks
    paragraphs = text.split("\\n\\n")
    if len(paragraphs) != N:
        return False
    # i is 1-indexed for paragraph number
    if not (1 <= i <= len(paragraphs)):
        return False
    # Check first word of the i-th paragraph
    # .strip() to handle leading/trailing whitespace in paragraph
    # .split()[0] to get the first word
    try:
        actual_first_word = paragraphs[i - 1].strip().split()[0]
        # Case-insensitive comparison for first_word might be more robust
        return actual_first_word.lower() == first_word.lower()
    except IndexError: # Handles empty paragraph or paragraph without words
        return False


# Postscript: At the end of your response, please explicitly add a postscript starting with {postscript marker}
def verify_postscript(text: str, postscript_marker: str) -> bool:
    marker_index = text.rfind(postscript_marker) # Find last occurrence
    if marker_index == -1:
        return False
    # Check if it's truly a postscript (i.e., near the end, and has content after marker)
    # This interpretation: marker exists, and something follows it OR it's at the very end.
    # The original IFEval might have a stricter definition (e.g. specific distance from end)
    # A simple check: marker is present and the text from marker to end is mostly the postscript.
    # For RL, simpler: marker is present and is not just prefix of a word.
    # Test if the marker is at a word boundary if it's not the start of the string
    if marker_index > 0 and text[marker_index-1].isalnum() and postscript_marker[0].isalnum():
        # Avoid matching mid-word, e.g. "script" in "postscript" if marker is "script"
        # This check is heuristic. A regex with word boundaries might be better.
        pass # Heuristic, might need refinement

    # Check if content exists after marker, or if marker itself is the end
    remaining_text = text[marker_index:].strip()
    return len(remaining_text) >= len(postscript_marker.strip())


# Number Placeholder: The response must contain at least {N} placeholders ... [address].
def validate_placeholders(text: str, N: int) -> Tuple[bool, List[str]]:
    placeholders_found = re.findall(r'\\[(.*?)\\]', text) # Matches [content]
    return len(placeholders_found) >= N, placeholders_found

# Number Bullets: Your answer must contain exactly {N} bullet points. * This is a point.
def verify_bullet_points(text: str, N: int) -> bool: # Original had tuple[bool,str] in doc, bool in code
    lines = text.splitlines()
    # Markdown bullets usually start with '*', '-', or '+' followed by a space.
    bullet_points = [line.strip() for line in lines if re.match(r'^(\\s*)[\\*\\-\\+]\\s+', line.strip())]
    return len(bullet_points) == N

# Title: Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.
def validate_title(text: str) -> bool:
    return bool(re.search(r'<<(.*?)>>', text))

# Choose: From Answer with one of the following options: {options}
def validate_choice(text: str, options: List[str]) -> bool:
    # Assuming 'text' should be one of the 'options' exactly, or contain one of them.
    # The original prompt "Answer with one of..." implies the response *is* one of the options.
    # Case-insensitive comparison for robustness.
    text_cleaned = text.strip().lower()
    return any(text_cleaned == opt.strip().lower() for opt in options)

# Minimum Number Highlighted Section: Highlight at least {N} sections ... *highlighted section*
def validate_highlighted_sections(text: str, N: int) -> bool:
    # Markdown italics/bold *highlight* or **highlight**
    # This regex looks for single asterisks: *content*
    matches = re.findall(r'\\*(.*?)\\*(?<!\\*)', text) # Non-greedy match inside single asterisks
    # Filter out empty matches or those that are just whitespace if needed.
    # matches = [m for m in matches if m.strip()]
    return len(matches) >= N

# Multiple Sections: Your response must have {N} sections. Mark ... with {section splitter} X.
def validate_sections(text: str, N: int, section_splitter: str) -> bool:
    # Example: section_splitter = "Section" -> "Section 1", "Section 2"
    # This implies the splitter itself might include a number or be just the prefix.
    # If splitter is "---", then text.split("---").
    # If splitter is "Topic X:", this is more complex.
    # Assuming a simple string split is intended by the original IFEval function.
    # The prompt phrasing "Mark the beginning of each section with {section splitter} X"
    # suggests counting occurrences of the splitter pattern.
    
    # If section_splitter is like "SECTION", we'd look for "SECTION 1", "SECTION 2", ...
    # This is hard to generalize perfectly without knowing how IFEval defines 'X'.
    # Simplest: count occurrences of the base splitter string.
    # sections = text.split(section_splitter)
    # num_sections = len(sections) -1 if sections[0].strip() == "" else len(sections)
    # A slightly more robust way for "Splitter X":
    # Count how many times "splitter" followed by something (like a number) appears.
    # Example: if splitter is "Chapter", we look for "Chapter 1", "Chapter ...".
    # This regex is a placeholder for more specific logic IFEval might use.
    
    # Let's use a simple count of the splitter string for now.
    # This might need to be adjusted based on IFEval's exact expectation for "X".
    # For "SECTION 1.", "SECTION 2.", if splitter is "SECTION ":
    actual_sections = len(re.findall(re.escape(section_splitter) + r'\\s*\\d*[:\\.\\s]', text, re.IGNORECASE))
    
    # If N=0 and no splitters, it's true. If N>0 and no splitters, false.
    if N == 0: return actual_sections == 0
    return actual_sections == N


# JSON Format : Entire output should be wrapped in JSON format.
def validate_json_format(text: str) -> bool:
    try:
        json.loads(text.strip()) # .strip() to handle leading/trailing whitespace
        return True
    except json.JSONDecodeError:
        return False

# Repeat Prompt: First, repeat the request without change, then give your answer
def validate_repeat_prompt(text: str, original_prompt: str) -> bool:
    # Normalize whitespace for comparison robustness
    text_norm = " ".join(text.strip().split())
    original_prompt_norm = " ".join(original_prompt.strip().split())
    return text_norm.startswith(original_prompt_norm)

# Two Responses: Give two different responses. Separated by 6 asterisk symbols: ******.
def validate_two_responses(text: str) -> bool:
    if text.count("******") == 1:
        parts = text.split("******")
        if len(parts) == 2:
            # Check if parts are non-empty and different
            resp1 = parts[0].strip()
            resp2 = parts[1].strip()
            return bool(resp1 and resp2 and resp1 != resp2)
    return False

# All Uppercase: Your entire response should be in English, capital letters only.
def validate_uppercase(text: str) -> bool:
    # Check if it has letters and all letters are uppercase
    if not any(c.isalpha() for c in text): # No letters, technically not violating "all capital"
        return True # Or False, depending on interpretation of "response"
    return text == text.upper()

# All Lowercase: Your entire response should be in English, and in all lowercase letters.
def validate_lowercase(text: str) -> bool:
    if not any(c.isalpha() for c in text):
        return True
    return text == text.lower()

# Frequency of All-capital Words
def validate_frequency_capital_words(text: str, N: int, quantifier: str) -> bool:
    # Words with all capital letters, e.g., "NASA", "AI". Min 2 chars to be a "word".
    capital_words = re.findall(r'\\b[A-Z]{2,}\\b', text)
    actual_count = len(capital_words)
    tolerance = max(round(N * 0.1), 1) # For 'around'

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "at most":
        return actual_count <= N
    elif quantifier == "around": # Using exact for 'around' with capital words unless specified
        return abs(actual_count - N) <= tolerance # Or just actual_count == N
    return False

# End Checker: Finish your response with this exact phrase {end phrase}.
def validate_end(text: str, end_phrase: str) -> bool:
    # Normalize whitespace at the end of text for robustness
    return text.strip().endswith(end_phrase.strip())

# Quotation: Wrap your entire response with double quotation marks.
def validate_quotation(text: str) -> bool:
    stripped_text = text.strip()
    return stripped_text.startswith('"') and stripped_text.endswith('"')

# No Commas: In your entire response, refrain from the use of any commas.
def validate_no_commas(text: str) -> bool:
    return "," not in text



IF_FUNCTIONS_MAP = {
    "verify_keywords": verify_keywords,
    "verify_keyword_frequency": verify_keyword_frequency,
    "validate_forbidden_words": validate_forbidden_words,
    "verify_letter_frequency": verify_letter_frequency,
    "validate_response_language": validate_response_language,
    "verify_paragraph_count": verify_paragraph_count,
    "validate_word_constraint": validate_word_constraint,
    "verify_sentence_constraint": verify_sentence_constraint,
    "validate_paragraphs": validate_paragraphs,
    "verify_postscript": verify_postscript,
    "validate_placeholders": validate_placeholders,
    "verify_bullet_points": verify_bullet_points,
    "validate_title": validate_title,
    "validate_choice": validate_choice,
    "validate_highlighted_sections": validate_highlighted_sections,
    "validate_sections": validate_sections,
    "validate_json_format": validate_json_format,
    "validate_repeat_prompt": validate_repeat_prompt,
    "validate_two_responses": validate_two_responses,
    "validate_uppercase": validate_uppercase,
    "validate_lowercase": validate_lowercase,
    "validate_frequency_capital_words": validate_frequency_capital_words,
    "validate_end": validate_end,
    "validate_quotation": validate_quotation,
    "validate_no_commas": validate_no_commas,
}

if __name__ == "__main__":
    InstructionFollowingEnv.cli()
