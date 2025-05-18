# environments/hack0/accessibility_env/accessibility_env.py
import os  # For API keys, etc.
from typing import Dict, List, Optional, Tuple  # Common type hints, added Dict

import tenacity

# from bs4 import BeautifulSoup
from transformers.models.auto.tokenization_auto import AutoTokenizer

# Corrected imports for Atropos types
from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import (  # GameHistory might not be needed yet, Item is common
    Item,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class AccessibilityEnvConfig(BaseEnvConfig):
    # Add any custom config fields specific to your env later
    pass


class AccessibilityEnv(BaseEnv):
    name = "accessibility_env"  # A unique name for your environment

    def __init__(
        self,
        config: AccessibilityEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        # Initialize any env-specific attributes here

    @classmethod
    def config_init(cls) -> Tuple[AccessibilityEnvConfig, List[APIServerConfig]]:
        env_config = AccessibilityEnvConfig(
            tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
            group_size=1,  # Smaller for faster testing initially
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=3,  # For process mode, number of items to generate
            batch_size=1,  # Max items in a single call to score (related to group_size)
            steps_per_eval=5,
            max_token_length=2048,
            wandb_name="accessibility_llama_dev",  # Dev run name
        )

        llama_api_key = os.environ.get("LLAMA_API_KEY")
        if not llama_api_key:
            print("WARNING: LLAMA_API_KEY environment variable not set!")

        server_configs = [
            APIServerConfig(
                model_name="Llama-4-Maverick-17B-128E-Instruct-FP8",
                base_url="https://api.llama.com/v1",  # <<<---- Llama API base URL
                api_key=llama_api_key,
                num_requests_for_eval=16,
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        print(f"[{self.name}] Setting up environment...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name, trust_remote_code=True
            )  # tokenizer_name is 'gpt2'
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Set a default chat template if it's not already set
            # This is crucial for tokenizers like 'gpt2' that don't have one by default.
            if self.tokenizer.chat_template is None:
                # A common, simple template. You might need to adjust based on how gpt-3.5-turbo expects chat.
                # For gpt-3.5-turbo, the actual formatting is handled by the API,
                # but for local tokenization for the trainer, we need *a* template.
                # A basic template for generic tokenization:
                self.tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{{ message['role'] + ': ' + message['content'] + '\\n' }}"
                    "{% endfor %}"
                )
                # Alternatively, for many models, a more structured Jinja template like
                # the Llama or ChatML one might be used if you were training with such a format.
                # For just getting token IDs for a generic model for RL, the simple one above might suffice.
                # Or, if tokenize_for_trainer is smart, it might just concatenate.
                # Let's check if a simpler approach is needed for tokenize_for_trainer.
                print(
                    f"[{self.name}] Set a default chat_template for tokenizer '{self.config.tokenizer_name}'."
                )

            print(
                f"[{self.name}] Tokenizer '{self.config.tokenizer_name}' loaded successfully."
            )
        except Exception as e:
            print(
                f"[{self.name}] Error loading tokenizer '{self.config.tokenizer_name}': {e}"
            )
            raise RuntimeError(f"Failed to load tokenizer: {e}") from e

        self.dataset = [
            {
                "id": "ex001",
                "html": "<h1>Welcome</h1><img src='image.jpg'>",
                "issues_to_fix": ["missing_alt_text"],
            },
            {
                "id": "ex002",
                "html": "<label>Name</label><input type='text' name='username'>",
                "issues_to_fix": ["missing_for_attribute_on_label"],
            },
        ]
        self.iter = 0
        print(f"[{self.name}] Setup complete. Loaded {len(self.dataset)} items.")

    async def get_next_item(self) -> Optional[Item]:
        if self.iter >= len(self.dataset):
            if (
                self.iter >= self.config.total_steps
            ):  # Stop after total_steps for 'process'
                return None
            # Potentially loop dataset or handle running out of unique items
            # For hackathon, just stopping might be fine if dataset is small
            # and total_steps is matched to dataset size.
            # self.iter = 0 # To loop
            print(f"[{self.name}] Reached end of dataset or total_steps.")
            return None

        item_data = self.dataset[self.iter]
        self.iter += 1
        # Format item_data into the 'Item' structure Atropos expects
        # Typically (prompt_messages_tuple, gold_answer_or_metadata_tuple)
        # Example:
        # user_prompt = {"role": "user", "content": f"Make this HTML accessible: {item_data['html_snippet']}"}
        # system_prompt_content = "You are an AI assistant specializing in web accessibility. Modify the given
        # HTML to meet WCAG AA standards. Output only the modified HTML."
        # system_prompt = {"role": "system", "content": system_prompt_content}
        # prompt_messages = (system_prompt, user_prompt) # This needs to be a tuple of dicts
        # messages_for_item = tuple(frozenset(p.items()) for p in prompt_messages) # Atropos often expects this format
        # return (messages_for_item, item_data.get('expected_outcome_or_id')) # Second part is for scoring reference

        # Simpler start for prompt:
        # prompt = (
        #     (
        #         {
        #             "role": "system",
        #             "content": "You are an AI assistant. Given HTML, make it more accessible.",
        #         },
        #     ),
        #     ({"role": "user", "content": f"Original HTML: {item_data['html']}"},),
        # )
        # This prompt structure might need adjustment based on how Atropos and the LLM API expect it.
        # The gsm8k example has:
        # user_message = {"role": "user", "content": item["question"]}
        # chat_completions = await self.server.chat_completion(
        #     messages=[{"role": "system", "content": system_prompt}, user_message], ...
        # So a list of dicts is passed to chat_completion.
        # The 'Item' type for get_next_item is often a tuple: ( (message_part_1, message_part_2, ...),
        # metadata_for_scoring )
        # where each message_part is often a frozenset of items from a dict. This is a bit complex.
        # Let's start with a simple string prompt and adapt.
        # For now, let's assume item is (prompt_string, metadata_for_scoring)
        # The `collect_trajectories` in coding_server.py takes `item: Item`
        # and then accesses `item[0][0]` which implies item is nested.
        # `prompt = tuple([frozenset({"role": "user", "content": next_item["description"]}.items())])`
        # `return (prompt, answer)`
        # So, first element of item is a tuple of frozensets.

        # Let's simplify for now and refine based on Atropos internals if needed.
        # We'll construct the messages list directly in collect_trajectories.
        # So get_next_item can return the raw data needed.
        return item_data  # This will be like {"html": "...", "id": "..."}

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        # 'item_data' here is what get_next_item returned.
        original_html = item["html"]
        system_message_content = (
            "You are an expert web developer specializing in accessibility. "
            "Given the following HTML snippet, please make the minimal necessary modifications "
            "to ensure it meets WCAG 2.1 AA standards for the issues present. "
            "Output only the complete, modified HTML snippet. Do not include explanations unless explicitly asked."
        )
        user_message_content = (
            f"Original HTML:\n```html\n{original_html}\n```\nModified HTML:"
        )

        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content},
        ]

        try:
            chat_completions = await self.server.chat_completion(
                messages=messages,
                n=self.config.group_size,  # Number of completions
                # `max_tokens` here is for the *completion* part, not the whole context.
                # Your Llama API example used 256. Adjust as needed for HTML output.
                max_tokens=1024,  # Max tokens for the LLM's response
                # temperature=0.7, # Optional: adjust for creativity vs. determinism
                # model=self.server_configs[0].model_name # This should be picked up automatically from server_configs
                # by the self.server object.
            )
        except tenacity.RetryError as retry_err:  # Specifically catch RetryError
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            print(f"[{self.name}] TENACITY RETRY ERROR during chat_completion call:")
            print(f"[{self.name}] RetryError Details: {retry_err}")
            # ... and the response details if available on 'e' ...
            original_exception = None
            if retry_err.last_attempt:
                if retry_err.last_attempt.failed:
                    original_exception = retry_err.last_attempt.exception()
                    print(
                        f"[{self.name}]   Last attempt failed. Original exception that caused retries:"
                    )
                    print(f"[{self.name}]     Type: {type(original_exception)}")
                    print(
                        f"[{self.name}]     Args: {original_exception.args if original_exception else 'N/A'}"
                    )
                    print(
                        f"[{self.name}]     Full Str: {str(original_exception)}"
                    )  # More direct string representation
                else:
                    # This case is unusual for a RetryError due to failure
                    print(
                        f"""[{self.name}]   Last attempt recorded but did not 'fail'.
                        Result: {retry_err.last_attempt.result()}"""
                    )
            else:
                print(
                    f"""[{self.name}]   Could not get 'last_attempt' details from
                    RetryError object. Raw RetryError: {retry_err}"""
                )

            # Now, if we have the original_exception, try to get more details (like HTTP response)
            if original_exception:
                # Check if the original exception is an OpenAI/HTTPX style error
                # by looking for a 'response' attribute.
                if (
                    hasattr(original_exception, "response")
                    and original_exception.response is not None
                ):
                    response_obj = original_exception.response
                    status_code_text = "Status code N/A"
                    response_content_text = "Response content N/A"

                    if hasattr(response_obj, "status_code"):
                        status_code_text = str(response_obj.status_code)

                    print(
                        f"[{self.name}]     Underlying API Response Status Code: {status_code_text}"
                    )

                    # Try to get JSON content first (common for API errors)
                    if hasattr(response_obj, "json") and callable(response_obj.json):
                        try:
                            response_json_parsed = (
                                response_obj.json()
                            )  # Note: this might need to be awaited if response_obj.json is async
                            # but typically in an exception, it's already processed.
                            print(
                                f"[{self.name}]     Underlying API Response JSON: {response_json_parsed}"
                            )
                        except Exception as json_e_inner:
                            print(
                                f"[{self.name}]     Could not parse underlying API response as JSON: {json_e_inner}"
                            )
                            # Fallback to text if JSON parsing fails
                            if hasattr(response_obj, "text"):
                                response_content_text = response_obj.text
                                print(
                                    f"[{self.name}]     Underlying API Response Text: {response_content_text}"
                                )
                            elif hasattr(response_obj, "content"):  # often bytes
                                try:
                                    response_content_text = (
                                        response_obj.content.decode()
                                    )
                                    print(
                                        f"""[{self.name}]     Underlying API Response
                                        Content (decoded): {response_content_text}"""
                                    )
                                except Exception:
                                    response_content_text = str(response_obj.content)
                                    print(
                                        f"""[{self.name}]     Underlying API Response Content
                                        (raw bytes as str): {response_content_text}"""
                                    )
                    # If no json() method, try .text or .content directly
                    elif hasattr(response_obj, "text"):
                        response_content_text = response_obj.text
                        print(
                            f"[{self.name}]     Underlying API Response Text: {response_content_text}"
                        )
                    elif hasattr(response_obj, "content"):
                        try:
                            response_content_text = response_obj.content.decode()
                            print(
                                f"[{self.name}]     Underlying API Response Content (decoded): {response_content_text}"
                            )
                        except Exception:
                            response_content_text = str(response_obj.content)
                            print(
                                f"""[{self.name}]     Underlying API Response Content
                                (raw bytes as str): {response_content_text}"""
                            )

            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            print(
                f"[{self.name}] Messages that were sent during the attempt resulting in RetryError: {messages}"
            )
            return None, []

        to_score_inputs = []
        for choice in chat_completions.choices:
            llm_response_content = choice.message.content
            # The 'messages' to store for scoring/tokenization should represent the full exchange
            # that led to this specific llm_response_content.
            # This includes the original system and user messages, and the assistant's response.
            full_exchange_messages = messages + [
                {"role": "assistant", "content": llm_response_content}
            ]
            to_score_inputs.append(
                {
                    "full_exchange_messages": full_exchange_messages,  # For tokenization
                    "llm_modified_html": llm_response_content,  # For direct scoring
                    "original_html_info": item,  # To know what to check against
                }
            )

        # The `score` method in Atropos expects a list where each element typically is
        # (messages_tuple_for_tokenization, original_item_metadata_for_scoring_logic)
        # We need to adapt `to_score_inputs` to what `self.score` will expect.
        # Let's define that `self.score` will take this list of dicts directly.
        # The `collect_trajectories` from the blog post returns `to_postprocess, to_backlog`
        # where `to_postprocess` is the output of `self.score`.

        scored_data_group = await self.score(to_score_inputs)
        return scored_data_group, []  # Assuming no backlog for now

    async def score(
        self, rollout_group_data: List[dict]
    ) -> Optional[ScoredDataGroup]:  # Return type is still ScoredDataGroup
        print(f"[{self.name}] Scoring {len(rollout_group_data)} rollouts...")

        all_tokens: List[List[int]] = []
        all_masks: List[List[int]] = []
        all_scores: List[float] = []
        # For TypedDict, optional fields that are not provided will simply not be keys in the dictionary.
        # However, if we want to include them as None, we can. Let's prepare for that.
        all_advantages: Optional[List[List[float]]] = (
            None  # Or initialize as [] if you might populate it
        )
        all_ref_logprobs: Optional[List[List[float]]] = None  # Or initialize as []
        all_messages_for_trainer: Optional[List[List[Dict]]] = (
            None  # Assuming Message is also a dict-like structure or TypedDict
        )

        for data_item in rollout_group_data:
            llm_html = data_item["llm_modified_html"]
            original_info = data_item["original_html_info"]

            current_score = -1.0
            if "<img" in original_info["html"] and "alt=" in llm_html:
                current_score = 1.0
            elif "<label>" in original_info["html"] and "for=" in llm_html:
                current_score = 1.0

            try:
                # Ensure self.tokenizer is initialized in __init__ or setup
                if not hasattr(self, "tokenizer") or self.tokenizer is None:
                    print(f"[{self.name}] Error: Tokenizer not initialized.")
                    # Attempt to initialize it here if it makes sense, or ensure it's done in setup()
                    # from transformers import AutoTokenizer
                    # self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name, trust_remote_code=True)
                    # This is a fallback, better to ensure it's in setup()
                    # For now, let's assume it's there. If not, this will fail earlier or be caught by linter.
                    pass  # Assuming tokenizer is initialized

                tokenized_output = tokenize_for_trainer(
                    self.tokenizer,  # Make sure self.tokenizer is loaded, e.g., in setup()
                    data_item["full_exchange_messages"],
                )
            except Exception as e:
                print(f"[{self.name}] Error during tokenization: {e}. Skipping item.")
                continue

            if "tokens" not in tokenized_output or "masks" not in tokenized_output:
                print(
                    f"[{self.name}] Warning: Tokenization did not return tokens/masks for an item. Skipping."
                )
                continue

            all_tokens.append(tokenized_output["tokens"])
            all_masks.append(tokenized_output["masks"])
            all_scores.append(current_score)

            # If you were to populate optional fields, you'd do it here. For example:
            # if "advantages" in tokenized_output: # Fictional example
            #     if all_advantages is None: all_advantages = []
            #     all_advantages.append(tokenized_output["advantages"])

        if not all_scores:
            print(f"[{self.name}] No valid items to score, returning None.")
            return None

        # print(f"[{self.name}] Scoring complete. Scores: {all_scores}") # Already printed if successful below

        # Construct the dictionary that conforms to ScoredDataGroup TypedDict
        # Mandatory fields:
        data_to_return: ScoredDataGroup = {
            "tokens": all_tokens,
            "masks": all_masks,
            "scores": all_scores,
            "advantages": all_advantages,
            "ref_logprobs": all_ref_logprobs,
            "group_overrides": {},
            "messages": all_messages_for_trainer,
            "overrides": None,
        }

        print(
            f"""[{self.name}] Scoring complete. Data to return (first score):
            {data_to_return['scores'][0] if data_to_return['scores'] else 'N/A'}"""
        )
        return data_to_return

    async def evaluate(
        self,
    ):  # Optional, might not be needed for hackathon 'process' focus
        print(f"[{self.name}] Evaluate method called (placeholder).")
        # Implement evaluation logic if you have a separate test set and metrics
        pass

    # --- Helper methods for scoring ---
    # def check_wcag_fixes(self, modified_html: str, original_item_info: dict) -> bool:
    #     # Placeholder for your actual WCAG checking logic
    #     # e.g., using BeautifulSoup to parse modified_html
    #     # and checking against `original_item_info['issues_to_fix']`
    #     # from bs4 import BeautifulSoup
    #     # soup = BeautifulSoup(modified_html, 'html.parser')
    #     # ... logic ...
    #     return False


if __name__ == "__main__":
    # This makes your environment runnable with `python accessibility_env.py process`
    AccessibilityEnv.cli()
