# environments/hack0/accessibility_env/accessibility_env.py
import os  # For API keys, etc.
from typing import List, Optional, Tuple  # Common type hints, added Dict

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
            tokenizer_name="NousResearch/Llama-3-8B-Instruct- যেভাবে-তুমি-বাংলা-বলো",  # Placeholder
            group_size=2,  # Smaller for faster testing initially
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=10,  # For process mode, number of items to generate
            batch_size=4,  # Max items in a single call to score (related to group_size)
            steps_per_eval=5,
            max_token_length=2048,
            wandb_name="accessibility_env_hackathon_dev",  # Dev run name
        )
        server_configs = [
            APIServerConfig(
                model_name="gpt-3.5-turbo",  # Or your preferred model
                # base_url=None, # Defaults to OpenAI if None
                api_key=os.environ.get(
                    "OPENAI_API_KEY", "YOUR_API_KEY_PLACEHOLDER_IF_NOT_SET"
                ),  # Important!
                num_requests_for_eval=16,
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        print(f"[{self.name}] Setting up environment...")
        # Load dataset, initialize tools (e.g., HTML parser) here
        self.dataset = []  # Placeholder for your HTML snippets
        self.iter = 0
        print(f"[{self.name}] Setup complete.")

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
        self, item_data: Item
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        # 'item_data' here is what get_next_item returned.
        original_html = item_data["html"]
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

        chat_completions = await self.server.chat_completion(
            messages=messages,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
            # temperature=0.7, # Optional: adjust for creativity vs. determinism
        )

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
                    "original_html_info": item_data,  # To know what to check against
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

    async def score(self, rollout_group_data: List[dict]) -> Optional[ScoredDataGroup]:
        # rollout_group_data is a list of dicts, each like:
        # {
        #     "full_exchange_messages": [...],
        #     "llm_modified_html": "...",
        #     "original_html_info": {"html": "...", "id": "...", "issues": [...]}
        # }
        print(f"[{self.name}] Scoring {len(rollout_group_data)} rollouts...")
        scores_obj = ScoredDataGroup()  # Use the Atropos defined type
        # Initialize lists within scores_obj as per ScoredDataGroup structure
        # (typically 'tokens', 'masks', 'scores', maybe 'logprobs')
        scores_obj["tokens"] = []
        scores_obj["masks"] = []
        scores_obj["scores"] = []
        # scores_obj["infos"] = [] # Optional for extra debug info

        for data_item in rollout_group_data:
            llm_html = data_item["llm_modified_html"]
            original_info = data_item["original_html_info"]

            # Basic reward: 1.0 if fixed, -1.0 if not.
            # This will be replaced with actual WCAG checks.
            current_score = -1.0  # Default to failure
            # ---- YOUR SCORING LOGIC HERE ----
            # Example: (pseudo-code, requires BeautifulSoup and specific checks)
            # violations_fixed = self.check_wcag_fixes(llm_html, original_info)
            # if violations_fixed:
            #    current_score = 1.0
            # For now, a placeholder:
            if "<img" in original_info["html"] and "alt=" in llm_html:
                current_score = 1.0
            elif "<label>" in original_info["html"] and "for=" in llm_html:
                current_score = 1.0

            # Tokenize the full exchange for the trainer
            # The 'tokenize_for_trainer' util expects a tuple/list of message dicts
            tokenized_output = tokenize_for_trainer(
                self.tokenizer,
                data_item["full_exchange_messages"],  # Pass the list of message dicts
            )

            # Ensure tokenized_output contains 'tokens' and 'masks'
            if "tokens" not in tokenized_output or "masks" not in tokenized_output:
                print(
                    f"[{self.name}] Warning: Tokenization did not return tokens/masks for an item. Skipping."
                )
                continue

            scores_obj["tokens"].append(tokenized_output["tokens"])
            scores_obj["masks"].append(tokenized_output["masks"])
            scores_obj["scores"].append(current_score)
            # scores_obj["infos"].append({"original_id": original_info["id"], "llm_output_preview": llm_html[:100]})

        # Handle case where no valid items were scored
        if not scores_obj["scores"]:
            print(f"[{self.name}] No valid items to score, returning None.")
            return None

        # Atropos convention: if all scores are identical, return None (no learning signal)
        # This might be too strict for early testing. Consider enabling later.
        # if len(set(scores_obj["scores"])) == 1 and len(scores_obj["scores"]) > 1 :
        #     print(f"[{self.name}] All scores are identical ({scores_obj['scores'][0]}), returning None.")
        #     return None

        print(f"[{self.name}] Scoring complete. Scores: {scores_obj['scores']}")
        return scores_obj

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
