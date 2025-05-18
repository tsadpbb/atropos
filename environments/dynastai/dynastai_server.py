import random
import re
import json
import os
from typing import Dict, List, Optional, Tuple, TypedDict, Union

from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item, number
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
)

system_prompt += """You are allocated a maximum of 2048 tokens, please strive to use less.

You are playing a game called DynastAI where you generate scenarios for a kingdom management game.
Each scenario should include a character presenting a dilemma to the ruler, with two choices that affect 
the four key resources of the kingdom: Piety, Stability, Power, and Wealth.

Your response must be a valid JSON object with the following structure:
{
  "Character": "Name/Title of the character",
  "Prompt": "The scenario description",
  "Left_Choice": "The first choice option",
  "Left_Piety": integer value between -30 and 30,
  "Left_Stability": integer value between -30 and 30,
  "Left_Power": integer value between -30 and 30,
  "Left_Wealth": integer value between -30 and 30,
  "Right_Choice": "The second choice option",
  "Right_Piety": integer value between -30 and 30,
  "Right_Stability": integer value between -30 and 30,
  "Right_Power": integer value between -30 and 30,
  "Right_Wealth": integer value between -30 and 30
}

Be creative and make each scenario interesting!"""


class DynastAIRow(TypedDict):
    scenario_prompt: str
    card: Optional[Dict] = None


class DynastAIEnv(BaseEnv):

    name = "dynastai"

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
        # Add tracking for wandb visualizations
        self.rollouts_for_wandb = []
        self.completion_lengths = []

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="Qwen/Qwen3-1.7B",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=2048,
            wandb_name="dynastai",
        )
        server_configs = [
            APIServerConfig(
                model_name="Qwen/Qwen3-1.7B",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=256,
            ),
        ]

        return env_config, server_configs

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
        
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        # Load cards from the JSON file
        cards_file = os.path.join(os.path.dirname(__file__), "dynastai_cards.json")
        with open(cards_file, "r") as f:
            cards = json.load(f)
        
        # Shuffle and split into train/test
        random.shuffle(cards)
        test_size = int(len(cards) * 0.1)  # 10% for test set
        
        self.train = cards[test_size:]
        self.test = cards[:test_size]
        
        # Keep scenario prompts for generating new scenarios
        self.scenario_prompts = [
            "Create a dilemma involving the Church and Treasury",
            "Create a dilemma involving the Military and People",
            "Create a scenario where a foreign diplomat visits",
            "Create a scenario about a natural disaster",
            "Create a scenario about a rebellious noble",
            "Create a scenario about a religious conflict",
            "Create a scenario about a military campaign",
            "Create a scenario about a royal marriage proposal",
            "Create a scenario about a trade agreement",
            "Create a scenario about a mysterious artifact",
            "Create a scenario about peasant unrest",
            "Create a scenario about a spy in the court",
            "Create a scenario about a disputed succession",
            "Create a scenario about a diplomatic incident",
            "Create a scenario about a technological innovation",
        ]
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def evaluate(self, *args, **kwargs):
        # For evaluation, we'll use the test set cards
        eval_tasks = []
        for card in self.test:
            eval_tasks.append(self.rollout_and_score_eval(f"Create a scenario similar to: {card['Prompt']}"))
        
        scores = await tqdm_asyncio.gather(*eval_tasks)
        self.eval_metrics.append(("eval/percent_correct", sum(scores) / len(scores)))

    async def rollout_and_score_eval(self, scenario_prompt: str) -> number:
        completion = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": scenario_prompt},
            ],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )
        
        completion_content = completion.choices[0].message.content
        score = self.validate_json_structure(completion_content)
        return score

    def validate_json_structure(self, content: str) -> number:
        # Extract content after </think> tag if present
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        
        # Find JSON structure
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            return 0
        
        json_str = json_match.group(0)
        
        try:
            # Attempt to parse as JSON
            data = json.loads(json_str)
            
            # Check for required fields
            required_fields = [
                "Character", "Prompt", 
                "Left_Choice", "Left_Piety", "Left_Stability", "Left_Power", "Left_Wealth",
                "Right_Choice", "Right_Piety", "Right_Stability", "Right_Power", "Right_Wealth"
            ]
            
            if not all(field in data for field in required_fields):
                return 0
            
            # Check numeric fields
            numeric_fields = [
                "Left_Piety", "Left_Stability", "Left_Power", "Left_Wealth",
                "Right_Piety", "Right_Stability", "Right_Power", "Right_Wealth"
            ]
            
            for field in numeric_fields:
                if not isinstance(data[field], int):
                    return 0
                if data[field] < -30 or data[field] > 30:
                    return 0
            
            # If we made it here, the JSON is valid
            return 1
            
        except json.JSONDecodeError:
            return 0

    async def collect_trajectories(
        self, item: DynastAIRow
    ) -> Tuple[ScoredDataGroup, list[Item]]:
        user_message = {"role": "user", "content": item["scenario_prompt"]}

        chat_completions = await self.server.chat_completion(
            messages=[{"role": "system", "content": system_prompt}, user_message],
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
        )
        
        to_score = list()
        to_backlog = list()
        
        for i, chat_completion in enumerate(chat_completions.choices):
            messages = (
                {"role": "system", "content": system_prompt},
                user_message,
                {"role": "assistant", "content": chat_completion.message.content},
            )
            to_score.append({
                "messages": messages,
                "finish_reason": chat_completion.finish_reason,
            })
            
        to_postprocess = await self.score(to_score)
        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        random.shuffle(rollout_group_data)
        for item in rollout_group_data:
            completion_content = item["messages"][-1]["content"]
            reward = self.validate_json_structure(completion_content)
            
            out_dict = tokenize_for_trainer(
                self.tokenizer, item["messages"], item["finish_reason"]
            )
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]
            
            # Remove obviously bad examples
            if len([1 for i in masks if i != -100]) < 10:
                continue
                
            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(1.0 if reward else -1.0)
            
            if len(scores["tokens"]) >= self.config.group_size:
                break
                
        for score in scores["scores"]:
            self.percent_correct_buffer.append(max(score, 0))
            
        # Check if all the same
        if all([score == scores["scores"][0] for score in scores["scores"]]):
            return None  # If all the same, we return None
            
        return scores

    async def get_next_item(self) -> DynastAIRow:
        # Alternate between using saved cards and generating new scenarios
        if self.iter % 2 == 0 and self.train:
            # Use a card from the training set
            card_index = (self.iter // 2) % len(self.train)
            card = self.train[card_index]
            prompt = f"Create a scenario similar to: {card['Prompt']}"
            self.iter += 1
            return {"scenario_prompt": prompt, "card": card}
        else:
            # Generate a completely new scenario
            prompt = random.choice(self.scenario_prompts)
            self.iter += 1
            return {"scenario_prompt": prompt}


if __name__ == "__main__":
    DynastAIEnv.cli()
