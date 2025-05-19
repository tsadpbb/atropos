import os
import asyncio
from typing import List, Optional, Tuple

import wandb
from datasets import load_dataset

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class HumorEnvConfig(BaseEnvConfig):
    data_path: str = "environments/hack0/humor_dataset.jsonl"


class HumorEnv(BaseEnv):
    env_config_cls = HumorEnvConfig
    name = "humor"

    @classmethod
    def config_init(cls) -> Tuple[HumorEnvConfig, List[APIServerConfig]]:
        env_config = cls.env_config_cls(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=2,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=1024,
            steps_per_eval=100,
            max_token_length=2048,
            wandb_name="humor",
        )
        server_configs = [
            APIServerConfig(
                model_name="gpt-4o-mini",
                base_url=None,
                api_key=os.environ.get("OPENAI_API_KEY"),
                num_requests_for_eval=256,
            )
        ]
        return env_config, server_configs

    async def setup(self):
        ds = load_dataset("json", data_files=self.config.data_path, split="train")
        self.train = ds
        self.iter = 0

    async def get_next_item(self) -> Tuple[dict]:
        record = self.train[self.iter % len(self.train)]
        self.iter += 1
        return (record,)

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, List]:
        record = item[0]
        prompt = record["question"]
        chat_completions = await self.server.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
        )
        to_score = []
        for choice in chat_completions.choices:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": choice.message.content},
            ]
            to_score.append((tuple(messages), choice.finish_reason))
        scored = await self.score(to_score)
        return scored, []

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        scores = ScoredDataGroup(tokens=[], masks=[], scores=[])
        for (messages, _), idx in zip(rollout_group_data, range(len(rollout_group_data))):
            expected = self.train[idx % len(self.train)]["response"].strip()
            output = messages[-1]["content"].strip()
            score_val = 1.0 if output == expected else 0.0
            out = tokenize_for_trainer(self.tokenizer, list(messages))
            scores["tokens"].append(out["tokens"])
            scores["masks"].append(out["masks"])
            scores["scores"].append(score_val)
        return scores

    async def wandb_log(self, wandb_metrics: Optional[dict] = None):
        await super().wandb_log(wandb_metrics)

    async def evaluate(self, *args, **kwargs):
        # No-op evaluation; required by BaseEnv abstract interface
        return None


if __name__ == "__main__":
    import sys
    # default to 'serve' if no subcommand provided
    if len(sys.argv) == 1:
        sys.argv.append("serve")
    HumorEnv.cli()
