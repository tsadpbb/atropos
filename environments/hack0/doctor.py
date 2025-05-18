import re
from typing import Dict, List, Optional, Tuple

import gymnasium as gym

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.type_definitions import Item

system_prompt = """
You are a doctor. You are interacting with a patient.
You need to diagnose the patient based on the symptoms.
You will need to ask the patient follow up questions to diagnose them.
Once you are confident in your diagnosis, provide it in the format:

The patient is diagnosed with <diagnosis>{possible_illness}.</diagnosis>

For example,

user: I have a headache.
assistant: What is the severity of your headache?
user: It's a 3/10.
assistant: What is the location of your headache?
user: It's in the front of my head.
assistant: What is the duration of your headache?
user: It's been going on for 2 days.
assistant: The patient is diagnosed with <diagnosis>headache</diagnosis>
"""


class DoctorEnv(BaseEnv):

    name = "doctor"

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
        self.print_this_env = False

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=32,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            max_token_length=8192,
            wandb_name="gym_taxi",
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
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
        self.iter = 0

    async def evaluate(self, *args, **kwargs):
        pass

    async def get_patient_msg(self, env: gym.Env) -> str:
        # Call xAI to get a patient message
        return env.render()

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        # Grab a dedicated llm server to take advantage of caching
        async with self.server.dedicated_server() as server:
            init_msg = f"{system_prompt}\n\n"
            messages = [{"role": "system", "content": init_msg}]
            patient_msg = await self.get_patient_msg(item)
            messages.append({"role": "user", "content": patient_msg})
            score = -1
            while True:
                if (
                    len(self.tokenizer.apply_chat_template(messages))
                    > self.config.max_token_length - 10
                ):
                    score = 0
                    break
                max_tokens = self.config.max_token_length - len(
                    self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True
                    )
                )
                chat_completions = await server.chat_completion(
                    messages=messages,
                    n=1,
                    max_tokens=max_tokens,
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": chat_completions.choices[0].message.content,
                    }
                )
                diagnosis_match = re.search(
                    r"<diagnosis>(.*?)</diagnosis>",
                    chat_completions.choices[0].message.content,
                    re.DOTALL,
                )
                if diagnosis_match:
                    diagnosis = diagnosis_match.group(1).strip()
                    # Check if the diagnosis is correct
                    if diagnosis == item["diagnosis"]:
                        score = 1
                    else:
                        score = 0
                    break

                next_patient_msg = await self.get_patient_msg(item)
                messages.append(
                    {
                        "role": "user",
                        "content": next_patient_msg,
                    }
                )
            self.percent_correct_buffer.append(max(score, 0))
            tokens = self.tokenizer.apply_chat_template(messages)
            masks = []
            for i, msg in enumerate(messages):
                if i == len(messages) - 1:
                    masks.extend(tokens[len(masks) :])
                else:
                    curr_tokens = self.tokenizer.apply_chat_template(
                        messages[: i + 1],
                        add_generation_prompt=messages[i + 1]["role"] == "assistant",
                    )
                    if messages[i]["role"] == "user":
                        masks.extend([-100] * (len(curr_tokens) - len(masks)))
                    else:
                        masks.extend(curr_tokens[len(masks) :])
        scored_data_item = ScoredDataItem(
            messages=messages,
            finish_reason=score,
            tokens=tokens,
            masks=masks,
            scores=score,
        )
        return scored_data_item, []

    async def get_next_item(self):
        next_item = {"seed": self.iter}
        self.iter += 1
        return next_item


# if __name__ == "__main__":
#     GymTaxiEnv.cli()
