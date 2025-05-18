import json
import random
from typing import Dict, List, Optional, Tuple

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.type_definitions import Item
from atropos.environments.hack0.doctor_agent.patient import patient_profiles
from atropos.environments.hack0.doctor_agent.datasets import dataset
import re
from typing import Dict, List, Optional, Tuple
import os
from openai import OpenAI
import gymnasium as gym

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.type_definitions import Item

with open("environments/hack0/doctor_agent", "r") as f:
    keys = json.load(f)
    xai_key = keys["xai"]


client = OpenAI(
    api_key=xai_key,
    base_url="https://api.x.ai/v1",
)

final_message = "The diagnosis is:"
final_message_prompt = final_message + "<diagnosis>headache</diagnosis>"

doctor_system_prompt = """
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


doctor_model = "NousResearch/DeepHermes-3-Llama-3-8B-Preview"
gym_name = "gym_doctor"

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
            tokenizer_name=doctor_model,
            group_size=32,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            max_token_length=8192,
            wandb_name=gym_name,
        )
        server_configs = [
            APIServerConfig(
                model_name=doctor_model,
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

            patient_messages = []
            doctor_messages = [
                {
                    "role" : "system",
                    "content" : doctor_system_prompt
                }
            ]

            patient_profile = random.choice(patient_profiles)
            symptoms = dataset[0]
            patient_system_prompt = patient_profile.format(symptoms)

            patient_messages = [{"role": "system", "content": patient_system_prompt}]

            completion = client.chat.completions.create(
                model="grok-3-latest",
                messages=patient_messages,
            )

            patient_msg = completion.choices[0].message

            doctor_messages.append({"role": "user", "content": patient_msg})
            patient_messages.append({"role": "assistant", "content": patient_msg})

            score = -1
            while True:
                if (
                    len(self.tokenizer.apply_chat_template(doctor_messages))
                    > self.config.max_token_length - 10
                ):
                    score = 0
                    break
                max_tokens = self.config.max_token_length - len(
                    self.tokenizer.apply_chat_template(
                        doctor_messages, add_generation_prompt=True
                    )
                )
                doctor_completions = await server.chat_completion(
                    messages=doctor_messages,
                    n=1,
                    max_tokens=max_tokens,
                )

                doctor_msg = doctor_completions.choices[0].message.content

                doctor_messages.append({"role": "assistant", "content": doctor_msg})
                patient_messages.append({"role": "user", "content": doctor_msg})

                # check output
                if doctor_msg.startwith(final_message):
                    diagnosis = doctor_msg.strip(final_message)
                    diagnosis = diagnosis.strip()

                    if diagnosis == item["diagnosis"]:
                        score = 1
                    else:
                        score = 0
                    break


                completion = client.chat.completions.create(
                    model="grok-3-latest",
                    messages=patient_messages,
                )

                patient_msg = completion.choices[0].message

                doctor_messages.append({"role": "user", "content": patient_msg})
                patient_messages.append({"role": "assistant", "content": patient_msg})


            self.percent_correct_buffer.append(max(score, 0))
            tokens = self.tokenizer.apply_chat_template(doctor_messages)
            
            
            masks = []
            for i, msg in enumerate(doctor_messages):
                if i == len(doctor_messages) - 1:
                    masks.extend(tokens[len(masks) :])
                else:
                    curr_tokens = self.tokenizer.apply_chat_template(
                        doctor_messages[: i + 1],
                        add_generation_prompt=doctor_messages[i + 1]["role"] == "assistant",
                    )
                    if doctor_messages[i]["role"] == "user":
                        masks.extend([-100] * (len(curr_tokens) - len(masks)))
                    else:
                        masks.extend(curr_tokens[len(masks) :])
                        
        scored_data_item = ScoredDataItem(
            messages=doctor_messages,
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
