# kernelbench_env.py
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Union

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, APIServerConfig, ScoredDataGroup
from atroposlib.type_definitions import Item, number
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# KernelBench imports

from src.eval import eval_kernel_against_ref   # <- new import


KERNELBENCH_DIR = Path("/path/to/KernelBench")   # ← point here to your clone

class KBRow(TypedDict):
    """Single‑task record (prompt text plus meta)."""
    prompt: str           # full prompt given to the LLM
    sample_path: str


class KBEnv(BaseEnv):
    """
    A stripped‑down Atropos environment that only handles Level‑1 / problem‑1
    (square matrix multiplication).  It generates one kernel per rollout
    group, writes the kernel to the expected `runs/{run_name}` layout, then
    invokes KernelBench's evaluation script to obtain a scalar reward.
    """

    name = "kernelbench"

    # ---------- Static config helpers ----------------------------------------
    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_cfg = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            group_size=4,                     # 4 candidate kernels per step
            max_token_length=2048,
            batch_size=4,
            steps_per_eval=50,
            total_steps=1000,
            rollout_server_url="http://localhost:8000",
            use_wandb=False,                  # flip on if you want logging
            wandb_name="kb_level1_prob1",
        )

        server_cfgs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="DUMMY_KB_KEY",       # fill in if proxy requires it
                num_requests_for_eval=64,
            )
        ]
        return env_cfg, server_cfgs

    # --------------------- Data ------------------------------------------------
    async def setup(self):
        """
        Nothing to load from disk – we construct the single prompt on‑the‑fly
        with KernelBench's PromptConstructor so that it exactly matches their
        evaluation format.
        """
        # Hard‑code the HF dataset identifier for problem 1
        self.problem_spec = {
            "level": 1,
            "problem_id": 1,
            "problem_file": "1_Square_matrix_multiplication_.py",
        }
        self.iter = 0
        with open("prompt.txt", "r", encoding="utf-8") as f:
            self.prompt = f.read()

        self.sample_path="./sample.py"
        self.reward_buffer = list()

    # --------------------- Rollout / scoring ----------------------------------
    async def collect_trajectories(
        self, item: KBRow
    ) -> Tuple[ScoredDataGroup, List[Item]]:
        """
        Ask the LLM `group_size` times; each completion should be *only* the
        CUDA / Triton kernel (per KernelBench docs).  We store them to
        runs/{run_name}/{level}/{id}/sample_<n>.cu so that the official
        evaluator picks them up.
        """
        user_msg = {"role": "user", "content": self.prompt}

        chat_completions = await self.server.chat_completion(
            messages=[user_msg],
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
        )

        # Path: runs/<RUN_NAME>/level_1/1/
        run_dir = run_dir = KERNELBENCH_DIR / "runs"  / self.config.wandb_name / "level_1" / "1"
        run_dir.mkdir(parents=True, exist_ok=True)

        to_score: List[Dict] = []
        to_backlog: list()
        for i, choice in enumerate(chat_completions.choices):
            kernel_code = choice.message.content
            sample_path = run_dir / f"sample_{i}.cu"
            sample_path.write_text(kernel_code, encoding="utf‑8")

            messages = (user_msg, {"role": "assistant", "content": kernel_code})
            to_score.append(
                {
                    "messages": messages,
                    "sample_path": str(sample_path),
                    "finish_reason": choice.finish_reason,
                }
            )
        
        to_postprocess = await self.score(to_score)

        return to_postprocess, to_backlog
    
    async def score(
        self, rollout_group_data: List[Dict]
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:

        scores = ScoredDataGroup(tokens=[], masks=[], scores=[])

        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        # where we will build + compile kernels
        build_dir = os.path.join("build", "kernelbench", f"{1}", f"{1}")
        os.makedirs(build_dir, exist_ok=True)

        for item in rollout_group_data:
            generated_src = item["prompt"]
            custom_model_src = Path(item["sample_path"]).read_text()

            eval_result = eval_kernel_against_ref(
                ref_arch_src=generated_src,                      # blank per instructions
                custom_model_src=custom_model_src,
                measure_performance=True,
                verbose=True,
                num_correct_trials=1,
                num_perf_trials=1,
                build_dir=build_dir,
            )

            compiled_flag = bool(getattr(eval_result, "compiled", False))
            runtime_val   = float(getattr(eval_result, "runtime", -1.0))

            reward = 0.3 * (1 if compiled_flag else 0) + runtime_val

            out_dict = tokenize_for_trainer(self.tokenizer, item["messages"], item["finish_reason"])

            scores["tokens"].append(out_dict["tokens"])
            scores["masks"].append(out_dict["masks"])
            scores["scores"].append(reward)
        
        for score in scores["scores"]:
            self.reward_buffer.append(max(score, 0))

        return scores if scores["tokens"] else None
    
    async def get_next_item(self) -> KBRow:
        """Return the same single problem every time (env is tiny)."""
        return KBRow(prompt=self.prompt, sample_path=self.sample_path)

    async def evaluate(self, *args, **kwargs):
        """Evaluate the current model on a set of test problems."""
        # For now, we'll just log the average reward from the reward buffer
        if self.reward_buffer:
            avg_reward = sum(self.reward_buffer) / len(self.reward_buffer)
            self.eval_metrics.append(("eval/avg_reward", avg_reward))
            self.reward_buffer = list()

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Try to calculate percent_correct, pass if there's a division by zero
        try:
            wandb_metrics["train/reward"] = sum(
                self.reward_buffer
            ) / len(self.reward_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        self.reward_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    KBEnv.cli()

