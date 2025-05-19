# In your meteorology_forecast_env.py file:

import asyncio
import time
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import traceback # Import traceback for more detailed error logging

import wandb
from pydantic import Field
import httpx

# Assuming APIServer and ServerManager are imported correctly from atroposlib
# For this standalone example, let's define dummy classes if not available
try:
    from atroposlib.envs.base import (
        APIServerConfig,
        BaseEnv,
        BaseEnvConfig,
        EvalHandlingEnum,
        Item,
        ScoredDataGroup,
        APIServer
    )
    from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
except ImportError:
    # Dummy classes (keep these if you were using them for standalone testing)
    class BaseEnvConfig: pass
    class APIServerConfig:
        def __init__(self, model_name, base_url, api_key, timeout=1200, num_max_requests_at_once=512, num_requests_for_eval=64, rolling_buffer_length=1000, server_type='openai', n_kwarg_is_ignored=False, health_check=True):
            self.model_name = model_name
            self.base_url = base_url
            self.api_key = api_key
            self.timeout = timeout # etc.
    class BaseEnv:
        def __init__(self, config, server_configs, slurm, testing):
            self.config = config
            self.server = type('ServerManager', (), {'servers': [APIServer(sc) for sc in server_configs]})()
            self.tokenizer = type('Tokenizer', (), {'apply_chat_template': lambda *args, **kwargs: ""})()
            self.testing = testing
        def save_checkpoint(self, step, data): pass
        async def wandb_log(self, metrics): pass
        @classmethod
        def cli(cls): print("Dummy CLI called") # Add dummy cli
    class Item: pass
    class ScoredDataGroup(dict): pass
    class EvalHandlingEnum: LIMIT_TRAIN = "limit_train"; STOP_TRAIN = "stop_train" # Added STOP_TRAIN
    class APIServer:
        def __init__(self, config: APIServerConfig): self.config = config # Ensure config is taken
        async def chat_completion(self, **kwargs):
            print(f"Dummy APIServer chat_completion called with model: {kwargs.get('model', self.config.model_name)}")
            # Simulate a response structure
            class DummyMessage:
                def __init__(self, content): self.content = content
            class DummyChoice:
                def __init__(self, content): self.message = DummyMessage(content)
            class DummyCompletionResponse:
                def __init__(self, choices_content_list):
                    self.choices = [DummyChoice(c) for c in choices_content_list]
            if self.config.model_name.startswith("google"): # Simulate judge
                return DummyCompletionResponse(["REASONING_SCORE: 3\nTOOL_CALL_SCORE: 1\nFORECAST_SUMMARY_SCORE: 1\nTOTAL_SCORE: 5\nJUSTIFICATION: Dummy judge output"])
            else: # Simulate agent
                return DummyCompletionResponse(["<think>Dummy agent thinking</think>\nFORECAST_SUMMARY: Dummy forecast"])
        async def completion(self, **kwargs): return type('CompletionResponse', (), {'choices': []})()

    def tokenize_for_trainer(tokenizer, messages, max_length): return {"tokens": [1,2,3], "masks": [1,1,1]} # Ensure it returns non-empty
    print("Warning: atroposlib not fully found, using dummy classes for some components.")


# --- Setup Module-Level Logger (consistent with BaseEnv) ---
logger = logging.getLogger(__name__)
# Ensure basicConfig is called if not configured elsewhere, e.g., by atroposlib
if not logging.getLogger().hasHandlers(): # Check if root logger has handlers
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Configuration (MetRLConfig) remains the same ---
class MetRLConfig(BaseEnvConfig):
    tokenizer_name: str = Field(default="Qwen/Qwen3-8B")
    group_size: int = Field(default=2)
    use_wandb: bool = Field(default=True)
    max_num_workers: int = Field(default=64)
    rollout_server_url: str = Field(default="http://localhost:8000")
    total_steps: int = Field(default=2000)
    batch_size: int = Field(default=-1) # Default 
    steps_per_eval: int = Field(default=100) # Default f
    max_token_length: int = Field(default=2048) # Default 
    inference_weight: float = Field(default=1.0)
    wandb_name: Optional[str] = Field(default=None)
    data_path_to_save_groups: Optional[str] = Field(default='data/MeteorologyForecastRL.jsonl') # Default 
    eval_handling: EvalHandlingEnum = Field(default=EvalHandlingEnum.STOP_TRAIN) # Default 
    eval_limit_ratio: float = Field(default=0.5) # Default 
    num_eval_samples: int = Field(default=20)
    num_rollouts_to_log: int = Field(default=10)
    min_items_sent_before_logging: int = Field(default=2) # Default 
    include_messages: bool = Field(default=True) # Default 
    num_rollouts_to_keep: int = Field(default=32) # Default 
    num_rollouts_per_group_for_logging: int = Field(default=1) # Default 
    ensure_scores_are_not_same: bool = Field(default=False) # Default 
    max_eval_workers: int = Field(default=16) # Default 
    max_num_workers_per_node: int = Field(default=8) # Default 
    max_batches_offpolicy: int = Field(default=3) # Default 


    sounding_data_root: str = Field(
        default="/Users/dev/hackathon/atropos/environments/hack0/data/",
        description="Root directory for all sounding and AFD data."
    )
    target_date: str = Field(
        default="20250314",
        description="The specific date to load data for (YYYYMMDD format)."
    )
    judge_model_name: str = Field(
        default="google/gemini-2.5-flash-preview",
        description="Identifier for the Judge model on OpenRouter."
    )
    judge_api_key_env_var: str = Field(
        default="OPENROUTER_API_KEY",
        description="Environment variable name for OpenRouter API key for the Judge."
    )
    judge_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for the OpenRouter API (for Judge)."
    )
    nwp_models_to_use: List[str] = Field(
        default=["RAP"],
        description="List of NWP models to use (e.g., RAP, HRRR)."
    )
    forecast_hours_to_sample: List[int] = Field(
        default=[6, 9, 12, 15, 18],
        description="Which forecast hours (UTC) from the model run to provide to the LLM."
    )
    target_forecast_hour_offset: int = Field(
        default=1,
        description="Offset from the latest provided sounding hour to set the target forecast time."
    )
    max_afds_for_judge: int = Field(
        default=3,
        description="Maximum number of AFD files to provide to the judge model."
    )
    max_reasoning_tokens_llm: int = Field(
        default=3000,
        description="Max tokens for the agent LLM's generation."
    )
    max_tokens_judge: int = Field(
        default=2000,
        description="Max tokens for the judge model's generation."
    )

# --- Prompts (AGENT_SYSTEM_PROMPT, AGENT_USER_PROMPT_TEMPLATE, etc.) remain the same ---
AGENT_SYSTEM_PROMPT = """You are a highly skilled AI meteorologist. Your task is to analyze numerical weather prediction (NWP) model sounding data for a specific location and time period.
Based on your analysis, you must:
1.  Provide a detailed step-by-step reasoning process. This should include identifying trends, interpreting meteorological parameters, and connecting them to potential weather phenomena.
2.  If you determine that additional real-time observational data is crucial for a more accurate assessment, specify the tools you would use. For each tool, output a line in the exact format: TOOL_CALL: {{"tool_name": "tool_name_here", "arguments": {{"param1": "value1", ...}}}}
    Available conceptual tools: get_surface_observations, get_latest_radar_imagery, get_satellite_imagery, get_upper_air_sounding.
3.  Conclude with a concise forecast summary for the specified target time. Start this summary with "FORECAST_SUMMARY: ".

Analyze the provided data thoroughly. Your reasoning should be comprehensive."""

AGENT_USER_PROMPT_TEMPLATE = """Please analyze the following NWP model sounding data for station {location_id}.
The soundings provided are from the {model_name} model, run on {run_date_full_z}, valid at the following UTC times: {sounding_times_str}.
Your goal is to make a preliminary forecast assessment focusing on severe weather potential for {location_id} around {target_forecast_time_utc}.

Sounding Data:
{soundings_json_blob}

Remember to include your reasoning, any TOOL_CALL: {{"tool_name": "tool_name_here", "arguments": {{"param1": "value1", ...}}}} lines, and a final FORECAST_SUMMARY: statement."""

JUDGE_SYSTEM_PROMPT = """You are an expert meteorologist acting as a judge. You will evaluate an AI assistant's analysis of model sounding data.
The AI was asked to provide reasoning, call tools if necessary, and give a forecast summary.
You will be given the AI's output and relevant Area Forecast Discussions (AFDs) from human forecasters for context.

Your evaluation should focus on:
1.  **Meteorological Soundness of Reasoning (0-5 points):**
    *   Correct interpretation of sounding parameters and trends.
    *   Logical connections between data and potential weather.
    *   Avoidance of meteorological fallacies or hallucinations.
    *   Depth and detail of the thought process.
2.  **Tool Call Relevance & Justification (0-3 points):**
    *   Were the tools called (if any) appropriate given the AI's reasoning and the model data?
    *   Would these tools genuinely help a meteorologist in this situation?
    *   Were critical tool calls missed?
3.  **Forecast Summary Quality (0-2 points):**
    *   Clarity and conciseness.
    *   Alignment with the AI's own reasoning and the provided AFDs (or sensible deviation if model data strongly suggested it).

Provide a numerical score for each category and a total score (sum of the three, max 10.0). Also, provide a brief overall justification for your scores.
Your output MUST be in the following exact format:
REASONING_SCORE: {{{{0-5 score}}}}
TOOL_CALL_SCORE: {{{{0-3 score}}}}
FORECAST_SUMMARY_SCORE: {{{{0-2 score}}}}
TOTAL_SCORE: {{{{sum of scores, e.g., 7.5}}}}
JUSTIFICATION: {{{{Your brief textual justification here.}}}}"""

JUDGE_USER_PROMPT_TEMPLATE = """AI Assistant's Output:
---
{llm_full_output}
---

Contextual Area Forecast Discussions (AFDs):
---
{afds_blob}
---

Please evaluate the AI assistant's output based on the criteria and provide your scores and justification in the specified format."""


class MeteorologyForecastRLEnv(BaseEnv):
    env_config_cls = MetRLConfig
    name = "MeteorologyForecastRL"

    def __init__(
        self,
        config: MetRLConfig,
        server_configs: List[APIServerConfig],
        slurm=True, # Default based on your CLI help
        testing=False, # Default based on your CLI help
    ):
        super().__init__(config, server_configs, slurm, testing)

        # self.config: MetRLConfig = self.config # This is redundant if super().__init__ sets self.config
        # Ensure self.config is correctly typed if BaseEnv makes it generic
        if not isinstance(self.config, MetRLConfig): # Check type
            logger.warning(f"self.config in __init__ is not MetRLConfig, type: {type(self.config)}. This might indicate an issue with BaseEnv or pydantic_cli setup.")

        self.locations_data: List[Dict[str, Any]] = []
        self.agent_llm_server: Optional[APIServer] = None
        self.judge_server: Optional[APIServer] = None

        if not hasattr(self.server, 'servers') or not self.server.servers: # Added check for empty servers list
            logger.error("CRITICAL: ServerManager (self.server) does not have a 'servers' attribute or it's empty!")
        elif len(self.server.servers) > 1:
            self.agent_llm_server = self.server.servers[0]
            self.judge_server = self.server.servers[1]
            if self.agent_llm_server and hasattr(self.agent_llm_server, 'config') and self.agent_llm_server.config: # check config exists
                logger.info(f"Agent server: {self.agent_llm_server.config.model_name} @ {self.agent_llm_server.config.base_url}")
            else:
                logger.warning("Agent LLM server or its config is not properly initialized for logging.")

            if self.judge_server and hasattr(self.judge_server, 'config') and self.judge_server.config: # check config exists
                logger.info(f"Judge server: {self.judge_server.config.model_name} @ {self.judge_server.config.base_url}")
            else:
                logger.warning("Judge server or its config is not properly initialized for logging.")

        elif len(self.server.servers) == 1:
            logger.warning(
                "Only 1 API server configured in ServerManager. Agent and Judge will use the same server."
            )
            self.agent_llm_server = self.server.servers[0]
            self.judge_server = self.server.servers[0]
            if self.agent_llm_server and hasattr(self.agent_llm_server, 'config') and self.agent_llm_server.config: # check config exists
                logger.info(f"Agent/Judge server: {self.agent_llm_server.config.model_name} @ {self.agent_llm_server.config.base_url}")
            else:
                logger.warning("Agent/Judge server or its config is not properly initialized for logging.")

        self.current_item_index: int = 0
        self.iter: int = 0
        self.judge_total_scores_buffer: List[float] = []
        self.judge_reasoning_scores_buffer: List[float] = []
        self.judge_tool_scores_buffer: List[float] = []
        self.judge_forecast_scores_buffer: List[float] = []
        self.rollouts_for_wandb_custom: List[Tuple[str, str, str, str, float, str]] = []
        self.eval_metrics_buffer: List[Dict[str, float]] = []


    @classmethod
    def config_init(cls) -> Tuple[MetRLConfig, List[APIServerConfig]]:
        # This method is usually called by pydantic_cli.
        # The CLI arguments will override these defaults.
        env_config = MetRLConfig() # Initialize with defaults, CLI will override

        # Get API keys and base URLs from environment or use defaults from MetRLConfig
        # Agent server config (server_configs[0])
        # These might be overridden by CLI --openai.model_name, --openai.base_url etc. for the *first* server
        # However, atroposlib's pydantic_cli might handle multiple server configs differently.
        # The provided help suggests a single --openai.* block, which implies it might apply to all servers
        # or only the first. We'll assume here it populates the first server, and the second uses MetRLConfig defaults.

        agent_model_name = os.environ.get("AGENT_LLM_MODEL_NAME", env_config.tokenizer_name) # Prioritize env var
        agent_api_key = os.environ.get("AGENT_LLM_API_KEY", "EMPTY_KEY_IF_LOCAL_VLLM")
        agent_base_url = os.environ.get("AGENT_LLM_BASE_URL", "http://localhost:8080/v1") # Example vLLM

        judge_api_key = os.environ.get(env_config.judge_api_key_env_var)
        if not judge_api_key:
            # This print is fine as it's at class method execution, not instance init
            # logger is not available at class level directly here, so print is okay or use logging.getLogger
            logging.warning(f"Environment variable {env_config.judge_api_key_env_var} not set for Judge API.")

        server_configs = [
            APIServerConfig(
                model_name=agent_model_name, # This should be the agent model
                base_url=agent_base_url,
                api_key=agent_api_key,
                # num_requests_for_eval=64, # these are often part of APIServer not its config
            ),
            APIServerConfig(
                model_name=env_config.judge_model_name,
                base_url=env_config.judge_base_url,
                api_key=judge_api_key,
                # num_requests_for_eval=64,
            )
        ]
        # logger.info(f"config_init: env_config={env_config}") # For debugging
        # logger.info(f"config_init: server_configs={server_configs}") # For debugging
        return env_config, server_configs

    # --- setup, get_next_item, _parse_llm_output, _parse_judge_output remain the same ---
    async def setup(self):
        logger.info(f"Setting up {self.name or self.__class__.__name__}...")
        data_root = Path(self.config.sounding_data_root)
        date_path = data_root / self.config.target_date

        if not date_path.is_dir():
            logger.error(f"Target date directory not found: {date_path}")
            return

        available_locations = [loc.name for loc in date_path.iterdir() if loc.is_dir()]
        logger.info(f"Found {len(available_locations)} locations for date {self.config.target_date}: {available_locations}")

        for loc_id in available_locations:
            loc_path = date_path / loc_id
            soundings_for_item = []
            sounding_times_for_item = []

            if not self.config.nwp_models_to_use or not self.config.forecast_hours_to_sample:
                logger.warning(f"NWP models or forecast hours to sample is empty in config. Skipping {loc_id}")
                continue
            selected_model = self.config.nwp_models_to_use[0]

            for hour_z in self.config.forecast_hours_to_sample:
                fname = f"{loc_id}_{selected_model}_{self.config.target_date}{hour_z:02d}Z.buf_default_llm_optimized.jsonl"
                sounding_file_path = loc_path / fname
                if sounding_file_path.exists():
                    try:
                        with open(sounding_file_path, 'r') as f:
                            line = f.readline()
                            if line:
                                soundings_for_item.append(json.loads(line))
                                sounding_times_for_item.append(f"{hour_z:02d}00Z")
                    except Exception as e:
                        logger.warning(f"Could not load or parse sounding file {sounding_file_path}: {e}")
                else:
                    logger.debug(f"Sounding file not found: {sounding_file_path}")

            if not soundings_for_item:
                logger.debug(f"No valid soundings found for {loc_id} on {self.config.target_date}. Skipping.")
                continue

            afd_files = sorted([f for f in loc_path.glob("AFD_*.txt")])
            selected_afd_texts = []
            if afd_files:
                if len(afd_files) <= self.config.max_afds_for_judge:
                    indices_to_take = list(range(len(afd_files)))
                else:
                    indices_to_take = sorted(list(set([0, len(afd_files) // 2, len(afd_files) - 1])))
                    indices_to_take = indices_to_take[:self.config.max_afds_for_judge]

                for i in indices_to_take:
                    try:
                        with open(afd_files[i], 'r', encoding='utf-8', errors='replace') as f: # Specify encoding and error handling
                            afd_text = f.read()
                            # Remove common control characters, especially ETX (\x03 or \u0003)
                            cleaned_afd_text = ''.join(c for c in afd_text if c.isprintable() or c.isspace())
                            # Or more specifically for \u0003:
                            # cleaned_afd_text = afd_text.replace('\u0003', '')
                            selected_afd_texts.append(cleaned_afd_text)
                    except Exception as e:
                        logger.warning(f"Could not read or clean AFD file {afd_files[i]}: {e}")


            if not sounding_times_for_item:
                logger.warning(f"No sounding times available for {loc_id}, cannot determine target forecast time. Skipping.")
                continue

            latest_sounding_hour_str = sounding_times_for_item[-1][:2]
            if not latest_sounding_hour_str.isdigit():
                logger.warning(f"Could not parse latest sounding hour from {sounding_times_for_item[-1]} for {loc_id}. Skipping.")
                continue
            latest_sounding_hour = int(latest_sounding_hour_str)

            target_hour = latest_sounding_hour + self.config.target_forecast_hour_offset
            target_forecast_time_utc = f"{target_hour:02d}00Z on {self.config.target_date[4:6]}/{self.config.target_date[6:8]}/{self.config.target_date[0:4]}"

            run_time_str = "UnknownRunTime"
            if soundings_for_item and 'tm' in soundings_for_item[0] and '/' in soundings_for_item[0]['tm']:
                try:
                    run_time_str = soundings_for_item[0]['tm'].split('/')[1][:2] + "Z"
                except IndexError:
                     logger.warning(f"Could not parse run time from 'tm' field: {soundings_for_item[0]['tm']} for {loc_id}")
            run_date_full_z = f"{self.config.target_date} at {run_time_str}"


            item_data = {
                "case_id": f"{self.config.target_date}_{loc_id}",
                "location_id": loc_id,
                "model_name": selected_model,
                "run_date_full_z": run_date_full_z,
                "target_forecast_time_utc": target_forecast_time_utc,
                "model_soundings_data": soundings_for_item,
                "sounding_times_str": ", ".join(sounding_times_for_item),
                "afd_texts": selected_afd_texts
            }
            self.locations_data.append(item_data)

        if not self.locations_data:
            logger.error("No data loaded. Environment cannot proceed.")
        else:
            logger.info(f"Successfully prepared {len(self.locations_data)} items for processing.")
            if self.locations_data: # Ensure not empty before shuffling
                random.shuffle(self.locations_data)
        self.iter = 0


    def save_checkpoint(self, step, data=None):
        if data is None: data = {}
        data["current_item_index"] = self.current_item_index
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def get_next_item(self) -> Optional[Dict[str, Any]]:
        if not self.locations_data:
            logger.warning("No locations data available in get_next_item.")
            return None
        if self.current_item_index >= len(self.locations_data):
            logger.info("Cycled through all available location data. Re-shuffling and resetting index.")
            random.shuffle(self.locations_data)
            self.current_item_index = 0
            if not self.locations_data: return None

        item_to_return = self.locations_data[self.current_item_index]
        self.current_item_index += 1
        self.iter +=1
        return item_to_return

    def _parse_llm_output(self, llm_text: str) -> Dict[str, Any]:
        think_content = ""
        tool_calls = []
        forecast_summary = ""

        think_match = re.search(r"<think>(.*?)</think>", llm_text, re.DOTALL | re.IGNORECASE)
        if think_match:
            think_content = think_match.group(1).strip()

        for line in llm_text.splitlines():
            line_upper = line.strip().upper()
            if line_upper.startswith("TOOL_CALL:"):
                try:
                    tool_json_str = line.strip()[len("TOOL_CALL:"):].strip()
                    if tool_json_str.startswith("{") and tool_json_str.endswith("}"):
                        tool_calls.append(json.loads(tool_json_str))
                    else:
                        logger.warning(f"Skipping malformed TOOL_CALL: {line}")
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse TOOL_CALL JSON: {line}")
            elif line_upper.startswith("FORECAST_SUMMARY:"):
                forecast_summary = line.strip()[len("FORECAST_SUMMARY:"):].strip()

        return {
            "think_content": think_content,
            "tool_calls": tool_calls,
            "forecast_summary": forecast_summary
        }


    def _parse_judge_output(self, judge_text: str) -> Dict[str, Any]:
        scores = {
            "reasoning": 0.0, "tool_call": 0.0, "forecast_summary": 0.0, "total": 0.0
        }
        justification = "No justification provided or parse error."

        patterns = {
            "reasoning": r"REASONING_SCORE:\s*([0-9.]+)",
            "tool_call": r"TOOL_CALL_SCORE:\s*([0-9.]+)",
            "forecast_summary": r"FORECAST_SUMMARY_SCORE:\s*([0-9.]+)",
            "total": r"TOTAL_SCORE:\s*([0-9.]+)"
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, judge_text, re.IGNORECASE)
            if match:
                try:
                    scores[key] = float(match.group(1))
                except ValueError:
                    logger.warning(f"Could not parse score for {key} from: {match.group(1)}")

        just_match = re.search(r"JUSTIFICATION:\s*(.*)", judge_text, re.DOTALL | re.IGNORECASE)
        if just_match:
            justification = just_match.group(1).strip()

        calculated_total = round(scores["reasoning"] + scores["tool_call"] + scores["forecast_summary"], 2)
        if abs(scores["total"] - calculated_total) > 0.1 :
            if scores["total"] != 0.0:
                 logger.warning(f"Parsed total score {scores['total']} differs from sum of components {calculated_total}. Using sum.")
            scores["total"] = calculated_total

        return {"scores": scores, "justification": justification}


    async def collect_trajectories(self, item: Dict[str, Any]) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        case_id = item.get('case_id', 'Unknown')
        logger.info(f"ITEM {case_id}: Starting collect_trajectories.")

        if item is None:
            logger.warning(f"ITEM {case_id}: Received None item in collect_trajectories.")
            return None, []

        soundings_blob = json.dumps(item.get("model_soundings_data", []), indent=2)
        agent_user_prompt = AGENT_USER_PROMPT_TEMPLATE.format(
            location_id=item.get("location_id", "N/A"),
            model_name=item.get("model_name", "N/A"),
            run_date_full_z=item.get("run_date_full_z", "N/A"),
            sounding_times_str=item.get("sounding_times_str", "N/A"),
            target_forecast_time_utc=item.get("target_forecast_time_utc", "N/A"),
            soundings_json_blob=soundings_blob
        )
        agent_messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}, {"role": "user", "content": agent_user_prompt}]

        if not self.agent_llm_server:
            logger.error(f"ITEM {case_id}: Agent LLM server not available.")
            return None, []

        logger.info(f"ITEM {case_id}: About to call Agent LLM...")
        agent_chat_completions_obj = None # Initialize
        try:
            start_time = time.time()
            agent_chat_completions_obj = await self.agent_llm_server.chat_completion(
                messages=agent_messages,
                model=self.agent_llm_server.config.model_name if self.agent_llm_server.config else "unknown_agent_model",
                n=self.config.group_size,
                max_tokens=self.config.max_reasoning_tokens_llm,
                temperature=0.7,
                stop=["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]
            )
            choices_received = len(agent_chat_completions_obj.choices) if agent_chat_completions_obj and hasattr(agent_chat_completions_obj, 'choices') else 0
            logger.info(f"ITEM {case_id}: Agent LLM call completed. Time: {time.time() - start_time:.2f}s. Choices received: {choices_received}")
        except Exception as e:
            logger.error(f"ITEM {case_id}: Agent LLM call failed: {e}")
            logger.error(traceback.format_exc())
            return None, []

        if not agent_chat_completions_obj or not hasattr(agent_chat_completions_obj, 'choices') or not agent_chat_completions_obj.choices:
            logger.warning(f"ITEM {case_id}: No choices received from Agent LLM or malformed response. Response: {agent_chat_completions_obj}")
            return None, []

        scored_data_group = ScoredDataGroup(tokens=[], masks=[], scores=[], overrides=[])
        afd_context_blob = "\n\n---\n\n".join(item.get("afd_texts", [])) if item.get("afd_texts") else "No AFDs provided for this case."

        for agent_choice_idx, agent_choice in enumerate(agent_chat_completions_obj.choices):
            choice_id = f"{case_id}_choice_{agent_choice_idx}"
            logger.info(f"ITEM {choice_id}: Processing agent choice.")

            llm_full_output_text = "" # Initialize
            if agent_choice and hasattr(agent_choice, 'message') and agent_choice.message and hasattr(agent_choice.message, 'content'):
                llm_full_output_text = agent_choice.message.content
                logger.debug(f"ITEM {choice_id}: Agent output received (first 200 chars): {llm_full_output_text[:200]}...")
            else:
                logger.warning(f"ITEM {choice_id}: Agent choice did not contain expected message content. Skipping this choice. Details: {agent_choice}")
                continue

            parsed_llm_out = self._parse_llm_output(llm_full_output_text)
            logger.info(f"ITEM {choice_id}: Parsed agent output.")

            judge_user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(llm_full_output=llm_full_output_text, afds_blob=afd_context_blob)
            judge_messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": judge_user_prompt}]

            final_score = 0.0
            judge_justification_text = "Judge call not made or failed."
            judge_parsed_scores = {}

            if not self.judge_server:
                logger.error(f"ITEM {choice_id}: Judge server not available. Assigning default score.")
            else:
                logger.info(f"ITEM {choice_id}: About to call Judge LLM...")

                # Log the request payload for the judge
                request_payload_for_judge = {
                    "messages": judge_messages,
                    "model": self.judge_server.config.model_name if self.judge_server.config else "unknown_judge_model",
                    "max_tokens": self.config.max_tokens_judge,
                    "temperature": 0.2,
                    "n": 1
                    # Add any other parameters being sent by APIServer implicitly or explicitly
                }
                logger.info(f"ITEM {choice_id}: Judge LLM Request Payload: {json.dumps(request_payload_for_judge, indent=2, ensure_ascii=False)}")


                try:
                    judge_start_time = time.time()
                    judge_completion_obj = await self.judge_server.chat_completion(
                        messages=judge_messages,
                        max_tokens=self.config.max_tokens_judge,
                        temperature=0.2,
                        n=1,
                        model=self.judge_server.config.model_name if self.judge_server.config else "unknown_judge_model"
                    )
                    logger.info(f"ITEM {choice_id}: Judge LLM call completed. Time: {time.time() - judge_start_time:.2f}s")

                    if judge_completion_obj and hasattr(judge_completion_obj, 'choices') and judge_completion_obj.choices and \
                    judge_completion_obj.choices[0] and hasattr(judge_completion_obj.choices[0], 'message') and \
                    judge_completion_obj.choices[0].message and hasattr(judge_completion_obj.choices[0].message, 'content'):
                        judge_output_text = judge_completion_obj.choices[0].message.content
                        logger.info(f"ITEM {choice_id}: Judge output received (first 200 chars): {judge_output_text[:200]}...")
                        parsed_judge_out = self._parse_judge_output(judge_output_text)
                        final_score = parsed_judge_out["scores"]["total"]
                        judge_justification_text = parsed_judge_out["justification"]
                        judge_parsed_scores = parsed_judge_out["scores"]
                        logger.info(f"ITEM {choice_id}: Parsed judge output. Score: {final_score}")
                        self.judge_total_scores_buffer.append(final_score)
                        self.judge_reasoning_scores_buffer.append(judge_parsed_scores.get("reasoning", 0.0))
                        self.judge_tool_scores_buffer.append(judge_parsed_scores.get("tool_call", 0.0))
                        self.judge_forecast_scores_buffer.append(judge_parsed_scores.get("forecast_summary", 0.0))
                    else:
                        logger.error(f"ITEM {choice_id}: Judge LLM response was empty or malformed. Details: {judge_completion_obj}")

                # ***** START OF MODIFIED/ADDED ERROR HANDLING BLOCK *****
                except httpx.HTTPStatusError as http_err:
                    logger.error(f"ITEM {choice_id}: Judge LLM call failed with HTTPStatusError: {http_err.response.status_code}")
                    logger.error(f"Request URL: {http_err.request.url}")

                    # Log request body (already logged above, but useful for context here)
                    logger.error(f"Request Body (for context):\n{json.dumps(request_payload_for_judge, indent=2, ensure_ascii=False)}")

                    logger.error(f"Response Headers:\n{http_err.response.headers}")
                    try:
                        # Attempt to read response body text.
                        # For async responses, if not already read, it might require await response.aread()
                        # but .text should be available if the response was processed by httpx.
                        response_body_str = http_err.response.text
                        logger.error(f"Response Body:\n{response_body_str}")
                    except Exception as resp_e:
                        logger.error(f"Could not decode or access response body text from HTTPStatusError: {resp_e}")
                    logger.error(traceback.format_exc())
                # ***** END OF MODIFIED/ADDED ERROR HANDLING BLOCK *****

                except Exception as e: # General catch-all
                    logger.error(f"ITEM {choice_id}: Judge LLM call failed with general Exception: {e}")
                    logger.error(traceback.format_exc())

            logger.info(f"ITEM {choice_id}: About to tokenize full trajectory...")
            full_trajectory_messages = agent_messages + [{"role": "assistant", "content": llm_full_output_text}]

            # Ensure self.tokenizer and self.config.max_token_length are available
            tokenized_output = None
            if hasattr(self, 'tokenizer') and self.tokenizer and hasattr(self.config, 'max_token_length'):
                tokenized_output = tokenize_for_trainer(self.tokenizer, full_trajectory_messages, self.config.max_token_length)
                logger.info(f"ITEM {choice_id}: Tokenization complete. Tokens length: {len(tokenized_output.get('tokens', [])) if tokenized_output else 'N/A'}")
            else:
                logger.error(f"ITEM {choice_id}: Tokenizer or max_token_length not available. Skipping tokenization.")


            if tokenized_output and tokenized_output.get("tokens") and len(tokenized_output["tokens"]) > 0:
                # Ensure scored_data_group is a dict-like object that supports append or assignment
                if not isinstance(scored_data_group, dict) and not hasattr(scored_data_group, 'append'):
                    logger.error(f"ITEM {choice_id}: scored_data_group is not a dict or list-like object. Type: {type(scored_data_group)}")
                else:
                    if isinstance(scored_data_group.get("tokens"), list): scored_data_group["tokens"].append(tokenized_output["tokens"])
                    if isinstance(scored_data_group.get("masks"), list): scored_data_group["masks"].append(tokenized_output["masks"])
                    if isinstance(scored_data_group.get("scores"), list): scored_data_group["scores"].append(final_score)

                    item_overrides = {
                        "case_id": case_id, "llm_think": parsed_llm_out["think_content"],
                        "llm_tools": str(parsed_llm_out["tool_calls"]), "llm_summary": parsed_llm_out["forecast_summary"],
                        "judge_justification": judge_justification_text,
                        "judge_score_reasoning": judge_parsed_scores.get("reasoning", 0.0),
                        "judge_score_tool": judge_parsed_scores.get("tool_call", 0.0),
                        "judge_score_forecast": judge_parsed_scores.get("forecast_summary", 0.0),
                    }
                    if isinstance(scored_data_group.get("overrides"), list): scored_data_group["overrides"].append(item_overrides)

                    if hasattr(self, 'rollouts_for_wandb_custom') and isinstance(self.rollouts_for_wandb_custom, list):
                        self.rollouts_for_wandb_custom.append((
                            agent_user_prompt[:300]+"...", parsed_llm_out["think_content"][:500]+"...",
                            str(parsed_llm_out["tool_calls"])[:300]+"...", parsed_llm_out["forecast_summary"][:300]+"...",
                            final_score, judge_justification_text[:500]+"..." ))
                    logger.info(f"ITEM {choice_id}: Added to scored_data_group and rollouts_for_wandb_custom.")
            else:
                logger.warning(f"ITEM {choice_id}: Tokenization failed or produced no tokens. Agent output was: {llm_full_output_text[:200]}...")

        if not scored_data_group.get("tokens"):
            logger.warning(f"ITEM {case_id}: No valid trajectories collected for this item.")
            return None, []

        logger.info(f"ITEM {case_id}: Finished processing all choices. Returning scored data group with {len(scored_data_group['tokens'])} trajectories.")
        return scored_data_group, []



    # --- evaluate and wandb_log remain the same or with similar logging detail if needed ---
    async def evaluate(self, *args, **kwargs):
        logger.info(f"Starting evaluation for {self.name or self.__class__.__name__}...")
        self.eval_metrics_buffer.clear()

        if not self.locations_data:
            logger.warning("No data available for evaluation.")
            return

        # Ensure config attributes exist before accessing
        num_eval_samples = getattr(self.config, "num_eval_samples", 20)
        eval_items_to_process = min(len(self.locations_data), num_eval_samples)

        if eval_items_to_process == 0:
            logger.warning("Not enough data or num_eval_samples is 0. Skipping evaluation.")
            return

        eval_list = []
        if self.locations_data : # ensure locations_data is not empty
             eval_list = random.sample(self.locations_data, k=eval_items_to_process)


        for eval_idx, eval_item_data in enumerate(eval_list):
            case_id = eval_item_data.get('case_id', f"UnknownEval_{eval_idx}")
            logger.info(f"EVAL ITEM {case_id}: Starting evaluation.")

            soundings_blob = json.dumps(eval_item_data["model_soundings_data"], indent=2)
            agent_user_prompt = AGENT_USER_PROMPT_TEMPLATE.format(
                location_id=eval_item_data["location_id"], model_name=eval_item_data["model_name"],
                run_date_full_z=eval_item_data["run_date_full_z"], sounding_times_str=eval_item_data["sounding_times_str"],
                target_forecast_time_utc=eval_item_data["target_forecast_time_utc"], soundings_json_blob=soundings_blob)

            agent_messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}, {"role": "user", "content": agent_user_prompt}]

            if not self.agent_llm_server:
                logger.error(f"EVAL ITEM {case_id}: Agent LLM server not available.")
                continue

            llm_full_output_text = "" # Initialize
            logger.info(f"EVAL ITEM {case_id}: About to call Agent LLM for evaluation.")
            try:
                agent_start_time = time.time()
                agent_completion_obj = await self.agent_llm_server.chat_completion(
                    messages=agent_messages,
                    n=1,
                    max_tokens=self.config.max_reasoning_tokens_llm,
                    temperature=0.1, # Low temp for eval
                    stop=["<|eot_id|>", "<|im_end|>", "<|endoftext|>"],
                    model=self.agent_llm_server.config.model_name if self.agent_llm_server.config else "unknown_agent_model"
                )
                logger.info(f"EVAL ITEM {case_id}: Agent LLM call completed. Time: {time.time() - agent_start_time:.2f}s")
                if agent_completion_obj.choices and agent_completion_obj.choices[0].message and agent_completion_obj.choices[0].message.content:
                    llm_full_output_text = agent_completion_obj.choices[0].message.content
                    logger.debug(f"EVAL ITEM {case_id}: Agent output received (first 200): {llm_full_output_text[:200]}")
                else:
                    logger.error(f"EVAL ITEM {case_id}: Agent LLM response was empty or malformed. Details: {agent_completion_obj}")
                    continue
            except Exception as e:
                logger.error(f"EVAL ITEM {case_id}: Agent LLM call failed: {e}")
                logger.error(traceback.format_exc())
                continue

            afd_context_blob = "\n\n---\n\n".join(eval_item_data["afd_texts"]) or "No AFDs."
            judge_user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(llm_full_output=llm_full_output_text, afds_blob=afd_context_blob)
            judge_messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": judge_user_prompt}]

            if not self.judge_server:
                logger.warning(f"EVAL ITEM {case_id}: Judge server not available.")
                continue

            logger.info(f"EVAL ITEM {case_id}: About to call Judge LLM for evaluation.")
            try:
                judge_start_time = time.time()
                judge_completion_obj = await self.judge_server.chat_completion(
                    messages=judge_messages,
                    max_tokens=self.config.max_tokens_judge,
                    temperature=0.1, # Low temp for eval
                    n=1,
                    model=self.judge_server.config.model_name if self.judge_server.config else "unknown_judge_model"
                )
                logger.info(f"EVAL ITEM {case_id}: Judge LLM call completed. Time: {time.time() - judge_start_time:.2f}s")
                if judge_completion_obj.choices and judge_completion_obj.choices[0].message and judge_completion_obj.choices[0].message.content:
                    judge_output_text = judge_completion_obj.choices[0].message.content
                    logger.debug(f"EVAL ITEM {case_id}: Judge output received (first 200): {judge_output_text[:200]}")
                    parsed_judge_out = self._parse_judge_output(judge_output_text)
                    self.eval_metrics_buffer.append(parsed_judge_out["scores"])
                    logger.info(f"EVAL ITEM {case_id}: Judge score {parsed_judge_out['scores']['total']} added to buffer.")
                else:
                    logger.error(f"EVAL ITEM {case_id}: Judge LLM response was empty or malformed. Details: {judge_completion_obj}")
            except httpx.HTTPStatusError as http_err: # Added specific error handling for eval as well
                logger.error(f"EVAL ITEM {case_id}: Judge LLM call failed with HTTPStatusError: {http_err.response.status_code}")
                logger.error(f"Request URL: {http_err.request.url}")
                # Consider logging request payload for eval too if needed for debugging eval failures
                logger.error(f"Response Headers:\n{http_err.response.headers}")
                try:
                    response_body_str = http_err.response.text
                    logger.error(f"Response Body:\n{response_body_str}")
                except Exception as resp_e:
                    logger.error(f"Could not decode or access response body text from HTTPStatusError during eval: {resp_e}")
                logger.error(traceback.format_exc())
            except Exception as e:
                logger.error(f"EVAL ITEM {case_id}: Judge LLM call failed: {e}")
                logger.error(traceback.format_exc())

        logger.info(f"Evaluation completed for {len(self.eval_metrics_buffer)} items.")

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if not self.config.use_wandb or not wandb.run: # Check if wandb is active
            logger.debug("WandB logging skipped (disabled or no active run).")
            if hasattr(super(), 'wandb_log'): # Call super if it exists, even if not logging locally
                 await super().wandb_log(wandb_metrics if wandb_metrics else {})
            return

        if wandb_metrics is None: wandb_metrics = {}
        logger.info("Preparing metrics for WandB log...")

        if self.judge_total_scores_buffer:
            wandb_metrics["train/avg_judge_total_score"] = sum(self.judge_total_scores_buffer) / len(self.judge_total_scores_buffer)
            wandb_metrics["train/avg_judge_reasoning_score"] = sum(self.judge_reasoning_scores_buffer) / len(self.judge_reasoning_scores_buffer)
            wandb_metrics["train/avg_judge_tool_score"] = sum(self.judge_tool_scores_buffer) / len(self.judge_tool_scores_buffer)
            wandb_metrics["train/avg_judge_forecast_score"] = sum(self.judge_forecast_scores_buffer) / len(self.judge_forecast_scores_buffer)
            logger.info(f"Train scores (total): {wandb_metrics['train/avg_judge_total_score']:.2f} from {len(self.judge_total_scores_buffer)} samples.")
            self.judge_total_scores_buffer.clear(); self.judge_reasoning_scores_buffer.clear(); self.judge_tool_scores_buffer.clear(); self.judge_forecast_scores_buffer.clear()

        if self.eval_metrics_buffer:
            avg_eval_total = sum(s['total'] for s in self.eval_metrics_buffer) / len(self.eval_metrics_buffer)
            avg_eval_reasoning = sum(s['reasoning'] for s in self.eval_metrics_buffer) / len(self.eval_metrics_buffer)
            avg_eval_tool = sum(s['tool_call'] for s in self.eval_metrics_buffer) / len(self.eval_metrics_buffer)
            avg_eval_forecast = sum(s['forecast_summary'] for s in self.eval_metrics_buffer) / len(self.eval_metrics_buffer)
            wandb_metrics["eval/avg_judge_total_score"] = avg_eval_total
            wandb_metrics["eval/avg_judge_reasoning_score"] = avg_eval_reasoning
            wandb_metrics["eval/avg_judge_tool_score"] = avg_eval_tool
            wandb_metrics["eval/avg_judge_forecast_score"] = avg_eval_forecast
            logger.info(f"Eval scores (total): {avg_eval_total:.2f} from {len(self.eval_metrics_buffer)} samples.")
            self.eval_metrics_buffer.clear()

        if self.rollouts_for_wandb_custom:
            if wandb.run: # Double check wandb is active
                table = wandb.Table(columns=["Prompt Hint", "LLM Think", "LLM Tools", "LLM Summary", "Judge Score", "Judge Justification"])
                num_to_log = min(len(self.rollouts_for_wandb_custom), getattr(self.config, "num_rollouts_to_log", 10)) # Use getattr for safety

                sample_to_log = []
                if num_to_log > 0 and self.rollouts_for_wandb_custom:
                    sample_to_log = random.sample(self.rollouts_for_wandb_custom, k=min(num_to_log, len(self.rollouts_for_wandb_custom)))

                for P, T, O, S, Sc, J in sample_to_log: table.add_data(P, T, O, S, Sc, J)
                if sample_to_log:
                    wandb_metrics["train/detailed_rollouts"] = table
                    logger.info(f"Logged {len(sample_to_log)} rollouts to WandB table.")
            self.rollouts_for_wandb_custom.clear()

        if wandb_metrics:
            logger.info(f"Logging to WandB: {list(wandb_metrics.keys())}")
            await super().wandb_log(wandb_metrics)
        else:
            logger.info("No new metrics to log to WandB in this step.")
            if hasattr(super(), 'wandb_log'): # Call super even if no new metrics locally
                await super().wandb_log({})


if __name__ == "__main__":
    try:
        MeteorologyForecastRLEnv.cli()
    except Exception as e:
        logger.critical(f"CRITICAL Error during CLI execution: {e}") # Use critical for top-level crash
        logger.critical(traceback.format_exc())