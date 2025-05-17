# Environments

This directory contains various environments for training and evaluating language models on different tasks. Each environment implements a specific task with its own input format, reward function, and evaluation metrics.

## Available Environments

---

###  MCQA Thinking Environment (`mcqa_thinking_env.py`)

Multiple Choice Question Answering environment that requires models to think through problems systematically.

**Input Format:**
- Questions from the MMLU (Massive Multitask Language Understanding) dataset
- Each item contains:
  - `prompt`: The question text
  - `answer`: Index of correct answer
  - `ground_truth`: Letter (A, B, C, D) of correct answer
  - `options`: List of possible answers

**System Prompt:**
```
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
```

**Reward Function:**
- Score of 1.0 if the model's answer matches the ground truth letter
- Score of 0.0 if incorrect or invalid response (multiple think tags, malformed thinking sections)
- Length penalty applied if all responses are correct:
  - No penalty for responses under 50% of max token length
  - Linear penalty scaling from 1.0 down to 0.0 for responses between 50% and 100% of max length
  - Returns None if all scores are identical (no learning signal)

---

### GSM8K Environment (`gsm8k_server.py`)

Mathematical reasoning environment using the GSM8K dataset.

**Input Format:**
- Questions from GSM8K dataset
- Each item contains:
  - `question`: The math problem
  - `answer`: The numerical answer

**System Prompt:**
```
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.

You are allocated a maximum of 2048 tokens, please strive to use less.

You will then provide your answer like this: \boxed{your answer here}
It is important that you provide your answer in the correct format.
If you do not, you will not receive credit for your answer.
So please end your answer with \boxed{your answer here}
```

**Reward Function:**
- Score of 1.0 if the model's answer matches the ground truth (using LaTeX verification)
- Score of 0.0 if incorrect or if ground truth is not parseable
- Length penalty applied if all responses are correct:
  - No penalty for responses under 50% of max token length
  - Linear penalty scaling from 1.0 down to 0.0 for responses between 50% and 100% of max length
  - Returns None if all scores are identical (no learning signal)

---

### Tool Calling Environment (`tool_calling_server.py`)

Environment for training models to make function calls in a structured format.

**Input Format:**
- Conversations from ShareGPT-Hermes function call dataset
- Each item contains:
  - `conversations`: List of messages with roles (system, human, gpt)
  - Expected tool calls in JSON format

**System Prompt:**
```
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
```

**Reward Function:**
- Score of 1.0 if all expected tool calls are present and match exactly (including nested JSON fields)
- Score of 0.0 if any tool calls are missing, incorrect, or malformed
- Length penalty applied if all responses are correct:
  - No penalty for responses under 50% of max token length
  - Linear penalty scaling from 1.0 down to 0.0 for responses between 50% and 100% of max length
  - Returns None if all scores are identical (no learning signal)

---

### Instruction Following Environment (`instruction_following_algorithm_environment.py`)

**Dependencies:**
- `datasets` (Hugging Face)
- `langdetect`

This environment was inspired by AllenAI's RLVR-IFEVAL environment and uses AllenAI's dataset from their Tulu3 paper and project:
- Dataset: https://huggingface.co/datasets/allenai/RLVR-IFeval
- Paper: https://arxiv.org/abs/2411.15124

Environment for training models to follow natural language instructions and constraints, based on the `allenai/RLVR-IFeval` dataset and environment.
**Dependencies:**
- `datasets` (Hugging Face)
- `langdetect`

**Input Format:**
- Each item from the processed `allenai/RLVR-IFeval` dataset contains:
  - `prompt`: The user's instruction string.
  - `func_name`: The string name of the verifier function (from a predefined map) used to check if the instruction is followed.
  - `args`: A dictionary of arguments for the specified verifier function.

**System Prompt:**
```
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
```

**Reward Function:**
- Score of 1.0 if the model's response correctly follows the instruction, as determined by the specific verifier function associated with the input prompt.
- Score of 0.0 if the response fails the verifier function.
- Length penalty applied if all responses in a batch are correct (receive a score of 1.0 before penalty):
  - No penalty for responses under a certain percentage (e.g., 75%) of max token length.
  - Linear penalty scaling from 1.0 down to 0.0 for responses between the threshold and 100% of max length.
  - Returns None if all scores are identical after potential penalties (no learning signal).

**Unique Configuration and Features:**
- **Dataset Configuration (`IFConfig`):
  - `dataset_name`: Specifies the primary dataset to use (defaults to `allenai/RLVR-IFeval`).
  - `dataset_config_name`: Optional name for a specific configuration or subset of the dataset.
  - `test_set_ratio`: Defines the proportion of the dataset reserved for testing (defaults to 5%).

- **Verifier-Based Scoring:** Utilizes a comprehensive map of verifier functions (`IF_FUNCTIONS_MAP`) to evaluate whether the model's
output adheres to diverse and specific constraints defined in the input instructions (e.g., keyword presence, response length, JSON format, etc.).

- **Specialized Dataset Processing:** The `setup` method is specifically designed to parse the `allenai/RLVR-IFeval` dataset, extracting user instructions, the corresponding verifier function name, and its arguments.

- **Fallback Mechanism:** Includes a fallback to a small, predefined dummy dataset if the primary dataset (`allenai/RLVR-IFeval`) cannot be loaded, ensuring operational continuity for testing or development.

## Common Features

All environments share these common features:

1. **Training/Test Split:**
   - 98% training, 2% test split
   - Random shuffling with fixed seed (42)

2. **Metrics Tracking:**
   - Percent correct buffer
   - Completion lengths
   - Wandb integration for visualization
   - Rollout tracking

3. **Token Management:**
   - Maximum token length limits
   - Token length statistics tracking
   - Length penalty for excessive responses

4. **Evaluation:**
   - Separate evaluation on test set
   - Comprehensive metrics logging
   - Support for multiple model completions per prompt

## Usage

Each environment can be initialized with:
- `config`: BaseEnvConfig object
- `server_configs`: List of OpenAI API configurations
- `slurm`: Boolean for distributed training
- `testing`: Boolean for testing mode

The environments follow a common interface with methods for:
- `setup()`: Loading and preparing datasets
- `get_next_item()`: Retrieving next training item
- `collect_trajectories()`: Generating model responses
- `score()`: Computing rewards
- `evaluate()`: Running evaluation on test set
- `wandb_log()`: Logging metrics to Weights & Biases
