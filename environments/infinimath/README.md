# InfiniteMath Environment

## Environment Overview

This environment provides procedurally generated math problems with curriculum-based advancement. It allows an agent to solve increasingly difficult math problems, with the difficulty level adapting based on performance.

**Demonstrates:**
- Procedural content generation (math problems).
- Curriculum learning: The environment automatically adjusts the difficulty (levels 1-7) based on the LLM's success rate.
- Step-by-step reasoning evaluation: Rewards correctness, the presence of reasoning steps (within `<think>` tags), and the final answer format (`\boxed{}`).
- Handling LaTeX formatting for problems and answers.

**Training Goal:**
- To train LLMs to solve mathematical problems accurately.
- To encourage explicit step-by-step reasoning before providing an answer.
- To improve the LLM's ability to follow specific formatting instructions (using `<think>` tags and `\boxed{}`).
- To teach the model to handle progressively more complex problems through the curriculum.

## Features

- Progressive difficulty scaling across 7 levels of math problems
- Built-in curriculum system that adapts to agent performance
- Automatic problem generation with solutions
- Reward functions for accuracy, formatting, and boxed answer checking

## Usage

### Running with Default Configuration

To run the InfiniteMath environment with the default configuration:

```bash
python environments/infinite_math/infinimath_local_server.py
```

This will use the default configuration from `configs/envs/infinimath.yaml`.

### Custom Configuration

You can specify a custom configuration file:

```bash
python environments/infinite_math/infinimath_local_server.py --config my_custom_config
```

The `--config` parameter can be:

1. A name (without `.yaml` extension) which will be looked up in `configs/envs/`
2. A relative or absolute path to a YAML file

For example:
```bash
# Using a config in configs/envs/
python environments/infinite_math/infinimath_local_server.py --config infinimath_hard

# Using a config with full path
python environments/infinite_math/infinimath_local_server.py --config /path/to/my/config.yaml
```

## Configuration Structure

The configuration file follows this structure:

```yaml
# Base environment parameters
tokenizer_name: "NousResearch/DeepHermes-3-Llama-3-8B-Preview"
group_size: 1
use_wandb: false
# ... other base parameters

# InfiniteMath specific configuration
infinimath:
  # Curriculum parameters
  starting_level: 1
  progress_threshold: 0.7
  # ... other InfiniteMath specific parameters

# Server configuration
server_configs:
  - model_name: "gpt-4.1-nano"
    api_key: ${OPENAI_API_KEY}
    num_requests_for_eval: 70
```

### Important Configuration Parameters

#### Base Parameters

- `tokenizer_name`: The tokenizer to use for encoding/decoding text
- `group_size`: Number of responses to collect per prompt
- `max_token_length`: Maximum token length for generation
- `steps_per_eval`: How often to run evaluations

#### InfiniteMath Specific Parameters

- `starting_level`: Initial difficulty level (1-7)
- `progress_threshold`: Success rate needed to advance levels
- `min_evaluations`: Minimum number of evaluations before level advancement
- `reward_functions`: List of reward functions to apply

#### Server Configuration

- `model_name`: LLM model to use
- `api_key`: API key for the model (can use environment variables with ${VAR_NAME} syntax)
- `num_requests_for_eval`: Number of evaluation requests to allocate
