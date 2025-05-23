# Lean Theorem Proving Environment for Atropos

This environment allows testing Language Learning Models (LLMs) on Lean theorem proving tasks using `atroposlib`.

## Setup

1.  **Atroposlib**: Ensure `atroposlib` is installed. If you are working within the main `atropos` repository, this is typically done by navigating to the `atropos` root directory (i.e., `cd LeanCopilot/atropos`) and running:
    ```bash
    pip install -e .
    ```

2.  **Python Dependencies**: Navigate to this environment's directory (`LeanCopilot/atropos/environments/lean_proof_env/`) and install the required Python packages using the provided `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    This will install `datasets`, `wandb`, `tqdm`, and `python-dotenv`.

3.  **OpenAI API Key**: This environment requires an OpenAI API key. You can set it in one of two ways:
    *   As an environment variable:
        ```bash
        export OPENAI_API_KEY="your_actual_openai_api_key"
        ```
    *   Create a `.env` file within this directory (`LeanCopilot/atropos/environments/lean_proof_env/.env`) with the following content:
        ```
        OPENAI_API_KEY=your_actual_openai_api_key
        ```
        The script will automatically load this if the environment variable is not set.

4.  **Lean Installation**: Ensure Lean 4 is installed and the `lean` executable is in your system's PATH. You can find installation instructions on the [Lean official website](https://leanprover.github.io/lean4/doc/setup.html).

## Running the Environment

Make sure your current working directory is this environment's directory (`LeanCopilot/atropos/environments/lean_proof_env/`).

Run the environment script using the `process` command provided by `AtroposBaseEnv`:

```bash
python lean_proof_env.py process \
    --env.lean_problem_dataset_name="internal_simple_test" \
    --env.total_steps=5 \
    --env.group_size=1 \
    --env.wandb_name="lean_simple_test_run" \
    --openai.model_name="gpt-4o"
```

### Command-Line Arguments:

*   `--env.lean_problem_dataset_name`:
    *   `"internal_simple_test"`: Uses a small set of hardcoded simple theorems (defined within the script).
    *   `"Tonic/MiniF2F"`: Attempts to load the MiniF2F benchmark from Hugging Face datasets. You might need to specify `--env.lean_problem_dataset_split` (e.g., "test" or "train").
*   `--env.total_steps`: Number of problems (or items) to process. The `internal_simple_test` set has 8 problems; if `total_steps` is less, a random subset will be chosen for that many steps.
*   `--env.group_size`: Number of LLM attempts per problem.
*   `--env.wandb_name`: Name for the WandB run if `use_wandb` is enabled (e.g., by passing `--env.use_wandb=True`).
*   `--openai.model_name`: The OpenAI model to use (e.g., "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"). This overrides the default in `config_init`.
*   `--openai.api_key`: Can be passed explicitly, but using the environment variable or `.env` file is recommended and more secure.
*   `--openai.base_url`: If using a custom OpenAI-compatible API endpoint.

The script will generate a `.jsonl` file with the trajectories and an HTML visualization of the rollouts. By default, these are saved in a `data/` subdirectory created within the current working directory (e.g., `LeanCopilot/atropos/environments/lean_proof_env/data/`).

## Customization

*   **Problem Sets**: Modify the `setup()` method in `lean_proof_env.py` to load different Lean problems or datasets.
*   **LLM Prompts**: Adjust the `get_next_item()` method in `lean_proof_env.py` to change the prompts sent to the LLM.
*   **Configuration**: Default configurations are defined in the `LeanProofEnvConfig` class within `lean_proof_env.py`. Most of these can be overridden by command-line arguments (e.g., `--env.max_proof_generation_tokens=256`). 