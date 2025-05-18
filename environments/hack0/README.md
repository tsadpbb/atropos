# Punchline VR-CLI Environment

This fork contains an Atropos environment designed to train a large language model to generate humorous punchlines for jokes. The environment utilizes a Reinforcement Learning (RL) technique called Verifiable Rewards via Completion Likelihood Improvement (VR-CLI), taken from the paper "Learning to Reason for Long-Form Story Generation" (Gurung & Lapata, 2025) [https://arxiv.org/html/2503.22828v1](https://arxiv.org/html/2503.22828v1).

## Environment Design and Motivation

The core idea is to teach a model not just to produce a punchline, but to first generate a "reasoning" or "thought process" that leads to a good punchline. The quality of this reasoning is then "verified" by measuring how much it improves the likelihood (reduces the perplexity) of the *actual* punchline from the dataset, as assessed by a separate, fixed reference model. This greatly reduces overfitting, as the model does not have access to the ground-truth answer.

## Example

Here's an example of how the model generates a punchline with reasoning:

```
Question: What do you call a herd of cows masturbating?

<think>
Okay, the user is asking, "What do you call a herd of cows masturbating?" Hmm, that's a play on words. Let me think. The key here is the word "masturbating" and the animal "cows." The answer needs to be a pun or a play on words.

First, I need to connect "herd" with "masturbating." A herd of cows is a group, so maybe something related to a group. "Masturbating" is a term that's not typically used for animals, but maybe there's a word that combines the two.........
</think>

Beef strokin off!
```

### Key Components:

*   **Dataset:** The environment uses the `"SocialGrep/one-million-reddit-jokes"` dataset, filtering for jokes with a question-answer format (setup and punchline) and a minimum number of upvotes.
*   **Task:** Given the setup of a joke (the "question"), the model is prompted to generate a thinking process `<think>...</think>` followed by the punchline.
*   **Reward (VR-CLI):**
    1.  A base perplexity of the "golden" punchline is calculated given only the joke's setup, using a reference LLM `Qwen/Qwen3-1.7B-Base`.
    2.  A new perplexity of the golden punchline is calculated, this time conditioned on both the joke's setup AND the model-generated reasoning.
    3.  The reward is proportional to the improvement in perplexity `(base_perplexity - plus_perplexity) / base_perplexity`. A positive reward indicates the reasoning was helpful.
*   **Models:**
    *   The environment is configured to use `Qwen/Qwen3-1.7B` for generating trajectories.
    *   A reference model `Qwen/Qwen3-1.7B-Base` is used locally to calculate the VR-CLI reward.

The motivation is to guide the LLM towards generating more creative and contextually relevant punchlines by explicitly rewarding the intermediate reasoning steps that make a punchline "work." Typical fine-tuning fails to do this, as it makes the models memorize the jokes rather than gain an understanding of what makes them funny.

## Prompt

The model is giving the following prompt to guide the reasoning process. Qwen3 models have reasoning built-in, so we do not need to explicitly ask for `<think></think>` tags.

`You have a strong sense of humor and answer the user's question with a punchline. You always give the funniest answer, even if it could offend some people. Consider the aspects that make a joke funny, for example the answer is usually surprising to hear but makes sense in hindsight. You shouldn't need to explain your answer, it should stand on its own.`

## Metrics

The `evaluate` function selects 64 random jokes, then computes the average reward using greedy decoding. This is a simple and effective way to measure the model's progress over time.

## Quickstart

### Prerequisites

Ensure you have Python installed, then install the dependencies:

```bash
pip install "vllm>=0.8.5" torch transformers datasets wandb tenacity atroposlib pydantic
```

### Processing the Dataset

To process the dataset, use the `punchline_env.py` script with the `process` command:

```bash
python punchline_env.py process \
  --env.data_path_to_save_groups jokes.jsonl \
  --env.wandb_name punchline_vrcli \
  --openai.base_url http://localhost:9001/v1 \
  --openai.api_key x \
  --openai.model_name Qwen/Qwen3-1.7B
```

You will need to have vLLM serving the model on port 9001 for this to work.

[Weights & Biases link](https://wandb.ai/jaboggs-nous-hackathon-nc-state-university/uncategorized/runs/c24sz5t5)

#### Output

Zip placeholder