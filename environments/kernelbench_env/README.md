# KernelBench Environment Setup Instructions

## Prerequisites

Before running this script, you need to install KernelBench:

1. Install KernelBench from source:
   ```bash
   pip install git@github.com:ScalingIntelligence/KernelBench.git
   cd KernelBench
   pip install -r requirements.txt
   pip install -e .
   cd -
   ```

2. Set variables at the top of this script:
   - `KERNELBENCH_LEVEL`: The difficulty level (1-3)
   - `KERNELBENCH_PROBLEM_NUMBER`: The specific problem number to solve
   - `KERNELBENCH_DIR`: The absolute path to your KernelBench install

These environment variables will be used to configure the evaluation environment.

## Citations

> Baronio, C., Marsella, P., Pan, B., & Alberti, S. (2025 May 6). **Kevin‑32B: Multi‑Turn RL for Writing CUDA Kernels**. *Cognition AI Blog*. Retrieved May 16, 2025, from <https://cognition.ai/blog/kevin-32b>
