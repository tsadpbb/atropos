# Community Environments

This directory is home to community-contributed training environments for Atropos. Environments submitted by the community will be placed here after an initial code review.

**Note:** Environments in this directory are pending full testing and integration. While they have passed a basic code check, they may not yet have been rigorously validated on our compute cluster.

## Contributing Your Environment

We encourage you to contribute your own RL environments! When developing a new environment, please follow these guidelines:

1. **Create your environment in this `environments/community/` subdirectory.** This helps us keep new submissions organized.
2. **Preferred Import Style:** We prefer that you treat your environment's directory as the package root for imports within your environment code. For example, if you need to import `SomeClass`, you can do so directly:
   ```python
   from some_file_in_my_env import SomeClass
   ```
   This helps maintain consistency and makes it easier to integrate your environment.

### Environment Standards

Community environments should:
- Include clear documentation and setup instructions
- Specify all dependencies in requirements files
- Provide example configurations and usage
- Follow the AtroposBaseEnv pattern for consistency
- Include appropriate error handling and validation

### Submission Process

To contribute a new environment to the community collection:

1. **Fork the repository** and create a new branch
2. **Add your environment** to this `community/` directory
3. **Include comprehensive documentation**:
   - README with setup instructions
   - Requirements file for dependencies
   - Example usage and configuration
4. **Follow naming conventions**:
   - Use descriptive directory names for complex environments
   - Single file environments should have descriptive names
5. **Test thoroughly** before submitting
6. **Submit a pull request** with a clear description

Once your environment is ready, please follow the guidelines in our main [CONTRIBUTING.md](../../../CONTRIBUTING.md) to submit your contribution.

---

## Available Environments

### 1. Lean Proof Environment (`lean_proof_env/`)
**Author**: [GabinFay](https://github.com/GabinFay)
**Purpose**: Testing Language Learning Models (LLMs) on Lean theorem proving tasks

A comprehensive environment for evaluating LLMs on formal mathematical reasoning using the Lean theorem prover. Features include:
- Support for custom problem datasets or MiniF2F benchmark
- Integration with Lean 4 theorem prover
- Configurable difficulty levels and problem sets
- Automated proof validation

**Requirements**: Lean 4 installation, OpenAI API key

### 2. Router Environment (`router_env/`)
**Author**: [GabinFay](https://github.com/GabinFay)
**Purpose**: Multi-agent routing and coordination system

A sophisticated environment for testing agent routing and coordination capabilities. Includes:
- Multiple specialized agents (calendar, contact, Gmail, telephony, etc.)
- Model Contextualized Protocol (MCP) tools integration
- Spotify, Google Maps, and Perplexity integrations
- Complex multi-turn conversation handling

**Features**:
- Telephony agent with inbound/outbound call handling
- Calendar and contact management
- Memory and calculation agents
- Router agent for intelligent task delegation

### 3. Philosophical RLAIF Environment (`philosophical_rlaif_env.py`)
**Author**: [GabinFay](https://github.com/GabinFay)
**Purpose**: Reinforcement Learning from AI Feedback (RLAIF) for philosophical reasoning

An environment focused on training models for deep philosophical inquiry and reasoning. Features:
- Deep thinking prompts with systematic reasoning processes
- Preference learning for philosophical depth and nuance
- Multi-perspective analysis and assumption questioning
- Evaluation of response quality for philosophical discussions

**Capabilities**:
- Generates paired responses for preference comparison
- Uses judge models to evaluate philosophical depth
- Tracks preference consistency and reasoning quality
- Supports WandB logging for training insights

---

## Support

For questions or issues with community environments:
- Check the individual environment's README first
- Open an issue in the main repository
- Tag the environment author if possible

*These environments are community contributions and may have different maintenance levels and support compared to core Atropos environments.*
