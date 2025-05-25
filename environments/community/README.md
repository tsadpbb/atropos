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

### 4. Playwright Agent Environment (`playwright_agent_env.py`)
**Author**: [erikqu](https://github.com/erikqu)
**Purpose**: Web automation and browser interaction for LLM agents

A comprehensive environment for training LLMs to interact with web pages through browser automation. Features:
- Playwright-based browser control with headless operation
- Screenshot-based visual input for LLM decision making
- JSON-based action commands (navigate, click, type, finish)
- Video recording of browser sessions for evaluation
- Google Gemini integration for success evaluation

**Capabilities**:
- Loads tasks from WebVoyager dataset or custom task definitions
- Supports development mode for testing without LLM calls
- Automatic reward computation based on success and efficiency
- Comprehensive error handling and fallback mechanisms
- Integration with Atropos training pipeline

**Requirements**: Playwright, optional Google Gemini API for evaluation

### 5. Metric Card Generator Environment (`metric_card_generator/`)
**Author**: [vivek100](https://github.com/vivek100)
**Purpose**: Structured JSON generation for AI model evaluation dashboards

A comprehensive environment for training LLMs to generate well-structured JSON configurations for Metric Card UI components. Features:
- Closed-loop generation, evaluation, and visualization pipeline
- Schema validation for JSON metric card configurations
- Multi-dimensional evaluation (validity, compliance, semantic quality)
- Support for various business domains and metric types
- WandB integration for performance tracking

**Capabilities**:
- Generates metric cards for diverse business contexts (e-commerce, finance, healthcare, etc.)
- Validates JSON structure against predefined schemas
- Evaluates semantic quality and formatting consistency
- Provides training data extraction and filtering utilities
- Includes visualization tools for score distribution analysis

**Components**:
- `metric_card_generator.py`: Main environment implementation
- `extract_metric_training.py`: Training data extraction utility
- `trainingDataScript.py`: Dataset creation from collected examples
- `show_score_distribution.py`: Performance analysis visualization

**Requirements**: Pydantic, tqdm

### 6. UFC Prediction Environment (`ufc_prediction_env/`)
**Author**: [edmundman](https://github.com/edmundman)
**Repository**: [UFC_FIGHT_PREDICTOR](https://github.com/edmundman/UFC_FIGHT_PREDICTOR)
**Purpose**: UFC fight prediction with entertaining TTS-ready commentary generation

A creative environment that transforms traditional fight prediction into engaging entertainment by generating dynamic, broadcast-style UFC fight commentary. Features both text-based and image-based prediction modes:

**Text-Based Predictor (`ufc_server.py`)**:
- Uses comprehensive fighter statistics (wins/losses, physical attributes, performance metrics)
- Generates dramatic fight commentary with commentator personalities
- TTS-ready output with natural speech patterns and emphasis markers
- Statistical analysis wrapped in entertaining storytelling

**Image-Based Predictor (`ufc_image_env.py`)**:
- Multimodal prediction using fighter profile images
- Visual analysis transformed into engaging commentary
- Base64 image encoding for API compatibility
- Creates dramatic narratives from fighter appearances

**Key Features**:
- Entertainment-first approach with broadcast-style commentary
- Direct TTS integration compatibility (designed for models like DIA)
- Dramatic elements including commentator phrases and pauses
- Proper formatting for voice synthesis applications
- Comprehensive scoring system for prediction accuracy and entertainment value

**Data Components**:
- `fighter_stats.csv`: Detailed fighter statistics and performance metrics
- `large_dataset.csv`: Sample historical fight data (799 records from original 7,440)
- `fighter_images/`: Profile images for visual-based predictions
- `get_images.py`: Web scraping utility for fighter image collection

**Note**: The included dataset is a sample for demonstration. The full dataset (7,440 fight records) is available in the original [UFC_FIGHT_PREDICTOR repository](https://github.com/edmundman/UFC_FIGHT_PREDICTOR).

**Additional Tools**:
- `ufc_predictor_ui.py`: Flask-based web interface for interactive predictions
- Video demonstrations and example runs available
- WandB integration for training tracking

**Requirements**: PIL, OpenAI API, Flask (for UI), BeautifulSoup4 (for image scraping)

### 7. Accessibility Auto-Fixer Environment (`accessibility_env/`)
**Author**: [joshgarza](https://github.com/joshgarza)
**Purpose**: Automated web accessibility remediation using WCAG guidelines

A specialized environment for training LLMs to automatically identify and fix web accessibility issues in HTML snippets. The environment focuses on objective, rule-based WCAG compliance improvements with minimal code changes.

**Features**:
- Rule-based scoring system for WCAG 2.1 AA compliance
- Support for multiple accessibility criteria (alt text, form labels, link text)
- BeautifulSoup-based HTML parsing and validation
- Automated scoring for accessibility improvements
- Integration with common accessibility testing patterns

**Targeted WCAG Criteria**:
- **Images**: Missing or empty `alt` attributes (WCAG 1.1.1)
- **Form Labels**: Improper `<label for="...">` associations (WCAG 1.3.1, 3.3.2, 4.1.2)
- **Links**: Lacking discernible text or accessible name (WCAG 2.4.4, 4.1.2)

**Scoring System**:
- +1.0: All targeted issues fixed correctly
- 0.0-0.8: Partial fixes applied
- -0.5: Parseable HTML but no issues fixed
- -1.0: Unparseable HTML or regressions introduced

**Note**: The accessibility dataset referenced in the environment (`data/accessibility_dataset.jsonl`) was not included in the contribution. Please contact the author for access to the training dataset.

**Requirements**: BeautifulSoup4, lxml, OpenAI API

### 8. ExamCraft - Adaptive LLM Teacher Environment (`examcraft/`)
**Author**: [RoshanSanjeev](https://github.com/RoshanSanjeev)
**Purpose**: Train language models to become adaptive teachers through reinforcement learning

A sophisticated environment for training LLMs to be effective teachers by generating adaptive questions, providing explanations, and creating personalized lesson plans. The environment simulates realistic student-teacher interactions with comprehensive reward systems for teaching effectiveness.

**Features**:
- Adaptive question generation targeting student weak areas
- Real-time difficulty adjustment based on student ability
- Multiple teaching actions (questions, explanations, lesson plans)
- Sophisticated multi-factor reward system for teaching effectiveness
- Realistic student learning simulation with proficiency progression
- Session momentum and learning impact tracking

**Teaching Actions**:
- **QUESTION**: Generate adaptive multiple-choice questions
- **EXPLANATION**: Provide detailed concept explanations
- **LESSON_PLAN**: Create personalized study plans

**Reward Components**:
- Correctness reward for student success
- Targeting bonus for focusing on weak topics
- Difficulty appropriateness scoring
- Content quality assessment
- Learning impact measurement

**Student Simulation**:
- Probabilistic responses based on topic proficiency
- Dynamic learning from effective teaching
- Realistic difficulty sensitivity and momentum effects
- Configurable learning styles and goals

**Applications**:
- Adaptive AI tutoring system development
- Personalized education at scale
- Automated knowledge gap identification
- Quality education accessibility improvement

**Requirements**: OpenAI API, JSON configuration support

### 9. Cat Behavior Communication Environment (`cat_behavior_env/`)
**Author**: [krishpop](https://github.com/krishpop)
**Purpose**: Train language models to communicate as cats with their caretakers

A unique environment for training LLMs to express needs and desires through authentic cat behaviors and vocalizations. Models must learn to communicate without using human language, relying instead on realistic cat sounds, body language, and behaviors to convey their needs to caretakers.

**Features**:
- **Authentic Cat Behavior Database**: 35 detailed cat behaviors with scientific descriptions
- **Diverse Scenario Coverage**: 61 cat care scenarios spanning nutrition, health, comfort, and enrichment
- **Multi-turn Interactions**: 5-turn conversations between cat and caretaker
- **Strict Communication Rules**: No English, no emojis - only realistic cat communication
- **"Purrfect" Evaluation**: Cats judge whether caretakers addressed all needs perfectly

**Cat Behaviors Included**:
- **Vocalizations**: Meowing, purring, trilling, yowling, hissing, growling
- **Body Language**: Tail position, ear orientation, back arching, slow blinking
- **Physical Actions**: Kneading, head butting, rubbing, scratching, following
- **Behavioral Indicators**: Hiding, litter box changes, grooming patterns

**Scenario Categories**:
- **Nutrition**: Balanced diet, feeding schedules, fresh water, treats
- **Health Care**: Veterinary visits, grooming, dental hygiene, medications
- **Comfort & Safety**: Sleeping areas, temperature control, secure environment
- **Enrichment**: Mental stimulation, play, social interaction, territory

**Communication Format**:
- `Sound! (Context)`: For vocalizations with body language
- `~Silent~ (Context)`: For non-vocal behaviors
- Examples: `Mew! (Looks up at you)`, `~Silent~ (Rubs against your legs)`

**Scoring System**:
- **1.0**: "Purr" - Perfect caretaking with no possible improvements
- **0.0**: "Meow" - Needs remain unmet or could be better addressed

**Research Applications**:
- Non-verbal communication modeling
- Animal-human interaction patterns
- Empathy and care training for AI
- Creative roleplay and character consistency

**Status**: ⚠️ Environment in active development - some code may need refinement

**Requirements**: Standard Atropos dependencies, JSON file handling

### 10. Punchline VR-CLI Environment (`punchline_vrcli/`)
**Author**: [JakeBoggs](https://github.com/JakeBoggs)
**Purpose**: Train LLMs to generate humorous punchlines using Verifiable Rewards via Completion Likelihood Improvement (VR-CLI)

A specialized environment for training LLMs to understand humor by generating joke punchlines through a novel RL technique from the paper "Learning to Reason for Long-Form Story Generation" (Gurning & Lapata, 2025). The environment teaches models to first generate reasoning that leads to good punchlines, with rewards based on how much the reasoning improves the likelihood of the actual punchline.

**Features**:
- **VR-CLI Methodology**: Uses Verifiable Rewards via Completion Likelihood Improvement for reduced overfitting
- **Reasoning-First Approach**: Models learn to generate `<think>...</think>` reasoning before punchlines
- **Perplexity-Based Rewards**: Reward calculated by improvement in punchline likelihood given reasoning
- **Reddit Jokes Dataset**: Uses SocialGrep/one-million-reddit-jokes filtered for quality
- **Anti-Memorization**: Prevents overfitting by using separate reference model for evaluation

**Training Process**:
1. Model generates reasoning for joke setup
2. Reference model calculates base perplexity of golden punchline given setup only
3. Reference model recalculates perplexity with setup + generated reasoning
4. Reward = `(base_perplexity - reasoning_perplexity) / base_perplexity`
5. Positive rewards indicate helpful reasoning that improves punchline understanding

**Key Components**:
- **Dataset**: SocialGrep one-million-reddit-jokes with question-answer format filtering
- **Model**: Qwen/Qwen3-1.7B for generation
- **Reference**: Qwen/Qwen3-1.7B-Base for perplexity evaluation
- **Evaluation**: 64 random jokes with greedy decoding for progress tracking

**Applications Beyond Humor**:
- Creative writing assistance
- Code generation without execution environments
- Business task reasoning with existing examples
- Any domain requiring explanatory reasoning before output

**Example Output**:
```
Question: What do you call a herd of cows masturbating?

<think>
The user is asking a play-on-words question. I need to connect "herd"
with "masturbating" to create a pun. "Masturbating" could become
"stroking" and combine with "beef"...
</think>

Beef strokin off!
```

**Requirements**: vllm>=0.8.5, torch, transformers, datasets, wandb, tenacity, pydantic

**W&B Results**: [Training Run](https://wandb.ai/jaboggs-nous-hackathon-nc-state-university/uncategorized/runs/0vly0u4p)

### 11. Selcube - Rubik's Cube Training Environment (`selcube/`)
**Author**: [joshuajerin](https://github.com/joshuajerin) with [Tvpower](https://github.com/Tvpower)
**Purpose**: Train LLMs to solve Rubik's cubes through structured 3D reasoning and multi-step planning

A comprehensive environment for training LLMs on the challenging task of Rubik's cube solving, designed to improve spatial reasoning, strategic planning, and structured problem-solving capabilities. The environment provides measurable, domain-specific challenges that require both visualization and logical reasoning.

**Features**:
- **Multi-step Planning**: Tests ability to understand cube mechanics and develop solving strategies
- **3D Spatial Reasoning**: Models must mentally track complex 3D spatial relationships
- **Curriculum Learning**: Configurable difficulty based on scramble complexity (1-22 moves)
- **Token-level Rewards**: Granular feedback system that enhances learning signal
- **Multiple Solving Strategies**: Supports Layer-by-Layer, CFOP, and other approaches
- **Anti-Reward Hacking**: Validates moves against actual cube state to prevent gaming

**Key Components**:
- **Environment Logic** (`rubiks_cube_environment.py`): Main training environment with curriculum support
- **Cube Mechanics** (`rubiks_cube_logic.py`): Core Rubik's cube state management and move validation
- **Solving Strategies** (`rubiks_strategies.py`): Multiple algorithmic approaches for teaching
- **Token Rewards** (`rubiks_token_rewards.py`): Sophisticated reward system for quality feedback
- **Curriculum** (`rubiks_cube_curriculum.py`): Progressive difficulty scaling
- **Enhanced Visualizer** (`rubiks_enhanced_visualizer.py`): Comprehensive progress tracking and analysis

**Training Performance**:
- **Level 1 (1-3 moves)**: 97% solve rate
- **Level 2 (4-7 moves)**: 85% solve rate
- **Level 3 (8-12 moves)**: 72% solve rate
- **Level 4 (13-17 moves)**: 53% solve rate
- **Level 5 (18-22 moves)**: 31% solve rate
- **Token efficiency improvement**: 34% reduction in training iterations vs episode-only rewards

**Reward Design**:
- Progress toward solution (correctly positioned cubies)
- Pattern recognition (cross formation, completed layers)
- Move efficiency compared to optimal solve
- Quality of reasoning in "thinking aloud" steps

**Applications**:
- 3D spatial reasoning development
- Multi-step strategic planning
- Structured problem-solving training
- Measurable progress tracking for LLM capabilities

**Demo**: [1-minute demonstration video](https://youtu.be/fi4lhIyF_5M)

**W&B Results**: [Training Dashboard](https://wandb.ai/joshuaxjerin-uc/atropos-environments)

**Requirements**: scipy, matplotlib, torch, transformers, wandb, plotly, flask, pydantic (see requirements.txt)

---

## Support

For questions or issues with community environments:
- Check the individual environment's README first
- Open an issue in the main repository
- Tag the environment author if possible

*These environments are community contributions and may have different maintenance levels and support compared to core Atropos environments.*
