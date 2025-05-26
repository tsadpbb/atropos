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

### 12. Pokemon Showdown Environment (`pokemon-showdown/`)
**Author**: [iyaja](https://github.com/iyaja)
**Purpose**: Train LLMs to play Pokemon battles through strategic decision-making in competitive battles

A game environment that teaches LLMs strategic thinking and decision-making through Pokemon battles using the Pokemon Showdown battle simulator. Models learn to analyze battle states, evaluate team compositions, predict opponent moves, and execute optimal strategies in turn-based combat scenarios.

**Features**:
- **Pokemon Showdown Integration**: Uses the official Pokemon Showdown battle simulator
- **Strategic Decision Making**: Models must choose between attacking, switching, and using items
- **Battle State Analysis**: Complete game state information including HP, status effects, and move sets
- **Self-Play Training**: GPT player vs Max Damage baseline for RL training
- **Random Battle Format**: Uses gen9randombattle for diverse team compositions
- **Real-time Battle Simulation**: Asynchronous battle management with poke-env library

**Training Components**:
- **GPT Player**: LLM-controlled player that receives battle state and must choose actions
- **Max Damage Player**: Baseline opponent that always selects highest damage moves
- **Battle History**: Complete move sequences and outcomes for learning from experience
- **Win/Loss Rewards**: Binary reward signal based on battle outcomes

**Strategic Elements**:
- **Type Effectiveness**: Understanding Pokemon type matchups and damage calculations
- **Status Effects**: Managing poison, burn, paralysis, sleep, and other conditions
- **Team Management**: Switching Pokemon strategically based on matchups
- **Move Selection**: Choosing between different moves based on situation
- **HP Management**: Risk assessment and resource management throughout battles

**Technical Implementation**:
- **Async Battle Management**: Non-blocking battle execution for training efficiency
- **poke-env Integration**: Robust Pokemon battle simulation and state management
- **Atropos RL Framework**: Standard reward signals and trajectory collection
- **Battle Format Support**: Configurable battle formats and rule sets

**Applications**:
- Strategic game AI development
- Turn-based decision making under uncertainty
- Complex state space navigation
- Competitive multi-agent training
- Game theory and opponent modeling

**Demo Resources**:
- **W&B Dashboard**: [Training Results](https://wandb.ai/ajayuppili/atropos-environments_game_environments_pokemon-showdown)
- **Overview Video**: TBD

**Setup Requirements**:
1. Pokemon Showdown server (local installation)
2. poke-env Python library
3. Node.js for Pokemon Showdown simulator
4. OpenAI API access for GPT player

**Battle Format**: gen9randombattle (Generation 9 Random Battles)

**Requirements**: poke-env, nodejs, pokemon-showdown simulator, OpenAI API

### 13. Conversational Style DPO Environment (`conversational_style_dpo/`)
**Author**: [Karthik-Ragunath](https://github.com/Karthik-Ragunath)
**Purpose**: Train LLMs for better conversational style through Direct Preference Optimization (DPO) with chosen/rejected response pairs

A specialized environment for training LLMs to adopt more engaging, empathetic, and helpful conversational styles using Direct Preference Optimization. The environment provides both synthetic and dynamically generated conversation pairs where "chosen" responses are engaging and thoughtful while "rejected" responses are blunt and unhelpful.

**Features**:
- **Two Environment Variants**: Static synthetic data and dynamic prompt generation
- **DPO Training Ready**: Pre-configured tokenization for chosen/rejected response pairs
- **Conversational Style Modeling**: Focus on empathy, engagement, and helpfulness
- **Synthetic Data Generation**: Uses LLMs to create diverse conversational prompts
- **Quality Response Pairs**: Carefully crafted chosen (good) vs rejected (poor) examples

**Environment Variants**:

1. **Static Synthetic Environment** (`conversational_style_dpo_env.py`):
   - Pre-defined conversational prompts with human-crafted response pairs
   - Focus on emotional support, explanations, excitement sharing, and help-seeking
   - Immediate training readiness without LLM dependencies

2. **Dynamic GSM8K-Style Environment** (`gsmk8k_conversational_style_dpo_env.py`):
   - LLM-generated conversational prompts for diverse training data
   - Real-time chosen/rejected response generation with different system prompts
   - Scalable dataset creation with fallback to static prompts

**Conversation Categories**:
- **Emotional Support**: Responding to feelings and personal sharing
- **Educational**: Explaining concepts clearly and engagingly
- **Enthusiasm Sharing**: Celebrating user excitement and interests
- **Help & Guidance**: Providing assistance with understanding problems
- **General Conversation**: Weather, casual topics, and everyday interactions

**Response Quality Characteristics**:
- **Chosen Responses**: Empathetic, engaging, ask follow-up questions, provide detailed explanations
- **Rejected Responses**: Blunt, minimal, dismissive, unhelpful

**Example Training Pair**:
```
Prompt: "I'm feeling a bit down today."
Chosen: "I'm sorry to hear that. Sometimes a little self-care can help. What's one small thing you could do for yourself right now?"
Rejected: "Okay."
```

**Technical Implementation**:
- **DPO Tokenization**: Ready-to-use tokenization for preference optimization
- **Configurable Parameters**: Temperature, max tokens, and dataset size controls
- **Modular Design**: Easy to extend with new conversation types
- **W&B Integration**: Comprehensive logging and experiment tracking

**Training Applications**:
- Customer service AI improvement
- Therapeutic chatbot development
- Educational AI tutoring systems
- General conversational AI enhancement
- Empathy and engagement training

**Configuration Options**:
- `chosen_temperature`: Temperature for generating engaging responses (default: 0.7)
- `rejected_temperature`: Temperature for generating blunt responses (default: 0.4)
- `shuffle_dataset`: Whether to randomize training order
- `data_path_to_save_groups`: Optional path for saving training artifacts

**Data Artifacts**:
- Archived training examples and HTML visualizations available (see `conversational_style_dpo_artifacts.zip`)
- Ready for upload to Hugging Face for community access

**Requirements**: Standard Atropos dependencies, transformers, torch

### 14. Solitaire Winning Probability Environment (`solitaire_winning_probability/`)
**Author**: [davidedipeppe](https://github.com/davidedipeppe)
**Purpose**: Train LLMs to analyze and predict winning probabilities in solitaire-style card games using both theoretical mathematics and empirical simulation

A sophisticated environment that combines AI-powered probability analysis with Monte Carlo simulation to teach LLMs mathematical reasoning about game theory and probability. Models learn to derive mathematical formulas for exact probability calculations and validate their theoretical predictions through empirical simulation.

**Features**:
- **Dual Analysis Approach**: Both theoretical mathematical formulas and empirical Monte Carlo simulation
- **AI Formula Derivation**: LLMs analyze game mechanics to derive exact probability formulas
- **Mathematical Expression Evaluation**: Supports factorials, combinations, permutations, and standard operations
- **Simulation Verification**: Runs thousands of game simulations to verify theoretical predictions
- **QA Dataset Generation**: Creates training data for AI models by generating question-answer pairs
- **Sophisticated Reward Function**: Evaluates prediction quality with relative error calculation and length penalties

**Game Types Included**:
- **Easy Probability Games**: Simple card draws and dice rolls (1/4, 1/6, 1/4 probabilities)
- **Card Matching Games**: Avoid counter-card matches with cycling counters (1-4 cycles)
- **Odd Card Game**: Draw odd-valued cards from standard deck (7/13 probability)
- **Extensible Framework**: Easy to add new solitaire game variants

**Mathematical Framework**:
- **Formula Notation**: Supports `C(n,r)` combinations, `P(n,r)` permutations, `factorial(n)`
- **Expression Parser**: Safe mathematical expression evaluation with asteval
- **Probability Comparison**: Measures theoretical vs empirical accuracy
- **Error Analysis**: Quantifies prediction quality with relative error metrics

**Reward System Design**:
1. **Base Reward**: `1 - min(abs(gt - predicted) / gt, 2)` with 0.2 bonus for valid predictions
2. **Length Penalty**: Applied to responses exceeding 50% of max token length
3. **Validation Checks**: Ensures proper formula formatting and mathematical syntax
4. **Quality Metrics**: Tracks prediction accuracy and response efficiency

**Training Components**:
- **Game Predictor Class**: Core AI analysis and formula evaluation engine
- **Simulation Engine**: Monte Carlo verification with configurable iteration counts
- **Mathematical Evaluator**: Safe expression parsing and computation
- **QA Data Generator**: Automated training dataset creation

**Example Training Flow**:
```
Game: Draw from [1,2,3,4], win if card is 1
AI Analysis: "1 favorable outcome out of 4 total..."
Formula: "1/4"
Calculated: 0.25
Simulated: 0.2499 (100k runs)
Reward: High (excellent theoretical-empirical match)
```

**Applications**:
- **Probability Theory Education**: Practical demonstration of theoretical concepts
- **Mathematical Reasoning Training**: Formula derivation and validation skills
- **Game Analysis Research**: Framework for analyzing card game mechanics
- **AI Math Capabilities**: Training models in structured mathematical thinking

**Technical Implementation**:
- **AsyncOpenAI Integration**: Efficient AI analysis with configurable models
- **CSV Data Management**: Structured question-answer pair storage
- **Comprehensive Error Handling**: Robust formula evaluation and validation
- **Performance Tracking**: Detailed analysis results and comparison metrics

**Quality Assessment**:
- **Excellent Match**: < 1% difference between theory and simulation
- **Good Match**: < 5% difference
- **Fair Match**: < 10% difference
- **Poor Match**: > 10% difference

**Configuration Options**:
- Simulation count (default: 100,000 runs)
- Model selection for AI analysis
- Token length limits and penalties
- Mathematical expression validation rules

**Requirements**: asyncio, openai, asteval, csv, datasets, math_verify, latex2sympy2_extended

### 15. Lean Theorem Proving Environment (`lean_proof_env/`)
**Author**: [justin5764](https://github.com/justin5764)
**Purpose**: Train LLMs to complete formal mathematical proofs in the Lean theorem proving language using compilation feedback

A comprehensive environment for training language models on formal mathematical reasoning through Lean theorem proving. Models learn to complete theorem statements by replacing `sorry` placeholders with valid proof steps, receiving immediate feedback through Lean compilation checks.

**Features**:
- **Formal Proof Completion**: LLMs complete theorem statements by replacing `sorry` with valid proofs
- **Lean 4 Integration**: Uses the modern Lean 4 theorem proving language and Mathlib
- **Compilation Feedback**: Real-time validation through Lean compiler integration (PyPantograph)
- **Mathematical Dataset**: Built on `brando/minif2f-lean4` Hugging Face dataset
- **Structured Training**: Separate validation/test splits for robust evaluation
- **Mock Compilation**: Includes simulation framework for development without full Lean setup

**Training Components**:
- **Problem Structure**: Import statements + formal theorem statement with `sorry`
- **Proof Generation**: LLM generates complete theorem blocks with proof steps
- **Compilation Validation**: Lean compiler checks proof correctness and syntax
- **Reward System**: Binary rewards (1.0 for compilation success, -1.0 for failure)
- **Progress Tracking**: Compilation success rates and detailed attempt logging

**Lean Integration**:
- **PyPantograph Interface**: Async integration with Lean theorem prover
- **Import Management**: Handles Mathlib imports and namespace declarations
- **Syntax Validation**: Ensures generated proofs follow Lean syntax rules
- **Error Reporting**: Detailed compilation error messages for debugging

**Dataset Features**:
- **MiniF2F-Lean4**: Curated collection of formal mathematics problems
- **Problem Diversity**: Covers various mathematical domains and difficulty levels
- **Structured Format**: Consistent header + formal statement organization
- **Train/Test Splits**: Uses validation split for training, test split for evaluation

**Example Training Flow**:
```
Input: "import Mathlib.Data.Nat.Basic\nopen Nat\n\ntheorem add_comm (a b : nat) : a + b = b + a := sorry"
LLM Output: "theorem add_comm (a b : nat) : a + b = b + a := by rw [Nat.add_comm]"
Compilation: Success ✓
Reward: 1.0
```

**Mock Development Mode**:
- **Simulation Framework**: Allows development without full Lean installation
- **Keyword-Based Validation**: Basic proof structure and content checks
- **Random Compilation**: Configurable success rates for testing
- **Error Simulation**: Realistic error messages for training

**WandB Integration**:
- **Compilation Metrics**: Track success rates during training and evaluation
- **Proof Attempt Tables**: Detailed logs of problem statements, generated proofs, and outcomes
- **Progress Visualization**: Training curves and performance analytics
- **Custom Metrics**: `train/batch_avg_percent_compiled` and `eval/percent_compiled`

**Training Applications**:
- **Formal Verification**: Training models for software and hardware verification
- **Mathematical Education**: AI tutoring systems for formal mathematics
- **Proof Assistant Development**: Improving automated theorem proving tools
- **Research Acceleration**: Automating routine mathematical proofs

**Technical Implementation**:
- **Async Architecture**: Non-blocking proof compilation and validation
- **Temperature Control**: Different settings for training diversity vs evaluation consistency
- **Token Management**: Configurable proof length limits and generation parameters
- **Error Handling**: Robust handling of compilation failures and edge cases

**Configuration Options**:
- **Model Selection**: Configurable LLM for proof generation (default: Qwen/Qwen3-235B-A22B)
- **Group Size**: Number of proof attempts per problem (default: 4)
- **Evaluation Frequency**: Steps between evaluation runs (default: 50)
- **Token Limits**: Maximum proof length (default: 1024 tokens)
- **Testing Mode**: Reduced dataset size for development

**Quality Metrics**:
- **Compilation Success Rate**: Primary measure of proof correctness
- **Proof Efficiency**: Token usage and proof length analysis
- **Error Pattern Analysis**: Common failure modes and improvement areas
- **Mathematical Coverage**: Breadth of successfully proven theorems

**Setup Requirements**:
1. Lean 4 installation with Mathlib
2. PyPantograph for Python-Lean integration
3. `brando/minif2f-lean4` dataset access
4. OpenAI-compatible LLM server

**Command Line Usage**:
```bash
# Connect to Atropos trainer
python environments/community/lean_proof_env/lean_env.py serve

# Local testing and development
python environments/community/lean_proof_env/lean_env.py process
```

**Requirements**: datasets, tqdm, wandb, PyPantograph (for full Lean integration), asyncio

### 16. DeepSacrifice - Human-in-the-Loop Chess RL Environment (`deepsacrifice_chess/`)
**Author**: [metonym](https://github.com/metonym)
**Purpose**: Train chess agents to play aggressive, sacrificial chess through human-in-the-loop reinforcement learning with LLM-based reward modeling

A unique chess environment that combines human gameplay with LLM evaluation to train agents in aggressive, sacrificial chess styles. The environment creates a reinforcement learning loop where the agent learns from direct human-vs-agent games, receiving dense feedback from language models that evaluate moves for aggression, brilliance, and sacrifice justification.

**Features**:
- **Human-in-the-Loop RL**: Users serve as the environment, directly playing against the agent
- **LLM-Based Reward Model**: GPT-4 evaluates trajectories for aggression, brilliance, and sacrifice quality
- **Aggressive Chess Focus**: Agent specifically trained to prioritize attacking, sacrificial play styles
- **Real-time Web Interface**: React-based chess board with live game interaction
- **Dense Feedback System**: Move-by-move scoring replaces sparse win/loss rewards
- **Policy Adaptation**: Agent adjusts strategy based on post-game LLM evaluations

**Core RL Components**:
- **State**: Chess board position (FEN notation) at each move
- **Action**: Legal chess moves by the agent (SAN notation)
- **Trajectory**: Complete game history of states and agent actions
- **Reward**: LLM-generated scores for aggression, brilliance, and game outcome
- **Policy**: Move selection logic with aggression weighting and sacrifice prioritization
- **Environment**: Human player interaction and game management system

**Training Flow**:
1. **Game Execution**: Agent and human alternate moves in chess environment
2. **Trajectory Recording**: Log complete sequence of FENs and agent moves
3. **LLM Evaluation**: Post-game analysis by GPT-4 for move quality assessment
4. **Reward Computation**: Aggregate LLM scores into scalar reward signal
5. **Policy Update**: Adjust agent parameters based on feedback (aggression threshold, sacrifice prioritization)
6. **Next Episode**: Updated policy used in subsequent games

**LLM Evaluation Criteria**:
- **Aggression Score**: 1-10 rating for move aggressiveness and attacking intent
- **Brilliance Assessment**: Evaluation of tactical creativity and unexpected moves
- **Sacrifice Justification**: Analysis of whether material sacrifices are strategically sound
- **Game Outcome Integration**: Win/loss results combined with style evaluation

**Agent Strategy**:
- **Capture Preference**: Prioritizes taking opponent pieces when available
- **Check Generation**: Seeks moves that put opponent king in check
- **Sacrifice Evaluation**: Learns to assess when material sacrifice leads to positional advantage
- **Adaptive Learning**: Adjusts aggression based on success rates against human opponents

**Technical Architecture**:
- **Frontend**: React + TypeScript with Vite build system
- **Backend**: Bun runtime with TypeScript API server
- **Chess Engine**: chess.js library for move validation and game state
- **LLM Integration**: OpenAI API for post-game move evaluation
- **Real-time Communication**: REST API for move exchange and game state updates

**Web Interface Features**:
- **Interactive Chess Board**: Visual board with drag-and-drop move input
- **Live Game State**: Real-time position updates and move history
- **LLM Feedback Display**: Post-game analysis with move-by-move scores
- **Game History**: Complete trajectory logging for analysis
- **Agent Learning Visualization**: Policy update tracking over time

**Example Training Session**:
```
Game 1: Agent plays aggressively, sacrifices queen for checkmate threat
LLM Evaluation: High aggression (9/10), brilliant sacrifice (justified)
Reward: +0.85 (high positive feedback)
Policy Update: Increase sacrifice threshold, maintain aggression weighting

Game 2: Agent makes conservative moves, wins material but loses initiative
LLM Evaluation: Low aggression (3/10), missed tactical opportunities
Reward: +0.15 (low positive feedback despite win)
Policy Update: Decrease conservative play, increase attacking move priority
```

**Applications**:
- **Chess AI Development**: Training agents for specific playing styles
- **Human-AI Interaction Research**: Studying adaptive learning from human feedback
- **Game Theory Analysis**: Understanding sacrifice and risk-taking in competitive games
- **Educational Chess Tools**: Teaching aggressive chess principles through AI demonstration
- **Reinforcement Learning Research**: Human-in-the-loop RL methodology development

**Setup Requirements**:
1. **Bun Runtime**: Modern JavaScript runtime and package manager
2. **OpenAI API Key**: For LLM-based move evaluation
3. **Web Browser**: For interactive chess interface
4. **Node.js Environment**: For development and build tools

**Installation & Usage**:
```bash
# Environment setup
cp .env.template .env
# Add OpenAI API key to .env file

# Install dependencies
bun install

# Run frontend (Terminal 1)
bun dev

# Run backend (Terminal 2)
bun dev:server
```

**Development Status**: Design prototype focusing on RL loop structure and LLM integration. Core learning algorithms are placeholder implementations ready for enhancement.

**Future Enhancements**:
- Advanced policy gradient methods for agent learning
- Multi-agent training with different chess styles
- Tournament mode for agent evaluation
- Chess engine integration for stronger baseline opponents
- Detailed analytics dashboard for training progress

**Requirements**: Bun runtime, OpenAI API, React, TypeScript, chess.js, Vite

### 17. Caput Mundi - Six-Seat No-Limit Hold'em Poker Environment (`poker_holdem/`)
**Author**: [yoniebans](https://github.com/yoniebans)
**Purpose**: Train language models to make optimal poker decisions through reinforcement learning on expert hand history data

A comprehensive poker training environment that teaches LLMs to play No-Limit Hold'em poker like winning players. The environment uses processed hand histories from successful poker players to create a supervised learning framework where models learn to match expert actions in various game situations.

**Features**:
- **Expert Hand History Training**: Uses curated dataset of winning player decisions
- **Multi-Stage Game Analysis**: Separate tracking for preflop, flop, turn, and river decisions
- **Dual Reward System**: Combined action matching and bet sizing evaluation
- **Comprehensive Evaluation**: Stage-specific performance metrics and cumulative tracking
- **HuggingFace Integration**: Direct dataset loading with train/test splits
- **WandB Monitoring**: Detailed logging of training progress and poker-specific metrics

**Core Training Components**:
- **Dataset**: `yoniebans/6max-nlh-poker` with formatted poker situations and expert actions
- **Input Format**: Structured poker prompts with game state, player positions, and betting history
- **Target Actions**: Expert player decisions including action type and bet sizing
- **Reward Functions**: Specialized evaluation for poker action correctness and bet precision
- **Evaluation Metrics**: Accuracy tracking by game stage and action distribution analysis

**Poker-Specific Features**:
- **Game Stage Tracking**: Separate analysis for preflop, flop, turn, and river decisions
- **Action Type Recognition**: Fold, check, call, bet, raise, re-raise, all-in classification
- **Bet Sizing Analysis**: Numerical precision evaluation for betting amounts
- **Position Awareness**: Training on positional play and strategic considerations
- **Hand History Format**: Realistic poker situation representation

**Reward System Architecture**:
- **Action Match Reward (60%)**: Evaluates correctness of chosen action type
  - Exact match: 1.0 score
  - Action type match: 0.7 score
  - Strategic intent match: 0.5 score
- **Bet Sizing Reward (40%)**: Evaluates precision of bet amount
  - Perfect amount: 1.0 score
  - Linear decay with deviation
  - Zero score beyond 50% deviation

**Training Data Structure**:
```
Input: "Position: BTN, Stack: 100bb, Pot: 3bb, Action: Hero faces 2bb raise..."
Expert Action: "call 2"
Model Output: "call 2.5"
Action Score: 0.7 (correct action type)
Sizing Score: 0.8 (close bet amount)
Combined Score: 0.74
```

**Evaluation Framework**:
- **Stage-Specific Metrics**: Separate accuracy tracking for each betting round
- **Action Distribution**: Monitoring of fold/call/raise frequencies
- **Cumulative Performance**: Long-term learning progress across training epochs
- **Threshold-Based Accuracy**: Configurable correctness thresholds for evaluation
- **Sample-Based Testing**: Efficient evaluation on dataset subsets

**Dataset Features**:
- **Six-Max Format**: Optimized for 6-player No-Limit Hold'em games
- **Winning Player Focus**: Hand histories from profitable poker players
- **Structured Prompts**: Consistent formatting for game state representation
- **Action Formatting**: Standardized expert action representation
- **Train/Test Splits**: Proper data separation for training and evaluation

**WandB Integration**:
- **Training Metrics**: Epoch tracking, stage-specific scores, action distributions
- **Evaluation Tracking**: Cumulative accuracy, stage performance, threshold analysis
- **Poker Analytics**: Action frequency analysis, betting pattern recognition
- **Progress Visualization**: Learning curves and performance trends

**Example Training Flow**:
```
Epoch 1: Load shuffled hand histories
Hand 1: Preflop decision - Model matches expert fold (Score: 1.0)
Hand 2: Flop decision - Model bets 8bb vs expert 10bb (Score: 0.85)
Hand 3: River decision - Model calls vs expert raise (Score: 0.5)
Evaluation: 73% accuracy across all stages
```

**Configuration Options**:
- **Model Selection**: Configurable LLM for poker decision making (default: Qwen/Qwen3-1.7B)
- **Batch Processing**: Group size and batch size for efficient training
- **Evaluation Parameters**: Sample size, temperature, and correctness thresholds
- **Reward Weighting**: Adjustable balance between action matching and bet sizing
- **Dataset Management**: Epoch-based shuffling and queue management

**Applications**:
- **Poker AI Development**: Training competitive poker playing agents
- **Decision Making Research**: Understanding strategic reasoning in uncertain environments
- **Game Theory Applications**: Learning optimal play in multi-agent competitive settings
- **Financial Modeling**: Risk assessment and decision making under uncertainty
- **Educational Tools**: Teaching poker strategy through AI demonstration

**Technical Implementation**:
- **Async Processing**: Non-blocking dataset loading and model inference
- **Memory Efficient**: Queue-based training data management
- **Robust Parsing**: Action extraction from natural language responses
- **Error Handling**: Graceful handling of malformed model outputs
- **Scalable Architecture**: Support for large-scale poker dataset training

**Performance Metrics**:
- **Overall Accuracy**: Primary measure of poker decision quality
- **Stage Accuracy**: Preflop/flop/turn/river specific performance
- **Action Distribution**: Frequency analysis of different poker actions
- **Bet Sizing Precision**: Numerical accuracy in betting decisions
- **Learning Progress**: Improvement tracking across training epochs

**Setup Requirements**:
1. HuggingFace Datasets library for data loading
2. Transformers library for tokenization
3. OpenAI-compatible LLM server for inference
4. WandB account for training monitoring

**Command Line Usage**:
```bash
# Start VLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-1.7B \
    --gpu-memory-utilization 0.95 \
    --dtype auto \
    --port 9002

# Run poker training environment
python environments/community/poker_holdem/poker_env.py process \
    --env.data_path_to_save_groups poker_rollouts.jsonl \
    --openai.base_url http://localhost:9002/v1 \
    --openai.api_key EMPTY \
    --openai.model_name Qwen/Qwen3-1.7B
```

**Data Pipeline**: Custom data processing pipeline available at [poker-rl-data](https://github.com/yoniebans/poker-rl-data) for creating poker training datasets from raw hand histories.

**Requirements**: datasets, transformers, wandb, atroposlib

---

## Support

For questions or issues with community environments:
- Check the individual environment's README first
- Open an issue in the main repository
- Tag the environment author if possible

*These environments are community contributions and may have different maintenance levels and support compared to core Atropos environments.*
