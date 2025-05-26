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

### 18. Quantum-Classical Hybrid Language Model Environment (`quantum_hybrid/`)
**Author**: [jeannemtl](https://github.com/jeannemtl)
**Purpose**: Train quantum-enhanced language models by combining classical transformers with quantum circuits using PennyLane and PyTorch

A novel environment that implements quantum-classical hybrid architecture for next-word prediction, trained on high-quality text generated by Hermes-3-70B. The key innovation is using quantum circuits to enhance traditional neural networks for language modeling tasks, exploring potential quantum advantages in natural language processing.

**Research Question**: Can quantum circuits provide advantages over purely classical approaches in natural language processing tasks?

**Architecture Overview**:
- **Data Flow**: Input Prompts → Hermes-3-70B (text generation) → Hybrid Model Training → Quantum-Enhanced Predictions
- **Hybrid Model Components**:
  - **Classical Pathway**: Standard transformer-style neural network head
  - **Quantum Pathway**: Dimensionality reduction (768D → 8D) → Two quantum circuit layers → Quantum-to-vocabulary mapping
  - **Learnable Mixing**: Parameter α balances classical vs quantum contributions

**Quantum Circuit Design**:
- **8 qubits with 3 parameterized layers**
- **RY rotation gates** for classical data encoding
- **CNOT gates** creating entanglement patterns
- **Pauli-Z measurements** for classical output extraction
- **Ring topology** for full qubit connectivity

**Dual Implementation Approach**:
The environment includes two complementary implementations:

**1. Optimized Hybrid Model (`atropos.py`)**:
- **Synthetic Training**: Uses simplified tokenizer and mock hidden states for rapid experimentation
- **Quantum Integration**: Full quantum circuit implementation with PennyLane
- **Hybrid Architecture**: Learnable mixing between classical and quantum pathways
- **Training Loop**: Direct optimization of quantum parameters via gradient descent
- **Evaluation**: Entropy-based comparison of hybrid vs classical predictions

**2. Dataset-Driven Training (`atopos_quant.py`)**:
- **Real Data Processing**: Uses WikiText dataset with HuggingFace integration
- **Quantum Text Analysis**: Standalone quantum analyzer for text coherence measurement
- **Server Integration**: Compatible with Atropos server infrastructure
- **Comprehensive Metrics**: Perplexity, quantum coherence, and combined scoring
- **Production Ready**: Full tokenization and dataset management

**Quantum Text Analysis Features**:
- **Text Feature Extraction**: Length, word count, character diversity, punctuation patterns
- **Quantum Encoding**: Features mapped to quantum states via rotation gates
- **Entanglement Patterns**: Complex qubit interactions for linguistic analysis
- **Coherence Measurement**: Quantum variance as text quality indicator
- **Fallback Mechanisms**: Graceful degradation when quantum circuits fail

**Training Strategy - Quantum-Enhanced Knowledge Distillation**:
1. **Teacher Model**: Hermes-3-70B generates diverse, high-quality responses
2. **Student Model**: Hybrid quantum-classical model learns next-word prediction
3. **Comparison**: Direct evaluation of quantum vs classical pathways within same model
4. **Optimization**: Both classical and quantum parameters trained via gradient descent

**Key Metrics & Evaluation**:

**Training Metrics**:
- `train/hybrid_loss`: Combined quantum-classical model loss
- `train/classical_loss`: Baseline classical-only model loss
- `train/quantum_loss`: Quantum-specific loss component
- `train/alpha_value`: Mixing parameter (0 = full quantum, 1 = full classical)

**Evaluation Metrics**:
- `eval/hybrid_performance`: Entropy-based comparison of hybrid vs classical predictions
- `eval/quantum_weight`: Current quantum contribution (1 - α)
- `train/quantum_coherence`: Measure of quantum circuit effectiveness

**Model Metrics**:
- `model/alpha`: Real-time mixing parameter
- `model/quantum_contribution`: Percentage of quantum influence

**Interpretation Guide**:
- **Decreasing hybrid_loss**: Model improving at next-word prediction
- **Stable alpha_value**: Balanced classical-quantum integration
- **High quantum_coherence**: Quantum circuits contributing meaningfully
- **hybrid_performance > 0.5**: Quantum enhancement provides benefits

**Technical Implementation Details**:

**Quantum Circuit Architecture**:
```python
# Data encoding
qml.RY(classical_data, wires=qubit)

# Parameterized layers
for layer in range(n_layers):
    for qubit in range(n_qubits):
        qml.RY(learnable_params[layer, qubit], wires=qubit)

    # Entanglement pattern
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[n_qubits - 1, 0])  # Ring topology

# Measurement
[qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
```

**Training Process**:
1. **Forward Pass**: Hidden states → quantum circuits → predictions
2. **Loss Calculation**: Cross-entropy on next-word prediction
3. **Backpropagation**: Gradients through quantum circuits via parameter-shift rule
4. **Optimization**: Adam optimizer updates both classical and quantum parameters

**Novel Contributions**:
- **First quantum-enhanced Atropos environment**
- **Hybrid architecture balancing quantum and classical processing**
- **Knowledge distillation from large classical models to small quantum models**
- **Quantum-aware evaluation metrics for NLP tasks**

**Current Limitations**:
- **Simulated Quantum**: Uses classical simulation (no quantum hardware)
- **Synthetic Features**: Uses random hidden states (not real text embeddings in optimized version)
- **Scale**: Limited to 8 qubits due to exponential simulation cost
- **Evaluation**: Simple entropy comparison (more sophisticated metrics possible)

**Potential Applications**:
- **Quantum NLP Research**: Differentiable quantum circuits for language tasks
- **Hybrid Model Architectures**: Resource-constrained environments with quantum enhancement
- **Novel Optimization**: Combining classical and quantum approaches
- **Benchmark Creation**: Quantum machine learning evaluation in language tasks

**Future Research Directions**:

**Immediate Improvements**:
- **Real Text Processing**: Replace synthetic hidden states with actual transformer embeddings
- **Advanced Quantum Circuits**: Implement quantum attention mechanisms
- **Scaling Studies**: Investigate qubit count vs performance relationships

**Long-term Goals**:
- **Quantum Hardware**: Deploy on IBM Quantum, IonQ, or other quantum computers
- **Larger Models**: Scale to 100+ qubit systems when available
- **Quantum Advantage**: Identify specific NLP tasks where quantum provides provable benefits
- **Production Systems**: Develop practical quantum-enhanced language models

**Configuration Options**:
- **Quantum Parameters**: Configurable qubit count (default: 8) and layer depth (default: 3)
- **Training Settings**: Learning rate, batch size, total steps, evaluation frequency
- **Model Architecture**: Base model selection, vocabulary size, hidden dimensions
- **Hybrid Weighting**: Adjustable balance between classical and quantum contributions
- **Dataset Selection**: WikiText variants or custom text datasets

**Setup Requirements**:
1. **PennyLane**: Quantum computing framework
2. **PyTorch**: Deep learning and automatic differentiation
3. **Transformers**: Tokenization and model utilities
4. **Datasets**: HuggingFace dataset loading
5. **NumPy**: Numerical computations
6. **WandB**: Experiment tracking and visualization

**Installation & Usage**:
```bash
# Install quantum dependencies
pip install pennylane torch transformers datasets numpy wandb

# Run optimized hybrid training
python environments/community/quantum_hybrid/atropos.py process \
    --env.n_qubits 8 \
    --env.n_layers 3 \
    --env.total_steps 50 \
    --env.quantum_weight 0.3

# Run dataset-driven training
python environments/community/quantum_hybrid/atopos_quant.py process \
    --env.dataset_name wikitext \
    --env.dataset_config wikitext-2-raw-v1 \
    --env.n_qubits 8
```

**Live Experiment Tracking**: Monitor training progress and quantum metrics at WandB dashboard with real-time visualization of quantum-classical balance and performance metrics.

**Research Impact**: This environment represents cutting-edge research in quantum machine learning for NLP. While quantum advantages are still under investigation, the framework provides a foundation for future breakthroughs in quantum-enhanced language processing.

**Repository Structure**:
```
environments/community/quantum_hybrid/
├── atropos.py                    # Optimized hybrid model implementation
├── atopos_quant.py              # Dataset-driven quantum training
├── requirements.txt             # Python dependencies
├── README.md                    # Detailed documentation
├── quantum_hybrid_artifacts.tar.gz  # Training artifacts
└── quantum_latest_artifacts.tar.gz  # Latest training data
```

**Requirements**: pennylane, torch, transformers, datasets, numpy, pydantic, atroposlib

### 19. PyTorch Optimizer Coding Environment (`pytorch_optimizer_coding/`)
**Author**: [arihanv](https://github.com/arihanv)
**Purpose**: Train code-generating agents to design and evaluate custom PyTorch optimizers through automated compilation, novelty assessment, and performance benchmarking

A comprehensive RL environment that enables language models to explore the optimizer design space by generating PyTorch optimizer code, which is then evaluated using a multi-faceted reward system combining compilation success, novelty scoring, and performance benchmarking on neural network training tasks.

**Research Question**: Can LLM coding agents automatically discover novel and effective PyTorch optimizers that outperform hand-designed alternatives?

**Environment Architecture**:
- **Agent Action**: Generate PyTorch optimizer source code as string output
- **Compilation Reward**: Sandboxed execution with Modal Labs for safe code evaluation
- **Novelty Assessment**: Grok API scoring for optimizer innovation (0-10 scale)
- **Performance Benchmarking**: Automated training on MLP/CNN/Transformer architectures
- **Composite Scoring**: Multi-dimensional reward combining all evaluation aspects

**Core Components**:

**1. Code Generation Interface (`optimizer_benchmark_environmenr.py`)**:
- **BaseEnv Integration**: Full compatibility with Atropos framework
- **Architecture Selection**: Configurable target architectures (mnist, classification_small, tabular)
- **Evaluation Pipeline**: Automated scoring through wrapper functions
- **Error Handling**: Graceful failure management for invalid code

**2. Sandboxed Execution System (`deploy.py`)**:
- **Modal Labs Integration**: Secure cloud-based code execution
- **Isolation**: Complete separation from host environment
- **Dependency Management**: Automatic PyTorch and related library installation
- **Output Capture**: Comprehensive stdout/stderr logging for debugging

**3. Multi-Dimensional Evaluation (`evaluator.py`)**:
- **Validity Pipeline**: Expert code validator using Grok-3-Latest
- **Novelty Pipeline**: Research conference-style novelty assessment
- **Compilation Checking**: Syntax, runtime, and compatibility validation
- **Scoring Aggregation**: MaxPool aggregation across multiple evaluation runs

**4. Performance Benchmarking (`FOB/`)**:
- **Framework for Optimizer Benchmarking (FOB)**: Comprehensive optimizer evaluation suite
- **Multi-Task Evaluation**: MNIST, classification, and tabular regression tasks
- **Automated Training**: 2-epoch training runs with performance metrics
- **Time Tracking**: Training duration measurement for efficiency assessment
- **Metric Collection**: Accuracy, loss, and convergence rate analysis

**Evaluation Pipeline**:

**Stage 1: Code Validation**
```python
# Validity criteria (all must pass):
1. Zero syntax or runtime errors
2. No undefined variables or type mismatches
3. No memory or CUDA/CPU compatibility issues
4. Successful import and instantiation
5. Complete optimization step execution
```

**Stage 2: Novelty Assessment**
```python
# Grok-3 evaluation prompt:
"You are a judge expert at evaluating optimizers for novelty
as they will be accepted to a prestigious research conference.
Rate on scale 1-10 based on novelty and impact in speeding up training."
```

**Stage 3: Performance Benchmarking**
```python
# FOB evaluation tasks:
- MNIST: Image classification (accuracy maximization)
- Classification Small: General classification (accuracy maximization)
- Tabular: Regression tasks (loss minimization)
```

**Reward Function Design**:
```python
total_reward = compilation_reward + novelty_score + performance_reward
where:
- compilation_reward: 1 if compiles successfully, 0 otherwise
- novelty_score: Grok assessment (0-10 scale)
- performance_reward: Task-specific metrics (accuracy/loss) - time_penalty
```

**FOB Integration Features**:

**Automated Optimizer Registration**:
- **Dynamic Code Injection**: Runtime optimizer.py file creation
- **Configuration Generation**: Automatic default.yaml creation with learning rates
- **Module Initialization**: Proper Python package structure setup
- **Experiment YAML**: Multi-task evaluation configuration

**Benchmarking Tasks**:
- **MNIST**: Handwritten digit classification (28x28 images, 10 classes)
- **Classification Small**: Reduced-scale classification for rapid evaluation
- **Tabular**: Regression on structured data with numerical features

**Performance Metrics**:
- **Training Time**: Wall-clock time for 2-epoch training
- **Final Accuracy**: Test set performance after training completion
- **Loss Convergence**: Final loss values for regression tasks
- **Efficiency Ratio**: Performance per unit time for optimizer comparison

**Technical Implementation**:

**Optimizer Code Template**:
```python
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import SGD  # or custom optimizer
from pytorch_fob.engine.parameter_groups import GroupedModel
from pytorch_fob.engine.configs import OptimizerConfig

def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    optimizer = CustomOptimizer(model.grouped_parameters(lr=lr), lr=lr)
    return {"optimizer": optimizer}
```

**Modal Labs Deployment**:
- **Serverless Execution**: On-demand code execution without infrastructure management
- **Automatic Scaling**: Dynamic resource allocation based on evaluation load
- **Security Isolation**: Complete separation from host environment
- **Dependency Injection**: Automatic PyTorch and scientific computing stack

**Grok API Integration**:
- **Verdict Framework**: Structured evaluation pipeline with retry mechanisms
- **Multi-Run Assessment**: 3 independent evaluations with MaxPool aggregation
- **Prompt Engineering**: Research conference review simulation for novelty assessment
- **Categorical Validation**: Binary valid/invalid classification with strict criteria

**Environment Configuration**:
```python
class OptimizerBenchmarkEnvConfig(BaseEnvConfig):
    architecture: str = "mnist"  # Target architecture for evaluation
    max_epochs: int = 2         # Training duration
    timeout: int = 300          # Maximum evaluation time
    novelty_threshold: float = 7.0  # Minimum novelty for acceptance
```

**Evaluation Workflow**:
1. **Code Generation**: Agent produces optimizer implementation
2. **Syntax Validation**: Pre-execution syntax and import checking
3. **Sandboxed Execution**: Modal Labs deployment and execution
4. **Compilation Assessment**: Success/failure determination
5. **Novelty Scoring**: Grok API evaluation for innovation
6. **Performance Testing**: FOB benchmark execution
7. **Reward Calculation**: Multi-dimensional scoring aggregation
8. **Feedback Provision**: Detailed results for agent learning

**Safety & Security Features**:
- **Sandboxed Execution**: Complete isolation from host system
- **Resource Limits**: CPU, memory, and time constraints
- **Code Validation**: Pre-execution safety checks
- **Error Containment**: Graceful handling of malicious or broken code
- **Audit Logging**: Comprehensive execution tracking

**Research Applications**:

**Optimizer Discovery**:
- **Novel Architectures**: Automatic discovery of new optimizer designs
- **Hyperparameter Optimization**: Learning rate, momentum, and decay schedules
- **Adaptive Methods**: Dynamic adjustment based on training progress
- **Task-Specific Optimization**: Specialized optimizers for different domains

**Meta-Learning**:
- **Cross-Task Transfer**: Optimizers effective across multiple domains
- **Few-Shot Adaptation**: Quick adaptation to new tasks
- **Architecture Awareness**: Optimizers tailored to specific model architectures
- **Efficiency Optimization**: Balancing performance with computational cost

**Automated ML Pipeline**:
- **End-to-End Optimization**: From code generation to performance validation
- **Continuous Improvement**: Iterative refinement based on evaluation feedback
- **Scalable Evaluation**: Parallel assessment across multiple architectures
- **Production Integration**: Direct deployment of discovered optimizers

**Current Limitations**:
- **Evaluation Scope**: Limited to 2-epoch training (rapid but potentially incomplete assessment)
- **Architecture Coverage**: Three tasks may not capture full optimizer effectiveness
- **Novelty Subjectivity**: Grok assessment may have biases or inconsistencies
- **Computational Cost**: Modal Labs execution adds latency and expense

**Future Enhancements**:

**Extended Evaluation**:
- **Longer Training**: Multi-epoch evaluation for convergence analysis
- **More Tasks**: Computer vision, NLP, and reinforcement learning benchmarks
- **Real-World Datasets**: ImageNet, GLUE, and other standard benchmarks
- **Hardware Diversity**: GPU, TPU, and distributed training evaluation

**Advanced Metrics**:
- **Convergence Analysis**: Learning curve shape and stability assessment
- **Generalization**: Performance on held-out validation sets
- **Robustness**: Sensitivity to hyperparameter changes
- **Memory Efficiency**: RAM and computational resource utilization

**Agent Integration**:
- **Curriculum Learning**: Progressive difficulty in optimizer design challenges
- **Multi-Agent Competition**: Competitive optimizer discovery
- **Human-in-the-Loop**: Expert feedback integration for novelty assessment
- **Transfer Learning**: Knowledge sharing across related optimization tasks

**Installation & Setup**:
```bash
# Install core dependencies
pip install modal verdict torch lightning

# Set up API keys
export GROK_API_KEY="your_grok_api_key"
export MODAL_TOKEN="your_modal_token"

# Deploy Modal function
modal deploy environments/community/pytorch_optimizer_coding/deploy.py

# Run evaluation
python environments/community/pytorch_optimizer_coding/wrapper.py
```

**Example Usage**:
```python
from environments.community.pytorch_optimizer_coding.optimizer_benchmark_environmenr import OptimizerBenchmarkEnvironment

# Initialize environment
env = OptimizerBenchmarkEnvironment(config=config)

# Generate optimizer code (from agent)
optimizer_code = """
class NovelOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    def step(self, closure=None):
        # Novel optimization logic here
        pass
"""

# Evaluate optimizer
reward = env.evaluate(optimizer_code)
print(f"Total reward: {reward}")
```

**Repository Structure**:
```
environments/community/pytorch_optimizer_coding/
├── optimizer_benchmark_environmenr.py  # Main environment interface
├── wrapper.py                          # Evaluation orchestration
├── evaluator.py                        # Grok-based assessment
├── deploy.py                           # Modal Labs deployment
├── run_optimizer_benchmark.py          # Standalone evaluation
├── requirements.txt                    # Python dependencies
├── FOB/                                # Framework for Optimizer Benchmarking
│   ├── optimizer_benchmark_env.py      # FOB integration
│   ├── pytorch_fob/                    # Core benchmarking framework
│   ├── baselines/                      # Baseline configurations
│   └── examples/                       # Usage examples
└── README.md                           # Detailed documentation
```

**Key Dependencies**:
- **Modal Labs**: Serverless code execution platform
- **Verdict**: Structured LLM evaluation framework
- **PyTorch Lightning**: Training framework for benchmarking
- **Grok API**: Novelty assessment via xAI's language model
- **FOB**: Framework for Optimizer Benchmarking

**Performance Expectations**:
- **Evaluation Time**: ~3 minutes per optimizer (compilation + 2 epoch training)
- **Memory Usage**: ~1GB RAM per evaluation
- **Throughput**: ~20 optimizers per hour (depending on Modal Labs capacity)
- **Success Rate**: ~60-80% compilation success for well-formed agent outputs

**Research Impact**: This environment addresses the underexplored area of automated optimizer discovery, providing a safe and comprehensive testbed for LLM-driven innovation in optimization algorithms. The multi-faceted evaluation ensures both novelty and practical effectiveness.

**Requirements**: modal, verdict, torch, lightning, transformers, datasets, pydantic, atroposlib

### 20. Helpful Doctors - Persona-Aware Medical QA Environment (`helpful_doctors/`)
**Author**: [tsadpbb](https://github.com/tsadpbb) with [AlxSp](https://github.com/AlxSp)
**Purpose**: Train LLMs to diagnose patients through multi-turn conversations while adapting to different patient communication styles and personalities

A sophisticated medical AI environment that simulates realistic doctor-patient interactions by introducing persona-based variability in patient communication styles. The environment tests whether medical reasoning systems can consistently arrive at correct diagnoses regardless of how patients present their symptoms, addressing the real-world challenge of variable patient communication.

**Features**:
- **Three Patient Personas**: Cooperative (verbose, informative), Reluctant (terse, evasive), and Neutral (brief, factual)
- **Multi-Turn Conversations**: Up to 10 follow-up questions before diagnosis requirement
- **MedQA Dataset Integration**: Uses `GBaker/MedQA-USMLE-4-options` for medical scenarios
- **Dual LLM Architecture**: Grok-3 for patient simulation, configurable model for doctor agent
- **Persona-Filtered Evaluation**: Tests diagnostic robustness across communication styles
- **Real-World Simulation**: Mirrors actual clinical variability in patient presentations

**Patient Persona Characteristics**:

**1. Cooperative Patient**:
- Open, verbose, and highly informative responses
- Provides detailed symptom descriptions
- Offers suggestions about potential diagnoses
- Answers with comprehensive information

**2. Reluctant Patient**:
- Terse, vague, and occasionally evasive responses
- Avoids direct answers to medical questions
- Provides minimal information initially
- Requires skillful questioning to extract details

**3. Neutral Patient**:
- Brief but factually consistent responses
- Provides accurate information without elaboration
- Straightforward communication style
- Balanced between cooperative and reluctant

**Training Process**:
1. **Scenario Setup**: MedQA question loaded with correct answer
2. **Persona Assignment**: Random selection of patient communication style
3. **Patient Simulation**: Grok-3 generates initial symptom presentation
4. **Doctor Interaction**: Agent asks follow-up questions (max 10)
5. **Diagnosis Requirement**: Agent must provide final diagnosis in specified format
6. **Evaluation**: Reward based on diagnostic accuracy

**Diagnostic Format**:
```
The diagnosis is: {medical_condition}
```

**Reward System**:
- **Correct Diagnosis**: +1.0 reward when diagnosis contains the correct answer
- **Incorrect Diagnosis**: 0.0 reward for wrong or missing diagnosis
- **Timeout Penalty**: 0.0 reward if conversation exceeds token limits
- **Accuracy Tracking**: Percentage correct maintained across training batches

**Example Interaction Flow**:
```
Patient (Cooperative): "I've been experiencing severe headaches for the past three days,
particularly in the morning. The pain is throbbing and located primarily in my forehead.
I've also noticed some sensitivity to light and mild nausea."

Doctor: "Can you describe the intensity of the pain on a scale of 1-10?"

Patient (Cooperative): "I'd say it's about an 8 out of 10 when it's at its worst.
The pain seems to worsen when I move my head quickly or bend over."

Doctor: "Have you experienced any visual disturbances or aura before the headaches?"

Patient (Cooperative): "Yes, actually I sometimes see flashing lights about 20 minutes
before the headache starts. There's also a strange zigzag pattern in my vision."

Doctor: "The diagnosis is: migraine with aura"
```

**Technical Implementation**:
- **Async Architecture**: Non-blocking patient simulation and doctor responses
- **Token Management**: Conversation length limits to prevent infinite loops
- **Dual API Integration**: Grok-3 for patients, configurable server for doctors
- **Message Threading**: Proper conversation state management
- **Evaluation Metrics**: WandB integration for training progress tracking

**Medical Applications**:
- **Clinical Training**: Teaching AI systems to handle diverse patient communication styles
- **Diagnostic Robustness**: Testing medical reasoning under realistic variability
- **Bedside Manner**: Training empathetic and adaptive questioning strategies
- **Triage Systems**: Developing AI for patient intake and initial assessment
- **Medical Education**: Simulating patient encounters for training purposes

**Research Contributions**:
- **Persona-Aware Benchmarking**: Novel approach to medical QA evaluation
- **Communication Variability**: Addresses gap between perfect narrators and real patients
- **Multi-Turn Reasoning**: Tests sustained diagnostic reasoning over conversations
- **Adaptive Questioning**: Encourages development of better follow-up strategies
- **Real-World Relevance**: Bridges gap between academic benchmarks and clinical practice

**Dataset Features**:
- **MedQA Foundation**: USMLE-style medical questions with multiple choice answers
- **Synthetic Patient Interactions**: AI-generated persona-based symptom presentations
- **Diagnostic Diversity**: Wide range of medical conditions and scenarios
- **Training/Test Splits**: 128 test examples for efficient evaluation
- **Shuffled Dataset**: Randomized order for robust training

**Configuration Options**:
- **Model Selection**: Configurable doctor model (default: NousResearch/DeepHermes-3-Llama-3-8B-Preview)
- **Conversation Limits**: Maximum 10 follow-up questions before diagnosis requirement
- **Token Management**: 15K token limit for conversation length
- **Batch Processing**: Group size 32 with 1024 batch size for training
- **Evaluation Frequency**: Steps per evaluation and limit ratios

**WandB Integration**:
- **Accuracy Tracking**: `train/percent_correct` for diagnostic success rate
- **Conversation Logging**: Complete doctor-patient interaction histories
- **Performance Metrics**: Training progress and evaluation results
- **Persona Analysis**: Success rates across different patient communication styles

**Future Enhancements**:

**Extended Persona Development**:
- **Cultural Variations**: Different cultural approaches to medical communication
- **Age-Specific Patterns**: Pediatric, adult, and geriatric communication styles
- **Emotional States**: Anxious, confused, or distressed patient personas
- **Language Barriers**: Non-native speaker communication patterns

**Advanced Medical Scenarios**:
- **Emergency Triage**: Time-critical diagnostic scenarios
- **Chronic Conditions**: Long-term patient management conversations
- **Mental Health**: Psychiatric evaluation and counseling scenarios
- **Specialist Consultations**: Domain-specific medical interactions

**Evaluation Improvements**:
- **Diagnostic Confidence**: Uncertainty quantification in medical decisions
- **Question Quality**: Assessment of follow-up question effectiveness
- **Empathy Scoring**: Evaluation of bedside manner and patient rapport
- **Efficiency Metrics**: Diagnostic accuracy per question asked

**Setup Requirements**:
1. **XAI API Key**: For Grok-3 patient simulation (`XAI_API_KEY` environment variable)
2. **Medical Dataset**: Automatic download of `GBaker/MedQA-USMLE-4-options`
3. **LLM Server**: Local or remote server for doctor agent inference
4. **WandB Account**: For training monitoring and experiment tracking

**Command Line Usage**:
```bash
# Set up API key
export XAI_API_KEY="your_xai_api_key"

# Start local LLM server for doctor agent
python -m vllm.entrypoints.openai.api_server \
    --model NousResearch/DeepHermes-3-Llama-3-8B-Preview \
    --port 9001

# Run helpful doctors environment
python environments/community/helpful_doctors/doctor.py process \
    --env.use_wandb true \
    --env.wandb_name helpful_doctors_training
```

**Demo Resources**:
- **YouTube Demo**: [Persona-Aware MedQA Benchmarking](https://youtube.com/shorts/02GEURik0PQ)
- **WandB Dashboard**: [Training Results](https://wandb.ai/nous-hackathon-2/atropos-environments_hack0_doctor_agent?nw=nwusertsadpbb)

**Research Impact**: This environment addresses a critical gap in medical AI evaluation by introducing realistic patient communication variability. Unlike traditional QA benchmarks that assume perfect narrators, this approach tests diagnostic robustness under human-like communication patterns, leading to more reliable and empathetic medical AI systems.

**Clinical Relevance**: Real doctors must interpret patient symptoms that are often incomplete, emotionally colored, or presented in various communication styles. This environment trains AI systems for these real-world challenges, potentially improving safety and effectiveness in clinical deployment.

**Requirements**: datasets, openai, atroposlib, wandb

### 21. DynastAI - Medieval Kingdom Management RL Environment with Adaptive Rewards (`dynastai/`)
**Author**: [Slyracoon23](https://github.com/Slyracoon23) with [davidvvliet](https://github.com/davidvvliet)
**Purpose**: Train LLMs to rule medieval kingdoms through strategic decision-making with an adaptive reward system that evolves based on player choices and outcomes

A comprehensive medieval kingdom management game that challenges agents to balance four critical metrics (Power, Stability, Piety, and Wealth) while ruling their kingdom. The environment features a novel adaptive reward mechanism that creates a dynamic learning landscape where the reward function evolves based on the agent's decision patterns, encouraging strategic balance rather than metric optimization.

**Features**:
- **Atropos Integration**: Full BaseEnv interface implementation for seamless training
- **FastAPI Backend**: REST API endpoints for game state management and card generation
- **Modern Web Frontend**: Responsive HTML/CSS/JS interface with character portraits
- **Adaptive Reward System**: Dynamic rewards that evolve based on player choices and outcomes
- **Dual Card Generation**: Pre-defined cards (400+) from JSON and dynamic generation via Qwen 1.7B
- **OpenRouter Integration**: Dynamic scenario generation using Qwen 1.7B language model

**Core Game Mechanics**:

**Kingdom Metrics (0-100 scale)**:
- **Power**: Royal authority and military strength
- **Stability**: Population happiness and social order
- **Piety**: Religious influence and spiritual authority
- **Wealth**: Kingdom finances and economic prosperity

**Decision Framework**:
Each turn presents scenario cards with binary choices (Yes/No) that affect multiple metrics. Cards are categorized by their primary impact area and generated based on current kingdom state and adaptive weights.

**Adaptive Reward Mechanism**:
The environment implements a novel reward formula that adapts to player behavior:

```
R = power_final * P + stability_final * S + piety_final * Pi + wealth_final * W
```

Where:
- `power_final`, `stability_final`, `piety_final`, `wealth_final` are final metric values
- `P`, `S`, `Pi`, `W` are counts of cards played in each category during the reign

**Category Weight Evolution**:
Weights update using exponential moving average (EMA) after each reign:
```
weights[category] = 0.9 * weights[category] + 0.1 * (final_metric * card_count)
```

This creates a self-adjusting reward landscape where:
1. Players who consistently favor one category shape the reward function to value that metric more
2. The game dynamically adjusts difficulty through card generation responding to player strengths
3. Strategic balance is rewarded over single-metric optimization

**Card Generation System**:

**Static Cards**: 400+ pre-defined scenario cards covering:
- **Power Scenarios**: Military conflicts, noble disputes, territorial expansion
- **Stability Scenarios**: Peasant concerns, social unrest, public works
- **Piety Scenarios**: Religious ceremonies, church politics, moral dilemmas
- **Wealth Scenarios**: Trade agreements, taxation, economic policies

**Dynamic Generation**: OpenRouter API integration with Qwen 1.7B for:
- **Context-Aware Scenarios**: Cards generated based on current kingdom state
- **Continuity Events**: References to previous reign outcomes (20% chance)
- **Adaptive Difficulty**: Scenario complexity adjusts to player performance
- **Historical Continuity**: Events that reference past decisions and outcomes

**Game Over Conditions**:
The reign ends when any metric reaches 0 or 100, representing:
- **Power 0**: Overthrown by nobles or enemies
- **Power 100**: Become a tyrant, assassinated
- **Stability 0**: Peasant revolt deposes ruler
- **Stability 100**: People establish republic
- **Piety 0**: Declared heretic, executed
- **Piety 100**: Church gains too much power
- **Wealth 0**: Kingdom bankruptcy
- **Wealth 100**: Invaded for vast wealth

**Training Architecture**:

**Environment Interface**:
```python
from environments.community.dynastai.src.dynastai_env import DynastAIEnv, DynastAIEnvConfig

# Create environment
config = DynastAIEnvConfig(
    api_host="localhost",
    api_port=9001,
    openrouter_api_key="your_key",
    web_ui=True,
    web_port=3000
)
env = DynastAIEnv(config)

# Training loop
observation = await env.reset()
action = {"session_id": observation["session_id"], "choice": "yes"}
observation, reward, done, info = await env.step(action)
```

**Action Space**:
- **session_id**: String identifier for game session
- **choice**: Binary decision ("yes" or "no") for current scenario card

**Observation Space**:
- **metrics**: Current kingdom metrics (power, stability, piety, wealth, reign_year)
- **current_card**: Active scenario card with text, options, and character information
- **session_id**: Game session identifier

**Reward Structure**:
- **During Reign**: 0.0 reward for intermediate steps
- **End of Reign**: Adaptive reward calculated using the formula above
- **Category Weight Updates**: Automatic adjustment for future reigns

**Web Interface Features**:

**Visual Design**:
- **Character Portraits**: Visual representation of advisors and scenario characters
- **Metric Displays**: Real-time kingdom status with visual indicators
- **Card Presentation**: Immersive scenario cards with medieval styling
- **Decision Feedback**: Visual effects showing metric changes

**Interactive Elements**:
- **Binary Choices**: Clear Yes/No decision buttons
- **Metric Tracking**: Live updates of kingdom statistics
- **Reign History**: Track of previous reigns and outcomes
- **Character Dialogue**: Immersive advisor interactions

**Technical Implementation**:

**FastAPI Endpoints**:
- `GET /api/`: API status and health check
- `POST /api/new_game`: Initialize new game session
- `GET /api/state/{session_id}`: Retrieve current game state
- `POST /api/generate_card`: Create new scenario card
- `POST /api/card_choice`: Process player decision
- `POST /api/end_reign`: Calculate final rewards and reset

**Card Generation Pipeline**:
1. **Category Selection**: Weighted random selection based on adaptive weights
2. **Source Determination**: Choose between static cards or dynamic generation
3. **Context Integration**: Include current metrics and previous reign history
4. **Validation**: Ensure card structure and effect ranges are valid
5. **Fallback System**: Mock cards if other sources fail

**Standalone Mode**:
The environment includes fallback classes for operation without atroposlib:
```python
# Automatic detection and graceful degradation
if HAS_ATROPOSLIB:
    from atroposlib.envs.base import BaseEnv, BaseEnvConfig
else:
    # Minimal stub classes for standalone operation
    class BaseEnv: pass
    class BaseEnvConfig: pass
```

**Configuration Options**:

**Environment Configuration**:
- **card_template_count**: Number of base card templates (default: 400)
- **api_host/api_port**: Server configuration for API endpoints
- **openrouter_api_key**: Optional key for dynamic card generation
- **llm_model**: Model selection for card generation (default: qwen/Qwen1.5-7B)
- **initial_category_weights**: Starting weights for card selection
- **web_ui/web_port**: Web interface configuration

**OpenRouter Integration**:
- **Model Selection**: Configurable LLM for card generation
- **Prompt Engineering**: Structured prompts for consistent card format
- **Error Handling**: Graceful fallback to static cards on API failures
- **Rate Limiting**: Respectful API usage patterns

**Research Applications**:

**Adaptive Learning**:
- **Dynamic Reward Landscapes**: Study how agents adapt to changing reward functions
- **Strategic Balance**: Investigate multi-objective optimization in RL
- **Preference Learning**: Understand how reward shaping affects agent behavior
- **Meta-Learning**: Adaptation to evolving game mechanics

**Decision Making**:
- **Long-Term Planning**: Balance immediate vs. long-term consequences
- **Risk Management**: Navigate trade-offs between different kingdom aspects
- **Contextual Reasoning**: Make decisions based on current kingdom state
- **Strategic Thinking**: Develop coherent governance strategies

**Human-AI Interaction**:
- **Preference Alignment**: Train agents that match human strategic preferences
- **Explainable Decisions**: Generate reasoning for kingdom management choices
- **Interactive Training**: Human-in-the-loop learning for governance strategies
- **Cultural Adaptation**: Adjust strategies based on different value systems

**Performance Characteristics**:

**Computational Requirements**:
- **Memory Usage**: <1GB RAM for environment operation
- **API Calls**: Optional OpenRouter usage for dynamic content
- **Web Server**: Lightweight FastAPI backend
- **Storage**: Minimal state storage for game sessions

**Scalability**:
- **Session Management**: In-memory storage with UUID-based sessions
- **Concurrent Games**: Multiple simultaneous game sessions supported
- **API Rate Limits**: Respectful usage of external services
- **Fallback Systems**: Graceful degradation when services unavailable

**Training Efficiency**:
- **Episode Length**: Variable (typically 10-30 decisions per reign)
- **Reward Frequency**: Sparse rewards at reign end encourage long-term planning
- **State Complexity**: Manageable state space for efficient learning
- **Action Space**: Simple binary decisions reduce exploration complexity

**Setup Instructions**:

**Basic Installation**:
```bash
# Install dependencies
pip install -r environments/community/dynastai/requirements.txt

# Optional: Set OpenRouter API key for dynamic cards
export OPENROUTER_API_KEY="your_api_key_here"

# Run standalone game
python environments/community/dynastai/run_dynastai.py
```

**Atropos Integration**:
```bash
# From atropos root directory
python -m atroposlib.envs.dynastai_env serve --slurm False
```

**Web Interface**:
Access the game at `http://localhost:3000` when running the server.

**Demo Resources**:
- **Live Demo Video**: [DynastAI Gameplay & API Overview](https://github.com/Slyracoon23/atropos/pull/81)
- **Screenshots**: Medieval-themed interface with character portraits and metric displays
- **API Documentation**: Comprehensive endpoint documentation in README

**Future Enhancements**:

**Advanced Mechanics**:
- **Multi-Turn Scenarios**: Complex events requiring multiple decisions
- **Diplomatic Relations**: Interactions with neighboring kingdoms
- **Random Events**: Natural disasters, plagues, and unexpected challenges
- **Character Development**: Persistent advisors with evolving relationships

**Enhanced Adaptation**:
- **Player Modeling**: Deeper understanding of decision patterns
- **Difficulty Scaling**: Automatic adjustment based on player skill
- **Narrative Continuity**: Stronger connections between reigns
- **Cultural Variations**: Different medieval settings and value systems

**Research Extensions**:
- **Multi-Agent Scenarios**: Competing kingdoms with shared resources
- **Hierarchical Decision Making**: Delegation to specialized advisors
- **Uncertainty Modeling**: Incomplete information about decision outcomes
- **Ethical Reasoning**: Moral dilemmas in governance decisions

**Educational Applications**:
- **History Teaching**: Interactive exploration of medieval governance
- **Ethics Education**: Moral reasoning in leadership contexts
- **Game Design**: Example of adaptive reward systems
- **AI Safety**: Testing alignment in complex decision environments

**Research Impact**: DynastAI introduces a novel approach to RL environment design through its adaptive reward mechanism. Unlike static reward functions, this system creates a co-evolving learning landscape where both agent and environment adapt, potentially leading to more robust and generalizable decision-making strategies.

**Governance Simulation**: The environment provides a rich testbed for studying AI governance and decision-making under uncertainty. The medieval setting offers familiar metaphors while the adaptive mechanics create genuine strategic challenges that mirror real-world leadership complexities.

**Requirements**: fastapi, uvicorn, pydantic, requests, python-dotenv, httpx, aiohttp, jinja2, tqdm, numpy, wandb, atroposlib

---

## Support

For questions or issues with community environments:
- Check the individual environment's README first
- Open an issue in the main repository
- Tag the environment author if possible

*These environments are community contributions and may have different maintenance levels and support compared to core Atropos environments.*
