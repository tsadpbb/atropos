# DynastAI - Installation and Running Guide

This guide provides step-by-step instructions to install and run the DynastAI game.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

## Installation

### Step 1: Get the Code

If using git:
```bash
git clone https://github.com/torinvdb/atropos.git
cd atropos
```

Or download and unzip the project, then navigate to the project folder.

### Step 2: Install Dependencies

Option 1 - Using the setup script (recommended):
```bash
cd environments/dynastai
python setup.py
```

Option 2 - Manual installation:
```bash
cd environments/dynastai
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all required packages including:
- FastAPI and Uvicorn for the backend server
- Pydantic for data validation
- Requests for API calls
- Python-dotenv for environment variable management

Note: If you're using Python 3.13+, the setup script handles compatibility issues automatically.

### Step 3 (Optional): Add OpenRouter API Key

For dynamic card generation using AI, create a `.env` file in the `environments/dynastai` directory:

```bash
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

If you don't have an OpenRouter API key, the game will use pre-defined cards from the cards.json file.

## Running the Game

### Option 1: Web Interface (Simple)

This is the easiest way to play the game directly:

```bash
python run_dynastai.py
```

Your default browser will open automatically to http://localhost:3000, and you can begin playing.

Command options:
- `--no-browser`: Don't open the browser automatically
- `--api-port 9001`: Use a different API port (default: 9001)  
- `--web-port 3000`: Use a different web port (default: 3000)

Example:
```bash
python run_dynastai.py --api-port 8080 --web-port 8000
```

### Option 2: Atropos Integration (Advanced)

For integration with the Atropos reinforcement learning framework:

```bash
# From the atropos root directory
python environments/dynastai_environment.py serve --web-ui
```

## Troubleshooting

- **"Missing cards.json" error**: Run `python test_card_generation.py` to generate it
- **API connection error**: Ensure the API server is running on the specified port
- **Import errors**: Verify that all dependencies are installed
- **Web UI not loading**: Check that both API and web servers are running correctly
- **Python 3.13+ compatibility issues**: Some packages may need manual installation:
  ```bash
  pip install --force-reinstall --no-binary aiohttp aiohttp>=3.9.0
  ```

## Playing the Game

- The game presents you with scenario cards that impact your kingdom
- Make choices (Yes/No) to affect your kingdom's metrics:
  - Power: Royal authority and military strength
  - Stability: Population happiness and civic order
  - Piety: Religious influence and moral standing
  - Wealth: Kingdom finances and economic prosperity
- Your reign ends when any metric reaches 0 or 100, or after 30 years
- Each decision affects the adaptive reward system that evolves gameplay based on your choices
