# DynastAI Integration with Atropos

This document provides instructions on how to use DynastAI with Atropos.

## Quick Start Guide

### Option 1: Using the Web Interface (No Atropos Required)

This is the simplest way to test the game mechanics:

```bash
# Navigate to the dynastai directory
cd environments/dynastai

# Run the quick start script
python run_dynastai.py
```

Your browser will open automatically to http://localhost:3000.

### Option 2: Using with Atropos

```bash
# From the atropos root directory
python environments/dynastai_environment.py serve --web-ui
```

The environment will start and be available for Atropos trainers.

## Testing Components

You can test individual components:

1. **Card Generation:**
   ```bash
   cd environments/dynastai
   python test_card_generation.py
   ```

2. **API Endpoints:**
   ```bash
   cd environments/dynastai
   # Start the server in one terminal
   python dynastai_server.py
   # In another terminal
   python test_dynastai_api.py
   ```

3. **Environment Integration:**
   ```bash
   cd environments/dynastai
   python test_dynastai_env.py
   ```

## Directory Structure

- `dynastai_environment.py`: Main entry point for Atropos integration
- `dynastai/`: Environment package
  - `src/dynastai_env.py`: Atropos environment implementation
  - `src/game_logic.py`: Core game mechanics
  - `src/web/`: Web interface
  - `src/data/`: Game data including cards.json

## Troubleshooting

- **Missing cards.json**: Run `test_card_generation.py` to generate it
- **API errors**: Ensure server is running on port 9001
- **Import errors**: Make sure you're in the correct directory and have installed dependencies
- **Web UI not loading**: Check that the server is running and the ports are correct

## Feedback and Issues

Report any issues on GitHub or contact the maintainer directly.
