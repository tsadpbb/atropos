#!/usr/bin/env python3
"""
DynastAI Test Script

This script tests the basic functionality of the DynastAI environment:
- Game state initialization
- Card generation
- Choice processing
- Game over conditions
- Reward calculation
"""

import os
import sys
import asyncio
import json
import random
from dotenv import load_dotenv

# Ensure the src directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.dynastai_env import DynastAIEnv, DynastAIEnvConfig
from atroposlib.envs.server_handling.server_baseline import ServerBaseline

# Load environment variables
load_dotenv()

async def test_environment():
    """Test the DynastAI environment"""
    print("Initializing DynastAI environment...")
    
    # Create config and environment
    config = DynastAIEnvConfig()
    server_config = ServerBaseline()
    env = DynastAIEnv(config, server_config)
    
    # Reset the environment
    print("Resetting environment...")
    state = await env.reset()
    session_id = state["session_id"]
    print(f"Session ID: {session_id}")
    print(f"Initial metrics: {state['metrics']}")
    
    # Run a simulated game
    done = False
    total_reward = 0
    steps = 0
    
    print("\nStarting simulated game...")
    while not done and steps < 20:  # Cap at 20 steps to avoid infinite loop
        # Generate a card (direct method call for testing)
        state["current_card"] = env.game_states[session_id].current_card = env._generate_card_internal(
            env.game_states[session_id].get_metrics(),
            env.category_weights
        )
        
        # Choose a random action
        choice = random.choice(["yes", "no"])
        action = {"session_id": session_id, "choice": choice}
        
        print(f"\nStep {steps+1}:")
        print(f"Card: {state['current_card']['text']}")
        print(f"Choice: {choice}")
        
        # Take the action
        state, reward, done, info = await env.step(action)
        
        # Print the results
        print(f"New metrics: {state['metrics']}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        
        total_reward += reward
        steps += 1
        
        if done:
            print("\nGame Over!")
            print(f"Final metrics: {state['metrics']}")
            print(f"Total reward: {total_reward}")
            print(f"Steps: {steps}")
            break
    
    print("\nTesting complete!")
    return True

def _generate_card_internal(self, metrics, category_weights):
    """Internal method for card generation during testing"""
    from src.game_logic import generate_mock_card
    
    # Select a category based on weights
    categories = list(category_weights.keys())
    weights = [category_weights[cat] for cat in categories]
    total_weight = sum(weights)
    
    # Normalize weights
    if total_weight > 0:
        normalized_weights = [w/total_weight for w in weights]
    else:
        normalized_weights = [1/len(categories)] * len(categories)
    
    category = random.choices(categories, weights=normalized_weights, k=1)[0]
    
    # Use the mock generator for testing
    return generate_mock_card(metrics, category)

# Add the test method to the environment class
DynastAIEnv._generate_card_internal = _generate_card_internal

if __name__ == "__main__":
    print("DynastAI Environment Test")
    print("=========================")
    
    # Run the test
    asyncio.run(test_environment())
