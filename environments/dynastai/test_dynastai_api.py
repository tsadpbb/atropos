#!/usr/bin/env python3
"""
DynastAI API Test Script

This script tests the FastAPI endpoints of the DynastAI game:
- Creating a new game
- Getting game state
- Generating cards
- Processing choices
- Ending reigns
"""

import os
import sys
import asyncio
import json
import random
import httpx
from dotenv import load_dotenv

# Ensure the src directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()

# API configuration
API_URL = "http://localhost:9001/api"

async def test_api():
    """Test the DynastAI API endpoints"""
    print("Testing DynastAI API...")
    
    async with httpx.AsyncClient() as client:
        # Test root endpoint
        print("\nTesting root endpoint...")
        response = await client.get(f"{API_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Create a new game
        print("\nCreating new game...")
        response = await client.post(f"{API_URL}/new_game")
        game_data = response.json()
        session_id = game_data["session_id"]
        print(f"Session ID: {session_id}")
        print(f"Initial metrics: {game_data['metrics']}")
        
        # Generate a card
        print("\nGenerating card...")
        response = await client.post(
            f"{API_URL}/generate_card",
            json={"session_id": session_id}
        )
        card = response.json()
        print(f"Card: {card['text']}")
        print(f"Option Yes: {card['yes_option']}")
        print(f"Option No: {card['no_option']}")
        
        # Make a choice
        print("\nMaking choice...")
        choice = random.choice(["yes", "no"])
        response = await client.post(
            f"{API_URL}/card_choice",
            json={"session_id": session_id, "choice": choice}
        )
        result = response.json()
        print(f"Choice: {choice}")
        print(f"New metrics: {result['metrics']}")
        print(f"Game over: {result.get('game_over', False)}")
        
        # End the reign
        print("\nEnding reign...")
        trajectory = [{
            "card_id": card["id"],
            "category": card["category"],
            "choice": choice,
            "effects": card["effects"][choice],
            "post_metrics": result["metrics"]
        }]
        
        response = await client.post(
            f"{API_URL}/end_reign",
            json={
                "session_id": session_id,
                "trajectory": trajectory,
                "final_metrics": result["metrics"],
                "reign_length": result["metrics"]["reign_year"],
                "cause_of_end": "test_termination"
            }
        )
        end_data = response.json()
        print(f"Reward: {end_data['reward']}")
        print(f"New weights: {end_data['new_weights']}")
    
    print("\nAPI tests completed successfully!")
    return True

if __name__ == "__main__":
    print("DynastAI API Test")
    print("================")
    
    # Check if server is running in a separate process
    print("NOTE: This test assumes the DynastAI server is running.")
    print("Please start the server with 'python dynastai_server.py' before running this test.")
    
    input("Press Enter to continue...")
    
    # Run the test
    asyncio.run(test_api())
