"""
Game Logic for DynastAI

This module handles the core game mechanics:
- Game state tracking
- Card generation via OpenRouter/Qwen
- Decision effects processing
- Win/loss condition checking
"""

import os
import json
import random
import uuid
import time
from typing import Dict, List, Tuple, Any, Optional
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class GameState:
    """
    Represents the state of a single DynastAI game session.
    """
    def __init__(self):
        # Initialize game metrics (0-100 scale)
        self.power = 50      # Royal authority/Power
        self.stability = 50  # Population happiness/Stability
        self.piety = 50      # Religious influence/Piety
        self.wealth = 50     # Kingdom finances/Wealth
        
        # Game state tracking
        self.reign_year = 1
        self.current_card = None
        self.card_history = []  # List of played cards
        self.choice_history = []  # List of yes/no choices made
        
        # Category counts for adaptive reward calculation
        self.category_counts = {"power": 0, "stability": 0, "piety": 0, "wealth": 0}
    
    def get_metrics(self) -> Dict[str, int]:
        """Return the current game metrics"""
        return {
            "power": self.power,
            "stability": self.stability, 
            "piety": self.piety,
            "wealth": self.wealth,
            "reign_year": self.reign_year
        }
    
    def get_category_counts(self) -> Dict[str, int]:
        """Return the count of cards played by category"""
        return self.category_counts
    
    def record_card_play(self, card, choice):
        """Record a card play and choice"""
        if card and "category" in card:
            # Increment the category count
            category = card["category"]
            if category in self.category_counts:
                self.category_counts[category] += 1
                
        # Store the card and choice in history
        self.card_history.append(card)
        self.choice_history.append(choice)
        
        # Increment reign year
        self.reign_year += 1


def generate_card(metrics: Dict[str, int], category_weights: Dict[str, int]) -> Dict:
    """
    Generate a new card using the OpenRouter API (Qwen 1.7B)
    
    Parameters:
    - metrics: Current game metrics
    - category_weights: Weights for selecting card categories
    
    Returns:
    - card: A card object with text, options and effects
    """
    # Select a category based on weights
    categories = list(category_weights.keys())
    weights = [category_weights[cat] for cat in categories]
    total_weight = sum(weights)
    
    # Normalize weights to avoid issues if weights are too small
    if total_weight > 0:
        normalized_weights = [w/total_weight for w in weights]
    else:
        normalized_weights = [1/len(categories)] * len(categories)
    
    category = random.choices(categories, weights=normalized_weights, k=1)[0]
    
    # Generate a unique card ID
    card_id = f"card_{str(uuid.uuid4())[:8]}"
    
    # Create a card prompt
    prompt = f"""System: "You are generating JSON event cards for a medieval kingdom management game."

User: "Create a {category} focused event card for a medieval ruler. 
Current metrics: Power:{metrics['power']}, Stability:{metrics['stability']}, Piety:{metrics['piety']}, Wealth:{metrics['wealth']}.
Output ONLY a JSON event card object like this:
{{
  'id': '{card_id}',
  'text': 'Card scenario description...',
  'yes_option': 'First option text...',
  'no_option': 'Second option text...',
  'effects': {{
    'yes': {{'power': int, 'stability': int, 'piety': int, 'wealth': int}},
    'no': {{'power': int, 'stability': int, 'piety': int, 'wealth': int}}
  }},
  'category': '{category}'
}}
Make sure the effects are integers between -20 and +20, with most being -10 to +10."
"""

    # Call the OpenRouter API
    try:
        if not OPENROUTER_API_KEY:
            # If no API key, generate a mock card for testing
            return generate_mock_card(metrics, category)
            
        response = call_openrouter(prompt)
        
        # Parse the JSON content from the response
        try:
            card_data = json.loads(response)
            # Validate that the card has all required fields
            if validate_card(card_data):
                return card_data
            else:
                # If validation fails, fall back to a mock card
                return generate_mock_card(metrics, category)
        except json.JSONDecodeError:
            # If JSON parsing fails, fall back to a mock card
            print(f"Error parsing card JSON: {response}")
            return generate_mock_card(metrics, category)
            
    except Exception as e:
        print(f"Error generating card: {e}")
        # Fall back to a mock card in case of any error
        return generate_mock_card(metrics, category)


def call_openrouter(prompt, model="qwen/Qwen1.5-7B"):
    """
    Send a prompt to the OpenRouter API and return the response
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"].strip()


def validate_card(card: Dict) -> bool:
    """
    Validate that the card has all required fields and proper structure
    """
    required_fields = ["id", "text", "yes_option", "no_option", "effects", "category"]
    if not all(field in card for field in required_fields):
        return False
        
    if not all(choice in card["effects"] for choice in ["yes", "no"]):
        return False
        
    metrics = ["power", "stability", "piety", "wealth"]
    for choice in ["yes", "no"]:
        if not all(metric in card["effects"][choice] for metric in metrics):
            return False
            
    return True


def generate_mock_card(metrics: Dict[str, int], category: str) -> Dict:
    """
    Generate a mock card for testing when OpenRouter API is unavailable
    """
    effect_range = (-10, 10)
    
    # Create effects for yes and no choices
    yes_effects = {metric: random.randint(*effect_range) for metric in ["power", "stability", "piety", "wealth"]}
    no_effects = {metric: random.randint(*effect_range) for metric in ["power", "stability", "piety", "wealth"]}
    
    # Ensure category effect is positive for 'yes' and negative for 'no'
    yes_effects[category] = random.randint(5, 15)
    no_effects[category] = random.randint(-15, -5)
    
    # Create mock scenarios based on category
    scenarios = {
        "power": "The Royal General requests funds to expand the army.",
        "stability": "Peasants from the northern province complain about high taxes.",
        "piety": "The Cardinal proposes building a new cathedral in the capital.",
        "wealth": "The Master of Coin suggests a new trade agreement with a neighboring kingdom."
    }
    
    yes_options = {
        "power": "Strengthen our military",
        "stability": "Reduce their tax burden",
        "piety": "Fund the cathedral project",
        "wealth": "Approve the trade agreement"
    }
    
    no_options = {
        "power": "Maintain current military size",
        "stability": "Keep the tax rates as they are",
        "piety": "Reject the cathedral project",
        "wealth": "Decline the trade agreement"
    }
    
    return {
        "id": f"mock_card_{str(uuid.uuid4())[:8]}",
        "text": scenarios.get(category, f"A {category} related scenario has emerged in your kingdom."),
        "yes_option": yes_options.get(category, "Approve"),
        "no_option": no_options.get(category, "Decline"),
        "effects": {
            "yes": yes_effects,
            "no": no_effects
        },
        "category": category
    }


def apply_choice_effects(game_state: GameState, choice: str) -> Tuple[bool, Dict[str, int], Dict]:
    """
    Apply the effects of a player's choice to the game state
    
    Parameters:
    - game_state: The current game state
    - choice: "yes" or "no"
    
    Returns:
    - is_game_over: Whether the game has ended
    - new_metrics: Updated metrics 
    - effects: The effects that were applied
    """
    if not game_state.current_card:
        raise ValueError("No current card in game state")
        
    # Get the effects based on the choice
    if choice not in ["yes", "no"]:
        raise ValueError(f"Invalid choice: {choice}. Must be 'yes' or 'no'")
        
    effects = game_state.current_card["effects"][choice]
    
    # Apply effects to game metrics
    game_state.power = max(0, min(100, game_state.power + effects["power"]))
    game_state.stability = max(0, min(100, game_state.stability + effects["stability"]))
    game_state.piety = max(0, min(100, game_state.piety + effects["piety"]))
    game_state.wealth = max(0, min(100, game_state.wealth + effects["wealth"]))
    
    # Record the card play
    game_state.record_card_play(game_state.current_card, choice)
    
    # Check for game over conditions
    is_game_over = check_game_over(game_state)
    
    # Return updated metrics
    new_metrics = game_state.get_metrics()
    
    return is_game_over, new_metrics, effects


def check_game_over(game_state: GameState) -> bool:
    """
    Check if the game is over based on the current metrics
    
    The game ends if any metric reaches 0 or 100
    """
    metrics = [game_state.power, game_state.stability, game_state.piety, game_state.wealth]
    
    return any(metric <= 0 or metric >= 100 for metric in metrics)