   """
FastAPI endpoints for DynastAI game

This module provides the REST API endpoints for the DynastAI game:
- GET /state: Get current game state
- POST /generate_card: Generate a new card
- POST /card_choice: Submit player choice
- POST /end_reign: End a reign and compute rewards
"""

import json
import uuid
import os
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the game logic
from ..game_logic import GameState, generate_card, apply_choice_effects

# In-memory store for game sessions
game_sessions: Dict[str, GameState] = {}

# In-memory store for category weights across reigns
category_weights: Dict[str, float] = {"power": 50, "stability": 50, "piety": 50, "wealth": 50}

# Path to save reign trajectories
trajectories_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trajectories.json")

# Make sure the data directory exists
os.makedirs(os.path.dirname(trajectories_path), exist_ok=True)

# Define the API models using Pydantic
class NewGameRequest(BaseModel):
    """Request model for creating a new game"""
    session_id: Optional[str] = None  # Optional custom session ID

class GameStateResponse(BaseModel):
    """Response model for game state"""
    session_id: str
    metrics: Dict[str, int]
    current_card: Optional[Dict[str, Any]] = None

class GenerateCardRequest(BaseModel):
    """Request model for generating a new card"""
    session_id: str

class CardChoiceRequest(BaseModel):
    """Request model for submitting a card choice"""
    session_id: str
    choice: str  # "yes" or "no"

class TrajectoryItem(BaseModel):
    """Model for a single trajectory item"""
    card_id: str
    category: str
    choice: str
    effects: Dict[str, Any]
    post_metrics: Dict[str, int]

class EndReignRequest(BaseModel):
    """Request model for ending a reign"""
    session_id: str
    trajectory: List[TrajectoryItem]
    final_metrics: Dict[str, int]
    reign_length: int
    cause_of_end: Optional[str] = None

class EndReignResponse(BaseModel):
    """Response model for ending a reign"""
    reward: float
    session_id: str
    new_weights: Dict[str, float]

# Create FastAPI instance
api = FastAPI(title="DynastAI API", 
              description="REST API for the DynastAI medieval kingdom management game",
              version="1.0.0")

# Add CORS middleware to allow requests from any origin
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins 
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@api.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to the DynastAI API"}

@api.post("/new_game", response_model=GameStateResponse)
async def new_game(request: NewGameRequest = None):
    """Create a new game session"""
    # Generate a session ID if not provided
    session_id = request.session_id if request and request.session_id else str(uuid.uuid4())
    
    # Create a new game state
    game_sessions[session_id] = GameState()
    
    # Return the initial game state
    return GameStateResponse(
        session_id=session_id,
        metrics=game_sessions[session_id].get_metrics(),
        current_card=None
    )

@api.get("/state/{session_id}", response_model=GameStateResponse)
async def get_state(session_id: str):
    """Get the current state of a game session"""
    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    return GameStateResponse(
        session_id=session_id,
        metrics=game_sessions[session_id].get_metrics(),
        current_card=game_sessions[session_id].current_card
    )

@api.post("/generate_card", response_model=Dict[str, Any])
async def generate_new_card(request: GenerateCardRequest):
    """Generate a new card for the game session"""
    if request.session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    game_state = game_sessions[request.session_id]
    
    # Generate a new card using the current metrics and category weights
    card = generate_card(game_state.get_metrics(), category_weights)
    
    # Store the card in the game state
    game_state.current_card = card
    
    # Return the card
    return card

@api.post("/card_choice", response_model=GameStateResponse)
async def process_card_choice(request: CardChoiceRequest):
    """Process a player's card choice"""
    if request.session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    if request.choice not in ["yes", "no"]:
        raise HTTPException(status_code=400, detail="Choice must be 'yes' or 'no'")
        
    game_state = game_sessions[request.session_id]
    
    if not game_state.current_card:
        raise HTTPException(status_code=400, detail="No active card for this session")
        
    # Apply the effects of the choice
    is_game_over, metrics, effects = apply_choice_effects(game_state, request.choice)
    
    # Return the updated game state along with game_over flag
    return {
        "session_id": request.session_id,
        "metrics": metrics,
        "current_card": game_state.current_card,
        "game_over": is_game_over
    }

@api.post("/end_reign", response_model=EndReignResponse)
async def end_reign(request: EndReignRequest):
    """
    End a reign and compute reward
    This endpoint receives the entire trajectory of a reign
    """
    if request.session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    # Calculate the adaptive reward
    reward = calculate_adaptive_reward(request.final_metrics, request.trajectory)
    
    # Update category weights
    update_category_weights(request.final_metrics, request.trajectory)
    
    # Log the trajectory
    log_trajectory(request, reward)
    
    # Clean up the session
    # Note: We don't delete it in case the client wants to start a new reign with same session
    game_sessions[request.session_id] = GameState()  # Reset the game state
    
    return EndReignResponse(
        reward=reward,
        session_id=request.session_id,
        new_weights=category_weights
    )

def calculate_adaptive_reward(final_metrics: Dict[str, int], trajectory: List[TrajectoryItem]) -> float:
    """
    Calculate the adaptive reward based on the final metrics and trajectory
    
    R = power_final * P + stability_final * S + piety_final * Pi + wealth_final * W
    """
    # Count the number of cards in each category
    category_counts = {"power": 0, "stability": 0, "piety": 0, "wealth": 0}
    
    for item in trajectory:
        if item.category in category_counts:
            category_counts[item.category] += 1
    
    # Calculate the reward
    reward = (
        final_metrics["power"] * category_counts["power"] +
        final_metrics["stability"] * category_counts["stability"] +
        final_metrics["piety"] * category_counts["piety"] +
        final_metrics["wealth"] * category_counts["wealth"]
    )
    
    return reward

def update_category_weights(final_metrics: Dict[str, int], trajectory: List[TrajectoryItem]):
    """
    Update category weights using exponential moving average (EMA)
    
    weights["power"]     = 0.9 * weights["power"]     + 0.1 * (power_final     * P_last)
    weights["stability"] = 0.9 * weights["stability"] + 0.1 * (stability_final * S_last)
    weights["piety"]     = 0.9 * weights["piety"]     + 0.1 * (piety_final     * Pi_last)
    weights["wealth"]    = 0.9 * weights["wealth"]    + 0.1 * (wealth_final    * W_last)
    """
    global category_weights
    
    # Count the number of cards in each category
    category_counts = {"power": 0, "stability": 0, "piety": 0, "wealth": 0}
    
    for item in trajectory:
        if item.category in category_counts:
            category_counts[item.category] += 1
    
    # Update weights using EMA
    alpha = 0.9  # Weight for the old value
    beta = 0.1   # Weight for the new value
    
    for category in category_weights:
        category_weights[category] = (
            alpha * category_weights[category] +
            beta * (final_metrics[category] * category_counts[category])
        )
        # Ensure weights stay in a reasonable range
        category_weights[category] = max(1, min(100, category_weights[category]))
    
    print(f"Updated category weights: {category_weights}")

def log_trajectory(request: EndReignRequest, reward: float):
    """Log the trajectory to a JSON file"""
    trajectory_data = {
        "session_id": request.session_id,
        "trajectory": [item.dict() for item in request.trajectory],
        "final_metrics": request.final_metrics,
        "reign_length": request.reign_length,
        "cause_of_end": request.cause_of_end,
        "reward": reward,
        "weights": category_weights
    }
    
    try:
        # Load existing trajectories
        trajectories = []
        if os.path.exists(trajectories_path):
            with open(trajectories_path, 'r') as f:
                trajectories = json.load(f)
        
        # Add new trajectory
        trajectories.append(trajectory_data)
        
        # Save back to file
        with open(trajectories_path, 'w') as f:
            json.dump(trajectories, f, indent=2)
            
    except Exception as e:
        print(f"Error logging trajectory: {e}")
        # Continue without failing the request