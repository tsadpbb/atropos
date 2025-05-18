#!/usr/bin/env python3
"""
DynastAI Local Server - For local development and testing

This script starts both the backend API server and serves the frontend static files.
"""

import os
import sys
import argparse
import subprocess
import time
import webbrowser
from threading import Thread

# Ensure src directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.web.server import run_server
from src.config import get_config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DynastAI Local Development Server")
    parser.add_argument("--api-port", type=int, default=9001, help="Port for the API server")
    parser.add_argument("--web-port", type=int, default=3000, help="Port for the web server")
    parser.add_argument("--no-open", action="store_true", help="Don't open the web browser automatically")
    return parser.parse_args()

def run_api_server(port):
    """Run the API server"""
    from src.web.server import run_server
    run_server(host="localhost", port=port)

def run_web_server(port, static_dir):
    """Run a simple HTTP server for the frontend"""
    import http.server
    import socketserver
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=static_dir, **kwargs)
    
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Web server running at http://localhost:{port}")
        httpd.serve_forever()

def main():
    """Main entry point"""
    args = parse_args()
    config = get_config()
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), "src/data"), exist_ok=True)
    
    # Initialize the web directory if not already set up
    web_dir = os.path.join(os.path.dirname(__file__), "src/web")
    static_dir = os.path.join(web_dir, "static")
    
    if not os.path.exists(static_dir):
        os.makedirs(static_dir, exist_ok=True)
        
        # Create basic HTML, CSS, JS files if they don't exist
        html_file = os.path.join(static_dir, "index.html")
        css_file = os.path.join(static_dir, "styles.css")
        js_file = os.path.join(static_dir, "game.js")
        
        if not os.path.exists(html_file):
            with open(html_file, 'w') as f:
                f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DynastAI - Medieval Kingdom Management</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <h1>DynastAI</h1>
        <p>Rule wisely or suffer the consequences</p>
    </header>
    
    <main>
        <div id="metrics-container">
            <div class="metric">
                <h3>Power</h3>
                <div class="meter-container">
                    <div class="meter power" id="power-meter"></div>
                </div>
                <span id="power-value">50</span>
            </div>
            <div class="metric">
                <h3>Stability</h3>
                <div class="meter-container">
                    <div class="meter stability" id="stability-meter"></div>
                </div>
                <span id="stability-value">50</span>
            </div>
            <div class="metric">
                <h3>Piety</h3>
                <div class="meter-container">
                    <div class="meter piety" id="piety-meter"></div>
                </div>
                <span id="piety-value">50</span>
            </div>
            <div class="metric">
                <h3>Wealth</h3>
                <div class="meter-container">
                    <div class="meter wealth" id="wealth-meter"></div>
                </div>
                <span id="wealth-value">50</span>
            </div>
        </div>
        
        <div id="card-container" class="hidden">
            <div id="card">
                <div id="card-text">Welcome to your kingdom, Your Majesty. Make your choices wisely...</div>
                <div id="card-options">
                    <button id="yes-button">Yes</button>
                    <button id="no-button">No</button>
                </div>
            </div>
        </div>
        
        <div id="start-screen">
            <h2>Begin Your Reign</h2>
            <button id="start-game">Start New Game</button>
        </div>
        
        <div id="game-over" class="hidden">
            <h2>Your Reign Has Ended</h2>
            <p id="game-over-reason"></p>
            <p id="reign-summary"></p>
            <button id="new-game">Start New Reign</button>
        </div>
    </main>
    
    <footer>
        <p>Year <span id="reign-year">1</span> of your reign</p>
    </footer>
    
    <script src="game.js"></script>
</body>
</html>
                """)
        
        if not os.path.exists(css_file):
            with open(css_file, 'w') as f:
                f.write("""
:root {
    --power-color: #e74c3c;
    --stability-color: #2ecc71;
    --piety-color: #f1c40f;
    --wealth-color: #3498db;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Georgia', serif;
    background-color: #f5f5f5;
    color: #333;
    line-height: 1.6;
    background-image: url('https://images.unsplash.com/photo-1534196511436-921a4e99f297?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

header, footer {
    text-align: center;
    padding: 1rem;
    background-color: rgba(0, 0, 0, 0.7);
    color: #fff;
}

header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

main {
    max-width: 800px;
    margin: 2rem auto;
    min-height: calc(100vh - 12rem);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

#metrics-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    margin-bottom: 2rem;
    background-color: rgba(255, 255, 255, 0.9);
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.metric {
    flex: 1;
    min-width: 150px;
    margin: 0.5rem;
    text-align: center;
}

.meter-container {
    background-color: #ddd;
    height: 20px;
    border-radius: 10px;
    margin: 0.5rem 0;
    overflow: hidden;
}

.meter {
    height: 100%;
    transition: width 0.5s ease-in-out;
}

.power { background-color: var(--power-color); width: 50%; }
.stability { background-color: var(--stability-color); width: 50%; }
.piety { background-color: var(--piety-color); width: 50%; }
.wealth { background-color: var(--wealth-color); width: 50%; }

#card-container {
    display: flex;
    justify-content: center;
    margin: 2rem 0;
}

#card {
    background-color: rgba(255, 255, 255, 0.95);
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    width: 100%;
    max-width: 600px;
}

#card-text {
    font-size: 1.2rem;
    margin-bottom: 2rem;
    line-height: 1.6;
}

#card-options {
    display: flex;
    justify-content: space-between;
}

button {
    padding: 0.8rem 2rem;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 1rem;
    font-weight: bold;
    transition: all 0.2s ease;
}

#yes-button {
    background-color: #2ecc71;
    color: white;
}

#no-button {
    background-color: #e74c3c;
    color: white;
}

button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

#start-screen, #game-over {
    background-color: rgba(255, 255, 255, 0.95);
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    text-align: center;
    margin: 2rem auto;
    max-width: 500px;
}

#start-game, #new-game {
    background-color: #3498db;
    color: white;
    margin-top: 1rem;
    padding: 1rem 2rem;
}

footer {
    margin-top: auto;
}

.hidden {
    display: none !important;
}
                """)
        
        if not os.path.exists(js_file):
            with open(js_file, 'w') as f:
                f.write("""
// Constants
const API_URL = 'http://localhost:9001/api';
let sessionId = null;
let currentCard = null;
let gameOver = false;

// DOM Elements
const powerMeter = document.getElementById('power-meter');
const stabilityMeter = document.getElementById('stability-meter');
const pietyMeter = document.getElementById('piety-meter');
const wealthMeter = document.getElementById('wealth-meter');

const powerValue = document.getElementById('power-value');
const stabilityValue = document.getElementById('stability-value');
const pietyValue = document.getElementById('piety-value');
const wealthValue = document.getElementById('wealth-value');

const reignYear = document.getElementById('reign-year');
const cardContainer = document.getElementById('card-container');
const cardText = document.getElementById('card-text');
const yesButton = document.getElementById('yes-button');
const noButton = document.getElementById('no-button');
const startScreen = document.getElementById('start-screen');
const startGameButton = document.getElementById('start-game');
const gameOverScreen = document.getElementById('game-over');
const gameOverReason = document.getElementById('game-over-reason');
const reignSummary = document.getElementById('reign-summary');
const newGameButton = document.getElementById('new-game');

// Game state
let metrics = {
    power: 50,
    stability: 50,
    piety: 50,
    wealth: 50,
    reign_year: 1
};

let trajectory = [];

// Event listeners
startGameButton.addEventListener('click', startGame);
yesButton.addEventListener('click', () => makeChoice('yes'));
noButton.addEventListener('click', () => makeChoice('no'));
newGameButton.addEventListener('click', startGame);

// Helper functions
function updateMeters() {
    powerMeter.style.width = `${metrics.power}%`;
    stabilityMeter.style.width = `${metrics.stability}%`;
    pietyMeter.style.width = `${metrics.piety}%`;
    wealthMeter.style.width = `${metrics.wealth}%`;
    
    powerValue.textContent = metrics.power;
    stabilityValue.textContent = metrics.stability;
    pietyValue.textContent = metrics.piety;
    wealthValue.textContent = metrics.wealth;
    
    reignYear.textContent = metrics.reign_year;
    
    // Change colors when values get dangerous
    if (metrics.power <= 20 || metrics.power >= 80) {
        powerMeter.style.backgroundColor = '#ff5252';
    } else {
        powerMeter.style.backgroundColor = 'var(--power-color)';
    }
    
    if (metrics.stability <= 20 || metrics.stability >= 80) {
        stabilityMeter.style.backgroundColor = '#ff9800';
    } else {
        stabilityMeter.style.backgroundColor = 'var(--stability-color)';
    }
    
    if (metrics.piety <= 20 || metrics.piety >= 80) {
        pietyMeter.style.backgroundColor = '#ff9800';
    } else {
        pietyMeter.style.backgroundColor = 'var(--piety-color)';
    }
    
    if (metrics.wealth <= 20 || metrics.wealth >= 80) {
        wealthMeter.style.backgroundColor = '#ff9800';
    } else {
        wealthMeter.style.backgroundColor = 'var(--wealth-color)';
    }
}

async function startGame() {
    try {
        // Reset game state
        gameOver = false;
        trajectory = [];
        
        // Create new game session
        const response = await fetch(`${API_URL}/new_game`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        sessionId = data.session_id;
        metrics = data.metrics;
        
        updateMeters();
        
        // Hide start screen, show game screen
        startScreen.classList.add('hidden');
        gameOverScreen.classList.add('hidden');
        cardContainer.classList.remove('hidden');
        
        // Generate first card
        await generateCard();
        
    } catch (error) {
        console.error("Error starting game:", error);
        alert("Failed to start game. Please check your connection to the game server.");
    }
}

async function generateCard() {
    try {
        const response = await fetch(`${API_URL}/generate_card`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });
        
        currentCard = await response.json();
        
        // Update card UI
        cardText.textContent = currentCard.text;
        yesButton.textContent = currentCard.yes_option;
        noButton.textContent = currentCard.no_option;
        
    } catch (error) {
        console.error("Error generating card:", error);
        cardText.textContent = "Something went wrong. Please try again.";
    }
}

async function makeChoice(choice) {
    try {
        const response = await fetch(`${API_URL}/card_choice`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                choice: choice
            })
        });
        
        const data = await response.json();
        
        // Record this move in trajectory
        trajectory.push({
            card_id: currentCard.id,
            category: currentCard.category,
            choice: choice,
            effects: currentCard.effects[choice],
            post_metrics: data.metrics
        });
        
        // Update game state
        metrics = data.metrics;
        updateMeters();
        
        // Check for game over
        if (data.game_over) {
            endReign();
            return;
        }
        
        // Generate next card
        await generateCard();
        
    } catch (error) {
        console.error("Error processing choice:", error);
        cardText.textContent = "Something went wrong. Please try again.";
    }
}

async function endReign() {
    try {
        // Determine cause of end
        let cause = null;
        if (metrics.power <= 0) cause = "power";
        else if (metrics.power >= 100) cause = "power";
        else if (metrics.stability <= 0) cause = "stability";
        else if (metrics.stability >= 100) cause = "stability";
        else if (metrics.piety <= 0) cause = "piety";
        else if (metrics.piety >= 100) cause = "piety";
        else if (metrics.wealth <= 0) cause = "wealth";
        else if (metrics.wealth >= 100) cause = "wealth";
        
        // Send end reign data to server
        const response = await fetch(`${API_URL}/end_reign`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                trajectory: trajectory,
                final_metrics: metrics,
                reign_length: metrics.reign_year,
                cause_of_end: cause
            })
        });
        
        const data = await response.json();
        
        // Show game over screen
        cardContainer.classList.add('hidden');
        gameOverScreen.classList.remove('hidden');
        
        // Set reason based on metrics
        let reason = "";
        if (metrics.power <= 0) reason = "You lost all authority. The nobles overthrew you!";
        else if (metrics.power >= 100) reason = "Your absolute power made you a tyrant. You were assassinated!";
        else if (metrics.stability <= 0) reason = "The people revolted against your rule!";
        else if (metrics.stability >= 100) reason = "The people loved you so much they established a republic!";
        else if (metrics.piety <= 0) reason = "The church declared you a heretic and had you executed!";
        else if (metrics.piety >= 100) reason = "The church became too powerful and took control of your kingdom!";
        else if (metrics.wealth <= 0) reason = "Your kingdom went bankrupt and you were deposed!";
        else if (metrics.wealth >= 100) reason = "Your vast wealth attracted invaders who conquered your kingdom!";
        
        gameOverReason.textContent = reason;
        reignSummary.textContent = `You ruled for ${metrics.reign_year} years. Final reward: ${data.reward.toFixed(2)}`;
        
    } catch (error) {
        console.error("Error ending reign:", error);
        gameOverReason.textContent = "Something went wrong when calculating your legacy.";
    }
}

// Check if API is available when page loads
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_URL}`);
        await response.json();
        console.log("API connection successful");
    } catch (error) {
        console.error("API connection failed:", error);
        cardText.textContent = "Cannot connect to game server. Please ensure the server is running.";
    }
});
                """)
    
    # Start API server in a separate thread
    print(f"Starting API server on port {args.api_port}")
    api_thread = Thread(target=run_api_server, args=(args.api_port,))
    api_thread.daemon = True
    api_thread.start()
    
    # Give the API server time to start
    time.sleep(2)
    
    # Start web server for frontend
    print(f"Starting web server on port {args.web_port}")
    web_url = f"http://localhost:{args.web_port}"
    
    # Open the browser if requested
    if not args.no_open:
        print(f"Opening {web_url} in your default browser")
        webbrowser.open(web_url)
    
    # Print instructions
    print("\n=== DynastAI Local Server ===")
    print(f"API Server: http://localhost:{args.api_port}/api")
    print(f"Web Frontend: {web_url}")
    print("Press Ctrl+C to stop the servers")
    
    # Run the web server in the main thread
    run_web_server(args.web_port, static_dir)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down servers...")
        sys.exit(0)