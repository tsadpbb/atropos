/**
 * DynastAI - Game Client 
 * 
 * This JavaScript file handles the client-side logic for the DynastAI game:
 * - Communicating with the API
 * - Updating the UI based on game state
 * - Handling user interactions
 */

// Configuration
const API_URL = 'http://localhost:9001/api';
let sessionId = null;
let currentCard = null;
let gameOver = false;

// Track game state
let metrics = {
    power: 50,
    stability: 50,
    piety: 50,
    wealth: 50,
    reign_year: 1
};

let trajectory = [];

// DOM Elements
const powerMeter = document.getElementById('power-meter');
const stabilityMeter = document.getElementById('stability-meter');
const pietyMeter = document.getElementById('piety-meter');
const wealthMeter = document.getElementById('wealth-meter');

const powerValue = document.getElementById('power-value');
const stabilityValue = document.getElementById('stability-value');
const pietyValue = document.getElementById('piety-value');
const wealthValue = document.getElementById('wealth-value');

const powerEffect = document.getElementById('power-effect');
const stabilityEffect = document.getElementById('stability-effect');
const pietyEffect = document.getElementById('piety-effect');
const wealthEffect = document.getElementById('wealth-effect');

const effectsDisplay = document.getElementById('effects-display');
const categoryIndicator = document.getElementById('category-indicator');
const reignYear = document.getElementById('reign-year');
const cardContainer = document.getElementById('card-container');
const cardText = document.getElementById('card-text');
const yesButton = document.getElementById('yes-button');
const noButton = document.getElementById('no-button');
const startScreen = document.getElementById('start-screen');
const startGameButton = document.getElementById('start-game');
const gameOverScreen = document.getElementById('game-over');
const gameOverReason = document.getElementById('game-over-reason');
const finalPower = document.getElementById('final-power');
const finalStability = document.getElementById('final-stability');
const finalPiety = document.getElementById('final-piety');
const finalWealth = document.getElementById('final-wealth');
const reignSummary = document.getElementById('reign-summary');
const newGameButton = document.getElementById('new-game');
const apiStatus = document.getElementById('api-status');

// Event listeners
startGameButton.addEventListener('click', startGame);
yesButton.addEventListener('click', () => makeChoice('yes'));
noButton.addEventListener('click', () => makeChoice('no'));
newGameButton.addEventListener('click', startGame);

/**
 * Check API availability and update the status indicator
 */
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_URL}`);
        await response.json();
        apiStatus.textContent = 'Connected';
        apiStatus.classList.add('online');
        return true;
    } catch (error) {
        console.error("API connection failed:", error);
        apiStatus.textContent = 'Disconnected';
        apiStatus.classList.add('offline');
        return false;
    }
}

/**
 * Update UI meters and values based on current metrics
 */
function updateMeters() {
    // Update meter widths
    powerMeter.style.width = `${metrics.power}%`;
    stabilityMeter.style.width = `${metrics.stability}%`;
    pietyMeter.style.width = `${metrics.piety}%`;
    wealthMeter.style.width = `${metrics.wealth}%`;
    
    // Update displayed values
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

/**
 * Display choice effects on the UI
 */
function displayEffects(effects) {
    // Update effect values
    powerEffect.textContent = formatEffect(effects.power);
    stabilityEffect.textContent = formatEffect(effects.stability);
    pietyEffect.textContent = formatEffect(effects.piety);
    wealthEffect.textContent = formatEffect(effects.wealth);
    
    // Show effects display
    effectsDisplay.classList.remove('hidden');
    
    // Hide after 3 seconds
    setTimeout(() => {
        effectsDisplay.classList.add('hidden');
    }, 3000);
}

/**
 * Format effect for display (+5, -3, etc)
 */
function formatEffect(value) {
    if (value > 0) return '+' + value;
    return value.toString();
}

/**
 * Start a new game session
 */
async function startGame() {
    try {
        // Check API availability
        const apiAvailable = await checkApiStatus();
        if (!apiAvailable) {
            cardText.textContent = "Cannot connect to game server. Please ensure the server is running.";
            return;
        }
        
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

/**
 * Generate a new card
 */
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
        
        // Update category indicator
        categoryIndicator.className = '';
        categoryIndicator.classList.add(currentCard.category);
        
    } catch (error) {
        console.error("Error generating card:", error);
        cardText.textContent = "Something went wrong generating the next scenario. Please try again.";
    }
}

/**
 * Make a card choice (yes/no)
 */
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
        
        // Display choice effects
        displayEffects(currentCard.effects[choice]);
        
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
        cardText.textContent = "Something went wrong processing your choice. Please try again.";
    }
}

/**
 * End the current reign and calculate final results
 */
async function endReign() {
    try {
        // Determine cause of end
        let cause = null;
        if (metrics.power <= 0) cause = "power_low";
        else if (metrics.power >= 100) cause = "power_high";
        else if (metrics.stability <= 0) cause = "stability_low";
        else if (metrics.stability >= 100) cause = "stability_high";
        else if (metrics.piety <= 0) cause = "piety_low";
        else if (metrics.piety >= 100) cause = "piety_high";
        else if (metrics.wealth <= 0) cause = "wealth_low";
        else if (metrics.wealth >= 100) cause = "wealth_high";
        
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
        
        // Set the final metric values
        finalPower.textContent = metrics.power;
        finalStability.textContent = metrics.stability;
        finalPiety.textContent = metrics.piety;
        finalWealth.textContent = metrics.wealth;
        
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
    await checkApiStatus();
});