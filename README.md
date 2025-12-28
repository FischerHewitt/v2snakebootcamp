# CSAI Fall 2025 Snake Bootcamp

This repository features a machine learning-powered Snake game, seamlessly integrated with a Next.js frontend using Socket.IO for real-time communication. Participants will implement AI agents using Deep Q-Networks (DQN) to train snakes that learn to play autonomously.

## Table of Contents

- [What You'll Build](#what-youll-build)
- [Project Architecture](#project-architecture)
- [Why We Need the Backend - The AI Game Engine](#why-we-need-the-backend---the-ai-game-engine)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Getting Started - Key Implementation Concepts](#getting-started---key-implementation-concepts)
- [File-by-File Implementation Guide](#file-by-file-implementation-guide)
- [Optional Challenges for Advanced Participants](#optional-challenges-for-advanced-participants)

## What You'll Build

By the end of this bootcamp, you'll have created:

- A **real-time multiplayer Snake game** with WebSocket communication
- An **AI agent** (`apps/backend/src/agent.py`) that learns through reinforcement learning
- A **neural network** (`apps/backend/src/model.py`) trained from scratch using PyTorch
- A **responsive web interface** (`apps/frontend/app/page.tsx`) for visualizing AI gameplay
- Understanding of modern AI/ML concepts and full-stack development

## Project Architecture

This project consists of two main components that work together:

### Backend (`apps/backend/`) - The AI Brain

- **`src/app.py`** - WebSocket server with event handlers (`connect`, `start_game`, `update_game`)
- **`src/agent.py`** - DQN agent class with methods like `get_state()`, `get_action()`, `train_long_memory()`
- **`src/model.py`** - PyTorch neural network (`LinearQNet`) and training logic (`QTrainer`)
- **`src/game.py`** - Game controller with `step()`, `reset()`, and state management (Working)
- **`src/snake.py`** - Snake entity with movement and collision detection (Working)
- **`src/food.py`** - Food generation and collision checking (Working)

### Frontend (`apps/frontend/`) - The Visual Interface

- **`app/page.tsx`** - Main game canvas with Socket.IO client and drawing functions
- **`components/`** - Reusable UI components for theming and layout

## Why We Need the Backend - The AI Game Engine

The backend serves as the **intelligent game engine** that powers the AI-driven Snake experience. Here's why it's essential:

### Real-Time AI Decision Making

The backend runs the **DQN (Deep Q-Network) agent** that makes split-second decisions about where the snake should move. Unlike traditional games where humans control the snake, our AI agent:

- **Analyzes game state** through the `get_state()` function in `agent.py`
- **Chooses optimal actions** using neural network predictions (`get_action()` method)
- **Learns from experience** through reinforcement learning (`train_long_memory()`)

### Game State Authority

The backend maintains the **single source of truth** for the game state:

- **Game physics** are computed server-side in `game.py` (`step()`, collision detection)
- **Score tracking** and game progression managed centrally
- **Multiple clients** can connect and watch the same AI agent play
- **Prevents cheating** since game logic isn't exposed to frontend

### Machine Learning Pipeline

The backend implements a complete **AI training system**:

- **Experience replay** memory system stores past game states for learning
- **Neural network training** happens in real-time as the snake plays
- **Epsilon-greedy exploration** balances trying new moves vs. using learned knowledge
- **Reward calculation** teaches the AI what constitutes good/bad gameplay

### WebSocket Communication Hub

The backend broadcasts **real-time updates** to connected frontends:

- **Game state streaming** via `update_game()` function in `app.py`
- **Multiple viewers** can watch the AI learn simultaneously
- **Low-latency updates** for smooth gameplay visualization
- **Event-driven architecture** with handlers for connection, game start, etc.

**Without the backend**: You'd have just a static frontend with no AI, no learning, and no real-time gameplay. The backend is where the magic happens!

## Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** (v18 or higher): [Download Node.js](https://nodejs.org/)
- **npm** (comes with Node.js)
- **Python** (version 3.9): [Download Python](https://www.python.org/downloads/release/python-390/)

You can verify installations with:

```bash
node -v
npm -v
python3 --version
```

## Getting Started

Follow these steps to run the project locally on **macOS**, **Windows**, or **Linux**.

### Frontend

```bash
cd apps/frontend
npm install
npm run dev
```

### Backend

#### Create and activate a Python virtual environment

- **macOS/Linux:**

  ```bash
  cd apps/backend
  python3 -m venv .venv
  source .venv/bin/activate
  ```

- **Windows (Command Prompt):**

  ```cmd
  cd apps\backend
  python -m venv .venv
  .venv\Scripts\activate
  ```

- **Windows (PowerShell):**

  ```powershell
  cd apps\backend
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```

#### Install dependencies and start the backend

```bash
pip install -r requirements.txt
python src/app.py
```

Once both servers are running, open [http://localhost:3000](http://localhost:3000) in your browser.

## Getting Started - Key Implementation Concepts

Here are the essential concepts and minimal starter code to guide your implementation:

### Backend WebSocket Server Setup (`apps/backend/src/app.py`)

```python
# Essential imports you'll need
import asyncio
import socketio
from aiohttp import web
from game import Game
from agent import DQN

# Basic server setup
sio = socketio.AsyncServer(cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

@sio.event
async def connect(sid, environ):
    print(f"Client {sid} connected")
    # TODO: Initialize game and agent for this client

@sio.event
async def start_game(sid, data):
    # TODO: Create Game() and DQN() instances
    # TODO: Save to session and start game loop
    pass
```

### AI State Representation Concepts (`apps/backend/src/agent.py`)

Your AI needs to "see" the game world as numbers. Design a `get_state()` function that converts the visual game into numerical features. Consider including:

- **Danger detection**: Is there danger in different directions relative to the snake's current heading?
- **Food direction**: Where is the food located relative to the snake head?
- **Distance information**: How far is the food from the snake?
- **Current direction**: Which way is the snake currently moving?
- **Snake body information**: What about the snake's own body positioning?

**Your challenge**: Decide how many features you need and how to extract them from the `game` object!

### Neural Network Design Goals (`apps/backend/src/model.py`)

Build a PyTorch neural network that:

- Takes your chosen number of input features (your state representation)
- Has one or more hidden layers (experiment with different sizes)
- Outputs 3 Q-values (for straight, right, left actions)
- Uses appropriate activation functions and loss functions

**Key concepts to research**: Q-learning, neural network forward pass, PyTorch basics

### Frontend Connection Strategy (`apps/frontend/app/page.tsx`)

Your React component should:

- Connect to WebSocket server at `localhost:8765`
- Listen for game state updates and render them on HTML5 canvas
- Handle connection events and game initialization
- Draw the snake, food, and game grid in real-time

**Your challenge**: Figure out the Socket.IO client setup and canvas drawing logic!

## File-by-File Implementation Guide

Understanding what each file does and what you need to implement:

### Files You Need to Edit (Your Implementation Tasks)

#### `apps/backend/src/app.py` - WebSocket Server

**What it does**: Main server that handles client connections and runs the game loop

**Your tasks**:

- Complete the `connect()` event handler to initialize client sessions
- Implement `start_game()` to create Game and DQN agent instances
- Build the `update_game()` loop for real-time AI gameplay
- Add `disconnect()` cleanup for when clients leave

**Learning focus**: Real-time communication, session management, async programming

#### `apps/backend/src/agent.py` - AI Brain

**What it does**: The DQN agent that learns to play Snake through reinforcement learning

**Your tasks**:

- Design `get_state()` to convert game data into 13 neural network features
- Implement `get_action()` with epsilon-greedy exploration strategy
- Build `calculate_reward()` to teach the AI good vs bad moves
- Add `remember()` and training functions for experience replay

**Learning focus**: State representation, reward engineering, reinforcement learning concepts

#### `apps/backend/src/model.py` - Neural Network

**What it does**: PyTorch neural network that predicts Q-values for each action

**Your tasks**:

- Build the `LinearQNet` class with proper layer architecture
- Implement `forward()` method for neural network inference
- Complete `QTrainer` with loss calculation and backpropagation
- Add model saving/loading for persistent learning

**Learning focus**: Neural network architecture, PyTorch basics, gradient descent

#### `apps/frontend/app/page.tsx` - Game Visualization

**What it does**: React component that displays the AI playing Snake in real-time

**Your tasks**:

- Set up Socket.IO connection to backend server
- Implement canvas drawing functions for snake, food, and grid
- Add state management for game data (snake position, score, etc.)
- Create responsive design for different screen sizes

**Learning focus**: WebSocket clients, HTML5 canvas, React state management

### Files Already Working (Reference Only)

#### `apps/backend/src/game.py` - Game Controller

**What it does**: Manages core game mechanics and state

**Key functions you can use**:

- `game.step()` - Advance game by one frame
- `game.reset()` - Start a new game round
- `game.send()` - Get current state for broadcasting
- `game.queue_change()` - Handle direction changes

#### `apps/backend/src/snake.py` - Snake Entity

**What it does**: Handles snake movement, growth, and collision detection

**Key properties you can access**:

- `snake.head` - Current head position
- `snake.body` - List of all body segments
- `snake.direction` - Current movement direction
- `snake.grow` - Whether snake is growing this frame

#### `apps/backend/src/food.py` - Food Management

**What it does**: Spawns food in random locations and detects when eaten

**Key properties you can access**:

- `food.position` - Current food coordinates
- `food.check_eaten()` - Test if snake ate food this frame

## Optional Challenges for Advanced Participants

Once you've completed the basic implementation, try these advanced challenges to level up your skills:

### Challenge 1: Training Optimization Laboratory

**Goal**: Experiment with different AI training strategies to find the optimal snake

**What to implement**:

- Create multiple DQN agents with different hyperparameters
- Compare performance across different configurations
- Track training metrics and analyze results

**Learning outcomes**: Hyperparameter tuning, experimental design, data analysis

### Challenge 2: Multi-Agent Snake Battle Arena

**Goal**: Create multiple AI snakes competing in the same environment

**What to implement**:

- Modify game logic to support multiple snakes
- Design competitive reward systems
- Implement tournament-style training

**Learning outcomes**: Multi-agent systems, competitive AI, game theory

### Challenge 3: Production Deployment Pipeline

**Goal**: Deploy your AI Snake game to the cloud for others to watch and interact with

**What to implement**:

- Containerize with Docker and deploy to cloud platforms
- Add authentication and leaderboards
- Implement production monitoring and logging

**Learning outcomes**: DevOps, cloud deployment, production systems, scalability

### Bonus Challenges

**For the truly ambitious**:

- **Advanced AI**: Implement Double DQN, Dueling DQN, or Rainbow DQN
- **Computer Vision**: Train an AI that plays by analyzing screen pixels
- **Evolutionary Algorithms**: Breed the best AI snakes using genetic algorithms
- **Real-time Analytics**: Create ML dashboard showing training metrics

### Getting Help with Challenges

- **Documentation**: Each challenge includes starter guidance and architecture suggestions
- **Club Support**: Advanced challenges are perfect for pair programming sessions
- **Showcase**: Present your completed challenges to the CSAI community
- **Open Source**: Contribute your solutions back to help future participants

**Remember**: These challenges are designed to be **portfolio-worthy projects** that demonstrate advanced AI/ML and full-stack development skills to potential employers!
