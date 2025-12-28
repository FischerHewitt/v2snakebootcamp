# Snake Bootcamp - Backend Guide

Welcome to the **CSAI Snake Bootcamp Backend**! This is a **guided learning project** where you'll build a Python WebSocket server that runs an AI-powered Snake game.

## Table of Contents

- [What You'll Learn](#what-youll-learn)
- [Project Overview](#project-overview)
- [Backend Architecture Deep Dive](#backend-architecture-deep-dive)
- [Our Teaching Philosophy: Comments Over Complexity](#our-teaching-philosophy-comments-over-complexity)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [What You Need to Implement](#what-you-need-to-implement)
- [Learning Path: What to Implement First](#learning-path-what-to-implement-first)
- [AI Concepts Explained](#ai-concepts-explained)
- [Key Technologies You'll Use](#key-technologies-youll-use)
- [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
- [Common Issues & Solutions](#common-issues--solutions)
- [Going Further (Optional Enhancements)](#going-further-optional-enhancements)
- [Tips for Success](#tips-for-success)
- [Understanding AI Performance](#understanding-ai-performance)
- [Resources](#resources)
- [Success Criteria](#success-criteria)
- [What You'll Accomplish](#what-youll-accomplish)

## What You'll Learn

This project will teach you:

- **WebSocket servers** with Python and SocketIO
- **Game development** fundamentals and game loops
- **Artificial Intelligence** with reinforcement learning
- **Neural networks** using PyTorch
- **Asynchronous programming** in Python
- **Real-time communication** between frontend and backend

## Project Overview

You're building a **game server** that:

1. Runs Snake game simulations
2. Uses AI agents to play the game autonomously
3. Sends real-time updates to frontend clients via WebSocket
4. Trains neural networks to improve AI performance

**This is NOT a build-from-scratch project** - we've provided the structure and guidance. Your job is to **fill in the implementation** where you see `TODO` comments!

## Backend Architecture Deep Dive

Understanding how all the pieces fit together:

### Component Relationships

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   app.py        │    │   game.py       │
│   (React)       │◄──►│   (WebSocket    │◄──►│   (Game Logic)  │
│                 │    │    Server)      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   agent.py      │    │ snake.py        │
                       │   (AI Brain)    │    │ food.py         │
                       │                 │    │ (Game Entities) │
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   model.py      │
                       │   (Neural Net)  │
                       │                 │
                       └─────────────────┘
```

### Data Flow Architecture

1. **Client Connection** → `app.py` handles WebSocket connections
2. **Game Initialization** → Creates `Game()` and `DQN()` instances
3. **AI Decision Loop**:
   - `game.py` provides current state
   - `agent.py` processes state through neural network
   - `model.py` predicts best action
   - `agent.py` applies action to game
   - `game.py` updates physics and checks collisions
4. **Real-time Updates** → `app.py` broadcasts state to all connected clients
5. **Learning Process** → `agent.py` stores experiences and trains `model.py`

### Session Management

Each connected client gets their own game session stored in SocketIO:

```python
# Session structure:
session = {
    'game': Game(),           # Individual game instance
    'agent': ...,           # Individual AI agent
    'statistics': ...         # Training metrics
}
```

## Our Teaching Philosophy: Comments Over Complexity

This bootcamp follows a **guided discovery approach** designed to help you learn AI concepts without becoming overwhelmed:

### Why We Use Extensive Comments

- **Learn by implementing**: You see the architecture and logic, then implement the missing pieces
- **Real understanding**: Comments try to explain _why_ something works, not just _what_ it does
- **Confidence building**: Clear guidance prevents frustration and builds momentum

### What's Already Implemented vs. What You'll Build

#### **Provided Foundation** (Focus on learning, not debugging)

- **Complete game engine** (`game.py`, `snake.py`, `food.py`) - all snake game mechanics implemented
- **Detailed class structures** with method signatures and docstrings
- **Educational comments** explaining AI concepts and neural network theory
- **Error handling and edge cases** to prevent common beginner mistakes

#### **Your Implementation Tasks** (The learning happens here)

- **WebSocket event handlers** - understand real-time communication
- **Neural network forward pass** - see how data flows through networks
- **Reward function design** - learn what motivates AI behavior
- **Training loop implementation** - experience how AI learns from experience

### Reinforcement Learning Made Accessible

**The goal**: You should understand every line of code you write and why it matters for the AI's learning process.

## Project Structure

Your backend has this structure:

```
backend/
├── src/
│   ├── app.py          # WebSocket server (lots of TODOs!)
│   ├── agent.py        # AI agent class (implement the brain!)
│   ├── model.py        # Neural network models (build the network!)
│   ├── game.py         # Game controller (already working!)
│   ├── snake.py        # Snake entity (already working!)
│   └── food.py         # Food entity (already working!)
└── requirements.txt    # Dependencies
```

## Getting Started

### 1. Set Up Python Environment

Create a virtual environment:

```bash
cd apps/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Server

```bash
python src/app.py
```

Your server will run at `http://localhost:8765`

### 4. Find the TODOs

Open the Python files and look for `TODO:` comments - these guide you through implementing the functionality!

## What You Need to Implement

### 1. **WebSocket Server** (`app.py`)

**What's provided**: Basic structure and helpful comments
**What you implement**:

- SocketIO server setup
- Event handlers for client connections
- Game session management
- Real-time update broadcasting

**Key TODOs**:

```python
# TODO: Create a SocketIO server instance with CORS settings
# TODO: Create event handlers for connect/disconnect
# TODO: Implement the main game loop
# TODO: Handle AI agent integration
```

### 2. **AI Agent** (`agent.py`)

**What's provided**: Complete class structure with educational comments
**What you implement**:

- State representation (convert game to neural network input)
- Reward calculation (teach the AI what's good/bad)
- Action selection (epsilon-greedy strategy)
- Memory and training systems

**Key concepts you'll learn**:

- **Deep Q-Networks (DQN)**: The AI learning algorithm
- **Epsilon-greedy exploration**: Balancing exploration vs exploitation
- **Experience replay**: Learning from past experiences
- **Neural network training**: Backpropagation and gradient descent

### 3. **Neural Network Model** (`model.py`)

**What's provided**: Complete structure with detailed explanations
**What you implement**:

- PyTorch neural network architecture
- Forward pass through the network
- Training loop with loss calculation
- Model saving and loading

**Key concepts you'll learn**:

- **Feedforward neural networks**: How data flows through layers
- **Q-learning**: The mathematical foundation of the AI
- **Bellman equation**: How future rewards are calculated
- **PyTorch fundamentals**: Building and training networks

## Learning Path: What to Implement First

### **Phase 1: Get the Server Running** (Start Here!)

1. **Basic WebSocket Setup** (`app.py`)

   - Import the required libraries
   - Create SocketIO server instance
   - Add connection event handlers
   - Test that clients can connect

2. **Simple Game Broadcasting**
   - Create a basic game session
   - Send game state updates to clients
   - Test with the frontend

### **Phase 2: Add AI Intelligence** (The Fun Part!)

3. **State Representation** (`agent.py`)

   - Convert game state to numerical features
   - Implement danger detection
   - Calculate distances to food

4. **Neural Network** (`model.py`)

   - Build the PyTorch network
   - Implement forward pass
   - Add training functionality

5. **AI Decision Making** (`agent.py`)
   - Implement action selection
   - Add epsilon-greedy exploration
   - Connect AI to game actions

### **Phase 3: Advanced Training** (Level Up!)

6. **Reward System**

   - Design reward functions
   - Balance exploration vs exploitation
   - Optimize for learning speed

7. **Experience Replay**
   - Implement memory system
   - Add batch training
   - Tune hyperparameters

## AI Concepts Explained

### Deep Q-Network (DQN)

The AI uses a **neural network** to estimate how good each action is in each game state. It learns by:

1. **Playing the game** and observing results
2. **Getting rewards** for good actions (eating food = +10, dying = -10)
3. **Training the network** to predict better rewards
4. **Improving over time** through experience

### Game State Features

Your AI needs to convert the visual game into numerical inputs. Choose features that help the AI understand the game situation, such as:

    ```
    Example features you might use:
    - Danger detection (collision risks)
    - Food direction (where's the food?)
    - Food distances (how far is it?)
    - Current direction (which way am I facing?)
    - Snake length, game boundaries, etc.
    ```

Research and experiment with different feature combinations to see what helps your AI learn best!

### Actions (3 outputs)

The AI can choose from 3 actions:

- **[1,0,0]**: Go straight
- **[0,1,0]**: Turn right
- **[0,0,1]**: Turn left

## Key Technologies You'll Use

### Socket.IO for Python

You'll learn to create WebSocket servers for real-time communication:

- Set up async servers with `socketio.AsyncServer()`
- Handle events like `connect`, `disconnect`, `start_game`
- Manage client sessions and broadcast updates
- Research: Socket.IO documentation and async programming patterns

### PyTorch for AI

You'll build neural networks from scratch:

- Create network architectures with `nn.Module`
- Implement forward passes and training loops
- Use optimizers like Adam and loss functions like MSE
- Research: PyTorch tutorials, Q-learning theory, and neural network basics

### Async Programming

You'll master Python's async/await pattern:

- Handle concurrent game sessions
- Manage real-time updates without blocking
- Research: `asyncio` documentation and event-driven programming

## Step-by-Step Implementation Guide

### Step 1: WebSocket Basics

1. Open `app.py` and find the TODO comments
2. Import `socketio` and `aiohttp.web`
3. Create server instances and attach them
4. Implement `connect` and `disconnect` event handlers
5. Test with a WebSocket client

### Step 2: Game Integration

1. Implement the `start_game` event handler
2. Create game sessions for each client
3. Start the game update loop
4. Send game state to clients

### Step 3: AI Integration

1. Open `agent.py` and implement `get_state()`
2. Build the neural network in `model.py`
3. Implement `get_action()` to choose moves
4. Connect the AI to the game loop

### Step 4: Training System

1. Implement reward calculation
2. Add experience replay memory
3. Create training loops
4. Tune hyperparameters for learning

## Common Issues & Solutions

### Import Errors

- Make sure all packages are installed: `pip install -r requirements.txt`
- Check your virtual environment is activated
- Verify Python version compatibility

### WebSocket Connection Issues

- Ensure server starts on the correct port (8765)
- Check CORS settings allow frontend connections
- Look at server logs for connection errors

### AI Not Learning

- Start with simple rewards (food = +10, death = -10)
- Check that state representation makes sense
- Verify neural network forward pass works
- Monitor epsilon decay (exploration vs exploitation)

### Training Too Slow

- Increase batch size for more stable training
- Adjust learning rate
- Try different network architectures
- Check reward scaling

## Going Further (Optional Enhancements)

### Beginner Enhancements

- Add game statistics tracking
- Implement model saving/loading
- Create different AI difficulty levels

### Intermediate Enhancements

- Multiple AI agents competing
- Different reward strategies
- Advanced neural network architectures

### Advanced Enhancements

- Distributed training across multiple games
- Advanced RL algorithms (Double DQN, Dueling DQN)
- Real-time learning analytics dashboard

## Tips for Success

1. **Start Simple**: Get basic WebSocket working before adding AI
2. **Test Frequently**: Print game states and AI decisions to debug
3. **Monitor Training**: Watch scores improve over time
4. **Experiment**: Try different rewards and network sizes
5. **Ask Questions**: Don't hesitate to ask for help!

## Understanding AI Performance

### Good Signs Your AI is Learning

- Average score increases over time
- Less random movement as epsilon decreases
- Longer games (snake gets bigger before dying)
- Consistent food-seeking behavior

### Troubleshooting Poor Performance

- **Scores not improving**: Check reward function
- **Too random behavior**: Verify epsilon decay
- **Getting stuck**: Adjust exploration rate
- **Learning too slow**: Increase learning rate or batch size

## Resources

- [Socket.IO Python Docs](https://python-socketio.readthedocs.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Q-Learning Explained](https://www.youtube.com/watch?v=79pmNdyxEGo)
- [Reinforcement Learning Course](https://spinningup.openai.com/en/latest/)

## Success Criteria

Your backend is complete when:

- WebSocket server accepts connections from frontend
- Game sessions run independently for each client
- AI agent makes decisions and controls the snake
- Neural network trains and improves over time
- Real-time updates sent to frontend smoothly
- Code is well-commented and demonstrates understanding

## What You'll Accomplish

By completing this project, you'll have built:

- A **real-time multiplayer game server**
- An **AI agent** that learns through reinforcement learning
- A **neural network** trained from scratch
- A **WebSocket communication system**
- Understanding of **modern AI/ML concepts**

---

**Remember**: This project combines **game development**, **artificial intelligence**, and **web technologies**. Take your time, experiment with different approaches, and don't be afraid to try creative solutions!

**Need help?** The existing code has extensive comments, and club members are here to support you. The goal is to learn while building something genuinely impressive!
