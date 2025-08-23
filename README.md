# Deep Reinforcement Learning - Space Invaders

A comprehensive Deep Q-Network (DQN) implementation for training an AI agent to master Atari Space Invaders using PyTorch and OpenAI Gymnasium.

## Project Overview

This project implements a **Double Deep Q-Network (DDQN)** agent that learns to play Space Invaders through reinforcement learning. The implementation features advanced training techniques, comprehensive metrics tracking, hyperparameter optimization, and robust checkpoint management.

### Key Features

- ** Double Deep Q-Network (DDQN)** with target network stabilization
- ** Experience Replay Buffer** for improved sample efficiency
- ** Custom Reward Shaping** with life loss and missed shot penalties
- ** Advanced Metrics Tracking** with TensorBoard integration
- ** Automatic Checkpointing** and seamless training resumption
- ** Frame Stacking & Preprocessing** for enhanced state representation
- ** Epsilon-Greedy Exploration** with exponential decay scheduling
- ** Hyperparameter Optimization** using Optuna framework
- ** Comprehensive Visualization** tools for training analysis

## Project Architecture

```
DeepRL_SpaceInvaders_project/
├── src/
│   ├── agent.py                    # DDQN Agent implementation
│   ├── environment.py              # Environment wrapper and preprocessing
│   ├── utils.py                    # Utility functions for paths and logging
│   └── training/
│       ├── train.py                # Main training script
│       ├── hyperparamoptim.py      # Hyperparameter optimization
│       ├── metrics.py              # Metrics tracking and analysis
│       ├── plot.py                 # Visualization utilities
│       └── config.yaml             # Training configuration parameters
├── logs/                           # Training logs and checkpoints
├── plots/                          # Generated training visualizations
├── tensorboard_logs/               # TensorBoard monitoring logs
├── models/                         # Saved model weights
├── requirements.txt                # Python dependencies
└── README.md                      # This documentation
```


## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd DeepRL_SpaceInvaders_project
   ```

2. **Create and activate conda environment:**
   ```bash
   conda create -n rl python=3.10
   conda activate rl
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```


## Usage Guide

### Training the Agent

Start training with default configuration:
```bash
python src/training/train.py
```


### Hyperparameter Optimization

Optimize hyperparameters using Optuna:
```bash
python src/training/hyperparamoptim.py
```

### Monitoring Training

Launch TensorBoard for real-time monitoring:
```bash
tensorboard --logdir=tensorboard_logs
```

## Model Architecture

### DDQN Agent Components

1. ** Main Network**: Current Q-value estimation
2. ** Target Network**: Stable Q-value targets (soft updates)
3. ** Experience Replay**: Decorrelated sample buffer
4. ** Epsilon-Greedy**: Balanced exploration/exploitation

### Network Architecture
```
Input: (4, 84, 84) stacked grayscale frames
├── Conv2D(32, 8×8, stride=4) + ReLU
├── Conv2D(64, 4×4, stride=2) + ReLU  
├── Conv2D(64, 3×3, stride=1) + ReLU
├── Flatten
├── Linear(512) + ReLU
└── Linear(action_space_size)
```