# Deep Reinforcement Learning Space Invaders Project

This project implements a Deep Reinforcement Learning (DeepRL) agent to play the classic Atari game **Space Invaders** using Double Deep Q-Learning (DDQN).

## Features
- **Training**: Train the agent using DDQN with prioritized experience replay.
- **Hyperparameter Optimization**: Use Optuna to optimize hyperparameters.
- **Evaluation**: Watch the trained agent play the game.
- **Visualization**: Plot training progress and hyperparameter optimization results.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/DeepRL_SpaceInvaders_project.git
   cd DeepRL_SpaceInvaders_project
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### **1. Train the Agent**
Train the agent using DDQN:
```bash
python src/training/train.py
```

### **2. Optimize Hyperparameters**
Run hyperparameter optimization using Optuna:
```bash
python src/training/hyperparamoptim.py
```

### **3. Watch the Trained Agent Play**
Watch the trained agent play Space Invaders:
```bash
python scripts/watch_agent.py
```

---

## Requirements

**Python Version**: Python 3.10+ is required for this project.

The project requires the following dependencies:

- `torch`: For building and training the neural networks.
- `gymnasium`: For the Atari Space Invaders environment.
- `ale-py`: Atari Learning Environment support.
- `numpy`: For numerical computations.
- `optuna`: For hyperparameter optimization.
- `tqdm`: For progress bars during training.
- `matplotlib`: For visualizing training progress and results.

All dependencies are listed in the `requirements.txt` file.
Install them with:
```bash
pip install -r requirements.txt
```

---

## Results

- **Training Progress**: 

itt lesz a reward plot

- **Hyperparameter Optimization**: 

itt lesz az optimization history plot

- **Gameplay**: 

itt lesz a GIF

---

## Directory Structure

```
DeepRL_SpaceInvaders_project/
├── src/
│   ├── agent.py
│   ├── environment.py
│   ├── training/
│   │   ├── train.py
│   │   ├── hyperparamoptim.py
│   │   ├── plot.py
│   └── utils.py
├── scripts/
│   ├── watch_agent.py
├── logs/
│   ├── ddqn_study.db
│   ├── training_scores.npy
├── models/
│   ├── best_ddqn_agent.pth
│   ├── temp_agent.pth
├── README.md
├── requirements.txt
├── .gitignore
```