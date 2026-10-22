# Reinforcement Learning from Scratch

Reinforcement learning agents implemented from first principles using only **NumPy**: no PyTorch, no TensorFlow, no RL libraries. The goal is to understand every piece of the pipeline, from the Bellman update to backpropagation, by building it by hand.

## Current status: Tabular Q-learning

The first stage is a Q-learning agent that learns to navigate a 5×5 grid world, reaching the goal while avoiding holes.

```
S . . . H        S = start
. H . . .        G = goal  (+1, episode ends)
. . . H .        H = hole  (-1, episode ends)
H . . . .        . = free  (-0.01 per step)
. . H . G
```

The small per-step penalty encourages the agent to find the shortest safe path.

### Results

Over 3,000 training episodes, the average reward rises from about **-0.71** (the agent falls into holes almost every time) to about **+0.80**, and the learned policy routes around every hole to the goal:

```
↓ ← ↓ ↓ H
↓ H → → ↓
→ → ↓ H ↓
H → → → ↓
↓ ↑ H → G
```

### How it works

- **Environment** (`GridWorld`): exposes a Gymnasium-style interface where `reset()` returns the initial state and `step(action)` returns `(next_state, reward, done)`.
- **Exploration**: epsilon-greedy action selection, with epsilon decaying from 1.0 to 0.05 so the agent explores early and exploits later.
- **Learning rule**: the Q-learning update

  `Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]`

  with no bootstrapping from terminal states.

| Hyperparameter | Value |
|---|---|
| Learning rate α | 0.1 |
| Discount γ | 0.99 |
| Epsilon decay | 0.998 per episode |
| Episodes | 3,000 |


## Roadmap

- [x] Tabular Q-learning on a deterministic grid world
- [ ] Stochastic ("slippery") environment
- [ ] Neural network framework in NumPy (dense layers, backprop, Adam, gradient checking)
- [ ] Deep Q-Network (DQN) on CartPole with experience replay and target network
- [ ] Ablations: DQN with and without replay / target network, across multiple seeds
- [ ] Double DQN and REINFORCE (policy gradient)
