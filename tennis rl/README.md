# Tennis RL (Pong)

Purpose
- Custom Pong environment and DQN agent experiments implemented in PyTorch.

Quick run
- The environment is packaged as `tennis_gym` and can be run as a module:
  ```powershell
  python -m tennis_gym [human | robot | god] [human | robot | god]
  ```
- See `test_agent.py` for evaluation examples.

Key files
- `tennis_gym/` — environment, renderer and utilities.
- `test_agent.py` — evaluation script for trained agents.
- `train_simple_torch.py` — training script.

Notes
- Training used a DQN with double-network (model + target) and PyTorch for speed.
