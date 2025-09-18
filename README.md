# Machine Learning Projects — Portfolio

This repository collects selected machine-learning and reinforcement-learning projects, experiments, and notes. It documents motivating ideas, results and how to run or reproduce each project locally. The goal is to showcase practical ML work, code organization, and reproducible experiments.

If you'd like a condensed list of projects, see the Projects section below. For details and runnable code, open the project folder you're interested in (each top-level folder contains notebooks, scripts and models where applicable).

**Maintainer**: Sultan Yarylgassimov

## Highlights / Quick Links
- `digit recognition/` — Handwritten digit classifier + simple drawing app and trained models.
- `disaster prediction by tweets/` — NLP pipeline and experiments for disaster tweet classification.
- `flappy bird rl/` — Several RL agents for Flappy Bird (2-feature, 12-feature and RGB experiments).
- `tennis rl/` — Custom Pong environment and DQN-based agent; includes training and evaluation code.
- `david silver's rl course/` — Notes and Easy21 assignments from David Silver's RL lectures.
- `intro to game ai and reinforcement learning/` — Coursework and exercises for RL fundamentals.

## How this repository is organized
- Top‑level folders correspond to projects or related experiments.
- Each project folder usually contains:
  - one or more Jupyter notebooks (`*.ipynb`) for exploration and analysis,
  - scripts/executables (e.g. `app.py`, `run_*.py`, or packages like `tennis_gym`),
  - saved models and sample data (where feasible),
  - a short README inside the folder describing how to run the code.

## How to run projects (general notes)
- Install Python 3.8+ (3.10 recommended) and create a virtual environment.

  Example (Windows PowerShell):
  ```powershell
  python -m venv .venv; .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```

- Not all projects include a `requirements.txt`. If a project is missing dependencies, check the top of the main notebook/script for needed packages (common ones used across projects: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, `tensorflow`, `gym`, `pygame`).
- To run a notebook interactively, start Jupyter: `jupyter lab` or `jupyter notebook` and open the desired `.ipynb`.

## Project summaries

- Digit Recognition (`digit recognition/`)
  - CNN trained on 28x28 digits (high test accuracy). Includes a small app (`app.py`) that accepts drawn digits and predicts labels using the trained model. Notes about preprocessing (grayscale vs binary, resizing) and retraining are in the folder.

- Disaster Prediction by Tweets (`disaster prediction by tweets/`)
  - NLP feature engineering, embeddings and experiments with shallow/dense models. Final model uses summed embeddings with a dense classifier (≈80% reported). The notebook includes visualizations and feature study.

- Flappy Bird RL (`flappy bird rl/`)
  - Multiple agents trained on feature sets of varying complexity: 2 features, 12 features, and an (attempted) RGB pipeline. Training experiments, GIFs and saved weights are included. See the folder for scripts to reproduce runs and visualizations.
  
  ![](https://github.com/sultanyaril/kaggles/blob/main/flappy%20bird%20rl/gifs/env_simple.gif)
  ![](https://github.com/sultanyaril/kaggles/blob/main/flappy%20bird%20rl/gifs/env_12.gif)
  <img src="flappy bird rl/gifs/big_score.png" width="288" height="512">

- Tennis RL (`tennis rl/`)
  - Custom Pong environment implemented as a small gym-like package (`tennis_gym`). DQN agent implementation (PyTorch), training scripts and evaluation (`test_agent.py`). The package can be run as a module: `python -m tennis_gym ...` (see folder README for exact usage and arguments).

  ![](https://github.com/sultanyaril/kaggles/blob/main/tennis%20rl/robot_vs_god.gif)

- RL Coursework and Notes
  - `david silver's rl course/` and `intro to game ai and reinforcement learning/` store notes, images and notebooks used while studying RL. The Easy21 assignment and lecture notes are included.

  ![](https://github.com/sultanyaril/kaggles/blob/main/david%20silver's%20rl%20course/easy21_illustration.jpg)

## Reproducibility & tips
- Where heavy training is required, pre-trained weights are saved in the project folders so you can evaluate without retraining.
- If a project uses `pygame` or rendering, a headless environment may fail — try running locally with a display or use virtual framebuffers on Linux.
- For deterministic results, set random seeds where appropriate (notebooks often show seed settings at the top).

## Improvements I plan
- Clean up and add `requirements.txt` files per project.
- Add small runnable README files inside each project folder (if missing) with step-by-step run instructions.
- Convert selected notebooks into clean scripts / CLI tools for easier demoing.

## License & contact
- This repository is a personal portfolio. If you want to reuse code, please ask or cite accordingly.
- Contact: `sultanyaril at gmail dot com`
