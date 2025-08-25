# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a PyTorch implementation of DreamerV3, a scalable reinforcement learning algorithm that uses world models to master diverse domains. The implementation supports multiple environments including Atari, DMC (DeepMind Control Suite), Crafter, Minecraft, and Memory Maze.

## Development Commands

### Training
```bash
# Train on DMC Vision (example with Walker Walk task)
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk

# Train on DMC Proprio
python3 dreamer.py --configs dmc_proprio --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk

# Train on Atari 100k
python3 dreamer.py --configs atari100k --task atari_pong --logdir ./logdir/atari_pong

# Train on Crafter
python3 dreamer.py --configs crafter --task crafter_reward --logdir ./logdir/crafter

# Train on Minecraft
python3 dreamer.py --configs minecraft --task minecraft_diamond --logdir ./logdir/minecraft

# Debug mode with minimal settings
python3 dreamer.py --configs debug --task dmc_walker_walk --logdir ./logdir/debug
```

### Monitoring
```bash
# View training metrics and videos
tensorboard --logdir ./logdir
```

### Environment Setup
```bash
# Install dependencies (requires Python 3.11)
pip install -r requirements.txt

# Set up Atari environment
bash envs/setup_scripts/atari.sh

# Set up Minecraft environment  
bash envs/setup_scripts/minecraft.sh
```

## Architecture

### Core Components

**dreamer.py**: Main entry point that orchestrates training. Contains the `Dreamer` class which manages the training loop, handles environment interaction, and coordinates between the world model and behavior policies. Launches via argparse with config system.

**models.py**: Implements the core DreamerV3 components:
- `WorldModel`: Combines encoder, dynamics (RSSM), decoder, and prediction heads
- `ImagBehavior`: Implements the actor-critic for imagination-based planning
- `RewardEMA`: Running mean/std normalization for rewards

**networks.py**: Neural network building blocks including:
- `RSSM`: Recurrent State Space Model for dynamics
- `MultiEncoder`/`MultiDecoder`: Handle mixed continuous/discrete observations
- Various distribution heads for predictions

**tools.py**: Utilities for training including:
- Episode replay buffer management
- Simulation loop for environment interaction
- Logging and checkpointing
- Optimizer utilities

### Environment Integration

**envs/**: Environment wrappers and setup scripts
- Individual wrappers for each environment type (atari.py, dmc.py, minecraft.py, etc.)
- `wrappers.py`: Common preprocessing wrappers (action repeat, observation normalization, etc.)
- Setup scripts for environment-specific dependencies

### Configuration System

Uses YAML-based hierarchical configs (configs.yaml):
- `defaults`: Base configuration
- Environment-specific configs override defaults (dmc_proprio, dmc_vision, atari100k, etc.)
- Command-line arguments override YAML configs

### Training Flow

1. Environment episodes are collected and stored in `traindir/evaldir`
2. `Dreamer` agent alternates between:
   - Collecting experience using current policy
   - Training world model on replay buffer
   - Training actor-critic via imagination in the world model
3. Checkpoints saved as `latest.pt` containing model and optimizer states
4. Tensorboard logging for metrics and video predictions

## Key Design Patterns

- Heavy use of configuration-driven development via configs.yaml
- Modular environment wrapping system
- Separation of world model (learning dynamics) from behavior (planning)
- Mixed precision training support via config.precision
- Parallel environment support via config.parallel