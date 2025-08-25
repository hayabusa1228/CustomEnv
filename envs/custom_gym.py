"""
Custom Gymnasium environment wrapper for DreamerV3.
This module provides an example of how to integrate custom Gymnasium environments.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import functools
import envs.wrappers as wrappers


class SimpleGridWorld(gym.Env):
    """
    A simple custom grid world environment as an example.
    The agent needs to reach a goal position in a grid.
    """
    
    def __init__(self, grid_size=10, max_steps=100, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        
        # Observation: agent position (x, y) and goal position (x, y) as image
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255, 
                shape=(64, 64, 3), 
                dtype=np.uint8
            ),
            'is_first': spaces.Box(0, 1, (1,), dtype=np.uint8),
            'is_last': spaces.Box(0, 1, (1,), dtype=np.uint8),
            'is_terminal': spaces.Box(0, 1, (1,), dtype=np.uint8),
        })
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Random initial positions
        self.agent_pos = np.random.randint(0, self.grid_size, size=2)
        self.goal_pos = np.random.randint(0, self.grid_size, size=2)
        
        # Make sure agent and goal are not at the same position
        while np.array_equal(self.agent_pos, self.goal_pos):
            self.goal_pos = np.random.randint(0, self.grid_size, size=2)
        
        self.step_count = 0
        
        obs = self._get_obs()
        obs['is_first'] = np.array([1], dtype=np.uint8)
        obs['is_last'] = np.array([0], dtype=np.uint8)
        obs['is_terminal'] = np.array([0], dtype=np.uint8)
        
        return obs, {}
    
    def step(self, action):
        # Move agent based on action
        if action == 0:  # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # down
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        
        self.step_count += 1
        
        # Calculate reward (negative distance to goal + bonus for reaching goal)
        distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        reached_goal = np.array_equal(self.agent_pos, self.goal_pos)
        
        if reached_goal:
            reward = 10.0
        else:
            reward = -0.1  # Small negative reward for each step
        
        # Check if episode is done
        done = reached_goal or self.step_count >= self.max_steps
        truncated = self.step_count >= self.max_steps and not reached_goal
        
        obs = self._get_obs()
        obs['is_first'] = np.array([0], dtype=np.uint8)
        obs['is_last'] = np.array([1 if done else 0], dtype=np.uint8)
        obs['is_terminal'] = np.array([1 if done and not truncated else 0], dtype=np.uint8)
        
        return obs, reward, done, truncated, {}
    
    def _get_obs(self):
        # Create a simple RGB image representation
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Fill background
        image[:, :] = [50, 50, 50]  # Gray background
        
        # Calculate positions in 64x64 image
        cell_size = 64 // self.grid_size
        
        # Draw goal (green)
        goal_y = self.goal_pos[1] * cell_size
        goal_x = self.goal_pos[0] * cell_size
        image[goal_y:goal_y+cell_size, goal_x:goal_x+cell_size] = [0, 255, 0]
        
        # Draw agent (blue)
        agent_y = self.agent_pos[1] * cell_size
        agent_x = self.agent_pos[0] * cell_size
        image[agent_y:agent_y+cell_size, agent_x:agent_x+cell_size] = [0, 0, 255]
        
        return {
            'image': image,
            'is_first': np.array([0], dtype=np.uint8),
            'is_last': np.array([0], dtype=np.uint8),
            'is_terminal': np.array([0], dtype=np.uint8),
        }
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_obs()['image']
        return None


class MountainCarCustom(gym.Env):
    """
    Custom wrapper around MountainCar that outputs image observations.
    This is an example of wrapping an existing Gym environment.
    """
    
    def __init__(self, continuous=False, render_mode=None):
        super().__init__()
        if continuous:
            self.env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')
        else:
            self.env = gym.make('MountainCar-v0', render_mode='rgb_array')
        
        self.action_space = self.env.action_space
        self.continuous = continuous
        
        # Convert to image observation
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(64, 64, 3),
                dtype=np.uint8
            ),
            'is_first': spaces.Box(0, 1, (1,), dtype=np.uint8),
            'is_last': spaces.Box(0, 1, (1,), dtype=np.uint8),
            'is_terminal': spaces.Box(0, 1, (1,), dtype=np.uint8),
        })
    
    def reset(self, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        
        # Render and resize
        image = self.env.render()
        if image.shape[:2] != (64, 64):
            import cv2
            image = cv2.resize(image, (64, 64))
        
        obs = {
            'image': image.astype(np.uint8),
            'is_first': np.array([1], dtype=np.uint8),
            'is_last': np.array([0], dtype=np.uint8),
            'is_terminal': np.array([0], dtype=np.uint8),
        }
        
        return obs, info
    
    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        
        # Render and resize
        image = self.env.render()
        if image.shape[:2] != (64, 64):
            import cv2
            image = cv2.resize(image, (64, 64))
        
        done = terminated or truncated
        
        obs = {
            'image': image.astype(np.uint8),
            'is_first': np.array([0], dtype=np.uint8),
            'is_last': np.array([1 if done else 0], dtype=np.uint8),
            'is_terminal': np.array([1 if terminated else 0], dtype=np.uint8),
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()


def make_custom_env(task, **kwargs):
    """
    Factory function to create custom Gymnasium environments.
    
    Args:
        task: Name of the environment to create
        **kwargs: Additional arguments for the environment
    
    Returns:
        Wrapped environment ready for DreamerV3
    """
    
    # Parse task name
    if task == 'custom_gridworld':
        env = SimpleGridWorld(**kwargs)
    elif task == 'custom_mountaincar':
        env = MountainCarCustom(continuous=False, **kwargs)
    elif task == 'custom_mountaincar_continuous':
        env = MountainCarCustom(continuous=True, **kwargs)
    elif task.startswith('gym_'):
        # Generic Gymnasium environment
        env_name = task[4:]  # Remove 'gym_' prefix
        env = gym.make(env_name, **kwargs)
        # Wrap to ensure proper observation format
        env = GymImageWrapper(env)
    else:
        raise ValueError(f"Unknown custom environment: {task}")
    
    # Apply standard wrappers
    env = GymWrapper(env)
    return env


class GymImageWrapper(gym.Wrapper):
    """
    Wrapper to convert any Gym environment to image observations.
    """
    
    def __init__(self, env, image_size=(64, 64)):
        super().__init__(env)
        self.image_size = image_size
        
        # Update observation space
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(*image_size, 3),
                dtype=np.uint8
            ),
            'is_first': spaces.Box(0, 1, (1,), dtype=np.uint8),
            'is_last': spaces.Box(0, 1, (1,), dtype=np.uint8),
            'is_terminal': spaces.Box(0, 1, (1,), dtype=np.uint8),
        })
        
        # Try to enable rendering
        if hasattr(env, 'render_mode'):
            env.render_mode = 'rgb_array'
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._convert_obs(obs, is_first=True), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        obs = self._convert_obs(
            obs, 
            is_first=False,
            is_last=done,
            is_terminal=terminated
        )
        
        return obs, reward, terminated, truncated, info
    
    def _convert_obs(self, obs, is_first=False, is_last=False, is_terminal=False):
        # Try to render the environment
        try:
            image = self.env.render()
            if image is None:
                # If rendering fails, create a simple visualization
                image = self._obs_to_image(obs)
        except:
            image = self._obs_to_image(obs)
        
        # Resize if necessary
        if image.shape[:2] != self.image_size:
            import cv2
            image = cv2.resize(image, self.image_size)
        
        return {
            'image': image.astype(np.uint8),
            'is_first': np.array([1 if is_first else 0], dtype=np.uint8),
            'is_last': np.array([1 if is_last else 0], dtype=np.uint8),
            'is_terminal': np.array([1 if is_terminal else 0], dtype=np.uint8),
        }
    
    def _obs_to_image(self, obs):
        """
        Convert observation to image if render is not available.
        """
        # Simple visualization for vector observations
        if isinstance(obs, np.ndarray):
            # Normalize observation to 0-255 range
            obs_norm = (obs - obs.min()) / (obs.max() - obs.min() + 1e-8)
            obs_norm = (obs_norm * 255).astype(np.uint8)
            
            # Create an image representation
            height, width = self.image_size
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw bars for each observation dimension
            bar_width = width // len(obs)
            for i, val in enumerate(obs_norm):
                bar_height = int(val * height / 255)
                x_start = i * bar_width
                x_end = min((i + 1) * bar_width, width)
                image[height-bar_height:, x_start:x_end] = [100, 150, 200]
            
            return image
        else:
            # Return a blank image if we can't visualize
            return np.zeros((*self.image_size, 3), dtype=np.uint8)


class GymWrapper:
    """
    Wrapper to make Gymnasium environments compatible with DreamerV3.
    """
    
    def __init__(self, env):
        self._env = env
        self._obs_dict = hasattr(env.observation_space, 'spaces')
        
    @property
    def observation_space(self):
        if self._obs_dict:
            return self._env.observation_space
        else:
            # Convert to dict space if needed
            return spaces.Dict({
                'image': self._env.observation_space,
                'is_first': spaces.Box(0, 1, (1,), dtype=np.uint8),
                'is_last': spaces.Box(0, 1, (1,), dtype=np.uint8),
                'is_terminal': spaces.Box(0, 1, (1,), dtype=np.uint8),
            })
    
    @property
    def action_space(self):
        return self._env.action_space
    
    def reset(self):
        obs, _ = self._env.reset()
        if not self._obs_dict:
            obs = {
                'image': obs,
                'is_first': np.array([1], dtype=np.uint8),
                'is_last': np.array([0], dtype=np.uint8),
                'is_terminal': np.array([0], dtype=np.uint8),
            }
        return obs
    
    def step(self, action):
        obs, reward, terminated, truncated, _ = self._env.step(action)
        done = terminated or truncated
        
        if not self._obs_dict:
            obs = {
                'image': obs,
                'is_first': np.array([0], dtype=np.uint8),
                'is_last': np.array([1 if done else 0], dtype=np.uint8),
                'is_terminal': np.array([1 if terminated else 0], dtype=np.uint8),
            }
        
        return obs, reward, done
    
    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)
    
    def close(self):
        self._env.close()


# Integration function for DreamerV3
def make_env(config, mode, id):
    """
    Create a custom Gymnasium environment for DreamerV3.
    
    Args:
        config: Configuration object
        mode: 'train' or 'eval'
        id: Environment ID for parallel environments
    
    Returns:
        Wrapped environment
    """
    # Get task-specific parameters
    task = config.task
    
    # Create the base environment
    env = make_custom_env(task)
    
    # Apply DreamerV3 wrappers
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key='action')
    env = wrappers.UUID(env)
    
    if config.reward_EMA:
        env = wrappers.RewardObs(env)
    
    return env