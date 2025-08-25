#!/usr/bin/env python3
"""
Test script for custom Gymnasium environments.
This script tests the custom environments before running full training.
"""

import sys
import pathlib
import numpy as np

# Add the project root to path
sys.path.append(str(pathlib.Path(__file__).parent))

from envs.custom_gym import make_custom_env, SimpleGridWorld, MountainCarCustom
import envs.wrappers as wrappers


def test_simple_gridworld():
    """Test the SimpleGridWorld environment."""
    print("Testing SimpleGridWorld...")
    
    env = SimpleGridWorld(grid_size=5, max_steps=50)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation keys: {obs.keys()}")
    print(f"Image shape: {obs['image'].shape}")
    print(f"is_first: {obs['is_first']}")
    
    # Test a few steps
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {step}: action={action}, reward={reward:.2f}, done={terminated or truncated}")
        
        if terminated or truncated:
            print(f"Episode ended after {step + 1} steps")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print("SimpleGridWorld test completed!\n")


def test_mountain_car():
    """Test the MountainCar custom wrapper."""
    print("Testing Custom MountainCar...")
    
    try:
        env = MountainCarCustom(continuous=False)
        
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Test reset
        obs, info = env.reset()
        print(f"Initial observation keys: {obs.keys()}")
        print(f"Image shape: {obs['image'].shape}")
        
        # Test a few steps
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {step}: action={action}, reward={reward:.2f}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("Custom MountainCar test completed!\n")
        
    except ImportError as e:
        print(f"Warning: Could not test MountainCar - {e}")
        print("This is normal if you don't have the required dependencies.\n")


def test_factory_function():
    """Test the make_custom_env factory function."""
    print("Testing factory function...")
    
    # Test creating different environments
    envs_to_test = [
        ('custom_gridworld', {}),
    ]
    
    # Add MountainCar if available
    try:
        import gymnasium as gym
        envs_to_test.extend([
            ('custom_mountaincar', {}),
            ('custom_mountaincar_continuous', {}),
        ])
    except ImportError:
        print("Warning: gymnasium not available, skipping some tests")
    
    for env_name, kwargs in envs_to_test:
        try:
            print(f"Creating {env_name}...")
            env = make_custom_env(env_name, **kwargs)
            
            obs = env.reset()
            print(f"  Reset successful, obs keys: {obs.keys()}")
            
            # Take a few random actions
            for _ in range(3):
                action = env.action_space.sample()
                obs, reward, done = env.step(action)
                if done:
                    break
            
            if hasattr(env, 'close'):
                env.close()
            
            print(f"  {env_name} test successful!")
            
        except Exception as e:
            print(f"  Error testing {env_name}: {e}")
    
    print("Factory function test completed!\n")


def test_dreamerv3_integration():
    """Test integration with DreamerV3 wrappers."""
    print("Testing DreamerV3 integration...")
    
    try:
        # Create environment
        env = make_custom_env('custom_gridworld')
        
        # Apply DreamerV3 wrappers
        env = wrappers.TimeLimit(env, 100)
        env = wrappers.SelectAction(env, key='action')
        env = wrappers.UUID(env)
        
        print(f"Wrapped environment action space: {env.action_space}")
        print(f"Wrapped environment observation space: {env.observation_space}")
        
        # Test episode
        obs = env.reset()
        print(f"Wrapped reset successful, obs keys: {obs.keys()}")
        
        total_reward = 0
        for step in range(10):
            # Create action dict as expected by DreamerV3
            action_dict = {'action': env.action_space.sample()}
            obs, reward, done = env.step(action_dict)
            total_reward += reward
            
            if done:
                print(f"Episode ended after {step + 1} steps")
                break
        
        print(f"Total reward: {total_reward:.2f}")
        print("DreamerV3 integration test completed!\n")
        
    except Exception as e:
        print(f"Error in integration test: {e}")


def main():
    """Run all tests."""
    print("Running custom environment tests...\n")
    
    test_simple_gridworld()
    test_mountain_car()
    test_factory_function()
    test_dreamerv3_integration()
    
    print("All tests completed!")
    print("\nTo train with your custom environment, run:")
    print("python dreamer.py --configs custom_gridworld --logdir ./logdir/custom_gridworld")
    print("python dreamer.py --configs custom_mountaincar --logdir ./logdir/custom_mountaincar")


if __name__ == "__main__":
    main()