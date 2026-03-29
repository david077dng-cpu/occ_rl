"""
Simple visualization of the trained 3M PPO policy.
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, '/Users/bobinding/Documents/robot/xrollout')

from occupancy_grid_rl.envs import OccupancyGridEnv
from occupancy_grid_rl.training.train_ppo_custom import ActorCriticPolicy

def load_policy(checkpoint_path, device='cpu'):
    """Load trained policy from checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    policy = ActorCriticPolicy(action_dim=3, hidden_size=256).to(device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    return policy

def run_episode(env, policy, seed=None, deterministic=True, device='cpu'):
    """Run a single episode and record trajectory."""
    obs, info = env.reset(seed=seed)

    positions = []
    rewards = []
    done = False
    steps = 0
    total_reward = 0

    while not done and steps < env.max_episode_steps:
        # Record current position (from robot_state [x, y, theta, vx, vy, omega])
        robot_pos = env.robot_state[:2].copy() if env.robot_state is not None else np.array([5.0, 5.0])
        positions.append(robot_pos)

        # Get action from policy
        obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0).to(device) for k, v in obs.items()}
        with torch.no_grad():
            if deterministic:
                mean, _, _ = policy(obs_tensor)
                action = mean.cpu().numpy()[0]
            else:
                action, _, _ = policy.get_action(obs_tensor, deterministic=False)
                action = action.cpu().numpy()[0]

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    return {
        'positions': np.array(positions),
        'rewards': np.array(rewards),
        'total_reward': total_reward,
        'steps': steps,
        'goal_reached': info.get('goal_reached', False),
        'goal_position': env.goal_position.copy(),
        'occupancy_grid': env.occupancy_grid.copy()
    }

def main():
    checkpoint_path = 'ppo_3m_output/checkpoint_3004416.pt'
    device = 'cpu'
    n_episodes = 5

    print('=' * 80)
    print('VISUALIZING 3M PPO POLICY - SIMPLE VERSION')
    print('=' * 80)
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Episodes to run: {n_episodes}')
    print()

    # Load policy
    print('Loading policy...')
    policy = load_policy(checkpoint_path, device)
    print('Policy loaded successfully!')
    print()

    # Create environment
    env = OccupancyGridEnv(
        world_width=10.0,
        world_height=10.0,
        num_static_obstacles=5,
        num_dynamic_obstacles=2,
        max_episode_steps=500,
        random_seed=42,
    )

    # Run episodes
    print('=' * 80)
    print(f'RUNNING {n_episodes} EPISODES')
    print('=' * 80)
    print()

    results = []
    for i in range(n_episodes):
        print(f'Episode {i+1}/{n_episodes}...')
        result = run_episode(env, policy, seed=i+1000, deterministic=True, device=device)
        results.append(result)

        status = "✓ SUCCESS" if result['goal_reached'] else "✗ Failure"
        print(f'  Status: {status}')
        print(f'  Reward: {result["total_reward"]:.2f}')
        print(f'  Steps: {result["steps"]}')
        print(f'  Start: ({result["positions"][0][0]:.2f}, {result["positions"][0][1]:.2f})')
        print(f'  End: ({result["positions"][-1][0]:.2f}, {result["positions"][-1][1]:.2f})')
        print(f'  Goal: ({result["goal_position"][0]:.2f}, {result["goal_position"][1]:.2f})')
        print()

    # Summary
    successes = sum(1 for r in results if r['goal_reached'])
    total_rewards = [r['total_reward'] for r in results]
    steps = [r['steps'] for r in results]

    print('=' * 80)
    print('EVALUATION SUMMARY')
    print('=' * 80)
    print(f'Success Rate: {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)')
    print(f'Average Total Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}')
    print(f'Average Steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}')
    print()
    print('=' * 80)
    print('VISUALIZATION COMPLETE')
    print('=' * 80)

if __name__ == '__main__':
    main()
