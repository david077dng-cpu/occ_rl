"""
Simple evaluation of the trained 3M PPO policy.
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
    """Run a single episode."""
    obs, info = env.reset(seed=seed)

    done = False
    steps = 0
    total_reward = 0

    while not done and steps < env.max_episode_steps:
        obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0).to(device) for k, v in obs.items()}
        with torch.no_grad():
            if deterministic:
                mean, _, _ = policy(obs_tensor)
                action = mean.cpu().numpy()[0]
            else:
                action, _, _ = policy.get_action(obs_tensor, deterministic=False)
                action = action.cpu().numpy()[0]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    return {
        'total_reward': total_reward,
        'steps': steps,
        'goal_reached': info.get('goal_reached', False)
    }

def main():
    checkpoint_path = 'ppo_3m_output/checkpoint_3004416.pt'
    device = 'cpu'
    n_episodes = 50

    print('=' * 80)
    print('EVALUATING 3M PPO POLICY')
    print('=' * 80)
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Episodes: {n_episodes}')
    print()

    # Load policy
    print('Loading policy...')
    policy = load_policy(checkpoint_path, device)
    print('Policy loaded!')
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
    print(f'Running {n_episodes} episodes...')
    print()

    results = []
    for i in range(n_episodes):
        result = run_episode(env, policy, seed=i+2000, deterministic=True, device=device)
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f'  Completed {i+1}/{n_episodes} episodes...')

    # Summary
    successes = sum(1 for r in results if r['goal_reached'])
    total_rewards = [r['total_reward'] for r in results]
    steps = [r['steps'] for r in results]

    print()
    print('=' * 80)
    print('EVALUATION SUMMARY')
    print('=' * 80)
    print(f'Success Rate: {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)')
    print(f'Average Total Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}')
    print(f'Average Steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}')
    print()

    # Compare with training stats
    print('=' * 80)
    print('COMPARISON WITH TRAINING')
    print('=' * 80)
    print('Training Success Rate (at 3M): 60.0% (30/50)')
    print(f'Evaluation Success Rate: {100*successes/n_episodes:.1f}% ({successes}/{n_episodes})')
    print()
    if abs(100*successes/n_episodes - 60.0) < 10:
        print('✓ Evaluation matches training performance!')
    else:
        print('⚠ Evaluation differs from training (expected variation)')
    print()

if __name__ == '__main__':
    main()
