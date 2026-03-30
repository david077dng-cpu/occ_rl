"""
Calculate success rate over many episodes for the trained model.
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import OccupancyGridEnv
from training.train_ppo_custom import ActorCriticPolicy

def load_policy(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    policy = ActorCriticPolicy(action_dim=3, hidden_size=256).to(device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    return policy

def run_episode(env, policy, seed=None, deterministic=True, device='cpu'):
    obs, info = env.reset(seed=seed)
    done = False
    steps = 0
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
        done = terminated or truncated
        steps += 1
    return info.get('goal_reached', False)

def main():
    checkpoint_path = './ppo_training_output/final_model.pt'
    device = 'cpu'
    n_episodes = 100

    print('=' * 80)
    print('CALCULATING SUCCESS RATE')
    print('=' * 80)
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Episodes: {n_episodes}')
    print()

    policy = load_policy(checkpoint_path, device)
    print('Policy loaded')

    env = OccupancyGridEnv(
        world_width=10.0,
        world_height=10.0,
        num_static_obstacles=5,
        num_dynamic_obstacles=2,
        max_episode_steps=500,
    )
    print('Environment created')
    print()

    successes = 0
    for i in range(n_episodes):
        success = run_episode(env, policy, seed=i+42, deterministic=True, device=device)
        if success:
            successes += 1
        if (i + 1) % 10 == 0:
            print(f'  Completed {i+1}/{n_episodes} - Current success rate: {successes/(i+1)*100:.1f}%')

    print()
    print('=' * 80)
    print('FINAL RESULTS')
    print('=' * 80)
    print(f'Total episodes: {n_episodes}')
    print(f'Successful: {successes}')
    print(f'Failed: {n_episodes - successes}')
    print(f'Success rate: {successes / n_episodes * 100:.2f}%')
    print()

if __name__ == '__main__':
    main()
