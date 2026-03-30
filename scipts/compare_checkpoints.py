"""
Compare success rate across different checkpoints.
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import OccupancyGridEnv
from training.train_ppo_custom import ActorCriticPolicy

def load_policy(checkpoint_path, device='cpu'):
    if not os.path.exists(checkpoint_path):
        return None
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

def evaluate_checkpoint(checkpoint_path, n_episodes=50, device='cpu'):
    policy = load_policy(checkpoint_path, device)
    if policy is None:
        return None

    env = OccupancyGridEnv(
        world_width=10.0,
        world_height=10.0,
        num_static_obstacles=5,
        num_dynamic_obstacles=2,
        max_episode_steps=500,
    )

    successes = 0
    for i in range(n_episodes):
        success = run_episode(env, policy, seed=i+1000, deterministic=True, device=device)
        if success:
            successes += 1

    success_rate = successes / n_episodes * 100
    return {
        'checkpoint': os.path.basename(checkpoint_path),
        'timesteps': checkpoint_path.split('_')[-1].split('.')[0] if 'checkpoint' in checkpoint_path else 'final',
        'successes': successes,
        'total': n_episodes,
        'success_rate': success_rate
    }

def main():
    output_dir = './ppo_training_output'
    device = 'cpu'
    n_episodes = 50

    checkpoints = [
        os.path.join(output_dir, 'checkpoint_100352.pt'),
        os.path.join(output_dir, 'checkpoint_200704.pt'),
        os.path.join(output_dir, 'checkpoint_301056.pt'),
        os.path.join(output_dir, 'final_model.pt'),
    ]

    print('=' * 80)
    print('COMPARING CHECKPOINT SUCCESS RATES')
    print('=' * 80)
    print(f'Evaluating each checkpoint over {n_episodes} episodes...')
    print()

    results = []
    for cp in checkpoints:
        basename = os.path.basename(cp)
        print(f'Evaluating {basename}...')
        result = evaluate_checkpoint(cp, n_episodes, device)
        if result:
            results.append(result)
            print(f'  Success rate: {result["success_rate"]:.1f}% ({result["successes"]}/{n_episodes})')
        print()

    print('=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'{"Checkpoint":<20} {"Timesteps":>10} {"Success":>8} {"Total":>6} {"Rate":>8}')
    print('-' * 60)
    for r in results:
        timesteps = r['timesteps']
        print(f"{r['checkpoint']:<20} {timesteps:>10} {r['successes']:>8} {r['total']:>6} {r['success_rate']:>7.2f}%")

    print()

if __name__ == '__main__':
    main()
