"""
Plot training curves from saved checkpoint.
"""
import os
import sys
import torch
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def plot_training_curves(checkpoint_path, save_dir='./visualizations'):
    """Plot policy loss, value loss, entropy from checkpoint."""
    os.makedirs(save_dir, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')

    policy_losses = checkpoint['policy_losses']
    value_losses = checkpoint['value_losses']
    entropies = checkpoint['entropies']
    total_timesteps = checkpoint['total_timesteps']

    updates = list(range(1, len(policy_losses) + 1))

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Policy loss
    axes[0].plot(updates, policy_losses, 'b-', linewidth=1.5, alpha=0.7)
    axes[0].set_xlabel('Update')
    axes[0].set_ylabel('Policy Loss')
    axes[0].set_title('Policy Loss over Training')
    axes[0].grid(True, alpha=0.3)

    # Value loss
    axes[1].plot(updates, value_losses, 'r-', linewidth=1.5, alpha=0.7)
    axes[1].set_xlabel('Update')
    axes[1].set_ylabel('Value Loss')
    axes[1].set_title('Value Loss over Training')
    axes[1].grid(True, alpha=0.3)

    # Entropy
    axes[2].plot(updates, entropies, 'g-', linewidth=1.5, alpha=0.7)
    axes[2].set_xlabel('Update')
    axes[2].set_ylabel('Entropy')
    axes[2].set_title('Policy Entropy over Training')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves: {save_path}")
    plt.close()

    return save_path

if __name__ == '__main__':
    checkpoint_path = './ppo_training_output/final_model.pt'
    plot_training_curves(checkpoint_path)
