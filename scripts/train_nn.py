# -*- coding: utf-8 -*-
"""
Train Neural Network for Behavior Cloning

Learns to imitate NMPC: (p_c, p_c_ref, kv_prev) → kv_set
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# Neural Network Architecture
# {10, 10, 1} — increased from {5,5,1} to handle ramps/bursts in v4 dataset
# Tanh activations for smooth control output
# Output bounded to [0.1, 1.0] matching kv physical constraints
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=10, output_dim=1):
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        raw = self.network(x)
        return 0.1 + 0.9 * torch.sigmoid(raw)   # bounded to [0.1, 1.0]


def load_data():
    """Load raw dataset and normalize inputs only.

    kv_set is already in [0.1, 1.0] — no normalization needed for output.
    p_c and p_c_ref are normalized to zero mean, unit variance for training stability.
    kv_prev is already in [0.1, 1.0] — no normalization needed.
    """
    data = np.load(os.path.join(os.path.dirname(__file__), '..', 'data', 'bc_dataset_v6.npz'))

    p_c     = data['p_c']
    p_c_ref = data['p_c_ref']
    kv_prev = data['kv_prev']
    kv_set  = data['kv_set']

    # Normalize pressure inputs only
    p_c_mean, p_c_std     = p_c.mean(),     p_c.std()
    p_ref_mean, p_ref_std = p_c_ref.mean(), p_c_ref.std()

    p_c_norm     = (p_c     - p_c_mean)  / p_c_std
    p_c_ref_norm = (p_c_ref - p_ref_mean) / p_ref_std

    # Inputs: normalized p_c, normalized p_c_ref, raw kv_prev
    X = np.stack([p_c_norm, p_c_ref_norm, kv_prev], axis=1)

    # Output: raw kv_set in [0.1, 1.0] — matches bounded output layer
    y = kv_set.reshape(-1, 1)

    stats = {'p_c_mean': p_c_mean, 'p_c_std': p_c_std,
             'p_ref_mean': p_ref_mean, 'p_ref_std': p_ref_std}

    return X, y, stats


def train_test_split(X, y, test_ratio=0.2, steps_per_episode=20):
    """Split data into train and test sets by episode to prevent data leakage"""
    n_samples = len(X)
    n_episodes = n_samples // steps_per_episode
    n_test_episodes = int(n_episodes * test_ratio)

    # Shuffle episodes, not individual timesteps
    episode_indices = np.random.permutation(n_episodes)

    test_episodes = episode_indices[:n_test_episodes]
    train_episodes = episode_indices[n_test_episodes:]

    # Convert episode indices to sample indices
    train_idx = np.concatenate([np.arange(ep * steps_per_episode, (ep + 1) * steps_per_episode)
                                for ep in train_episodes])
    test_idx = np.concatenate([np.arange(ep * steps_per_episode, (ep + 1) * steps_per_episode)
                               for ep in test_episodes])

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test


def weighted_mse(y_pred, y_batch, kv_prev_batch, active_weight=5.0, threshold=0.02):
    """Weighted MSE: active steps (|kv_set - kv_prev| >= threshold) get higher weight.
    active_weight=5.0 means active steps are 5x more important than trivial steps.
    """
    dkv = torch.abs(y_batch - kv_prev_batch)
    weights = torch.where(dkv >= threshold,
                          torch.full_like(dkv, active_weight),
                          torch.ones_like(dkv))
    return (weights * (y_pred - y_batch) ** 2).mean()


def train(model, train_loader, test_loader, epochs=2000, lr=1e-3, patience=100):
    """Train with weighted loss + LR scheduler + early stopping.

    - ReduceLROnPlateau: halves lr when test loss stops improving for 30 epochs
    - Early stopping: stops training and restores best weights after `patience` epochs
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=30
    )

    train_losses  = []
    test_losses   = []
    best_test_loss  = float('inf')
    best_state_dict = None
    no_improve      = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch, kv_prev_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = weighted_mse(y_pred, y_batch, kv_prev_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Testing
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch, kv_prev_batch in test_loader:
                y_pred = model(X_batch)
                loss = weighted_mse(y_pred, y_batch, kv_prev_batch)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        # LR scheduler step
        scheduler.step(test_loss)

        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss  = test_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0:
            lr_curr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Test: {test_loss:.6f} | Best: {best_test_loss:.6f} | LR: {lr_curr:.2e}')

        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
            break

    # Restore best weights
    model.load_state_dict(best_state_dict)
    print(f'Best model restored (test loss: {best_test_loss:.6f})')
    return train_losses, test_losses


def plot_training(train_losses, test_losses, arch_tag=''):
    """Plot training curves"""
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (log)')
    plt.title('Training Curve (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    fname = f'training_curve_{arch_tag}.png' if arch_tag else 'training_curve.png'
    fpath = os.path.join(plots_dir, fname)
    plt.savefig(fpath)
    plt.close()
    print(f'Saved: {fpath}')


def plot_predictions(model, X_test, y_test, arch_tag=''):
    """Plot predicted vs actual"""
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test)).numpy()

    plt.figure(figsize=(10, 5))

    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect')
    plt.xlabel('Actual kv_set (normalized)')
    plt.ylabel('Predicted kv_set (normalized)')
    plt.title('Predicted vs Actual')
    plt.legend()
    plt.grid(True)

    # Error histogram
    plt.subplot(1, 2, 2)
    errors = y_pred - y_test
    plt.hist(errors, bins=50, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title(f'Error Distribution (std={errors.std():.4f})')
    plt.grid(True)

    plt.tight_layout()
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    fname = f'prediction_results_{arch_tag}.png' if arch_tag else 'prediction_results.png'
    fpath = os.path.join(plots_dir, fname)
    plt.savefig(fpath)
    plt.close()
    print(f'Saved: {fpath}')


if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    print("Loading data...")
    X, y, stats = load_data()
    print(f"Total samples: {len(X)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, steps_per_episode=80)
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # kv_prev is the 3rd column of X — needed for weighted loss
    kv_prev_train = X_train[:, 2:3]
    kv_prev_test  = X_test[:, 2:3]

    # Create data loaders — include kv_prev for weighted loss
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train),
                                  torch.FloatTensor(kv_prev_train))
    test_dataset  = TensorDataset(torch.FloatTensor(X_test),  torch.FloatTensor(y_test),
                                  torch.FloatTensor(kv_prev_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    # Create model
    hidden_dim = 5
    model = PolicyNetwork(input_dim=3, hidden_dim=hidden_dim, output_dim=1)
    arch_tag = f"3_{hidden_dim}_{hidden_dim}_1"
    print(f"\nModel architecture: {arch_tag}\n{model}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params}")

    # Train
    print("\nTraining...")
    train_losses, test_losses = train(model, train_loader, test_loader)

    # Plot results
    plot_training(train_losses, test_losses, arch_tag)
    plot_predictions(model, X_test, y_test, arch_tag)

    # Save model and normalization stats
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'policy_model.pth')

    torch.save({
        'model_state_dict': model.state_dict(),
        'p_c_mean':   stats['p_c_mean'],
        'p_c_std':    stats['p_c_std'],
        'p_ref_mean': stats['p_ref_mean'],
        'p_ref_std':  stats['p_ref_std'],
    }, model_path)

    print(f"\nSaved: {model_path}")
    print("\nTraining complete!")
