#!/usr/bin/env python
"""Test a single PPO update to diagnose learning issues"""

import sys
import os
import numpy as np
import tensorflow as tf

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Configuration'))

print("=" * 60)
print("🔬 SINGLE PPO UPDATE DIAGNOSTIC TEST")
print("=" * 60)

# Import after path setup
import Configuration.config as cfg
from models.ppo.agent import PPOAgent

print(f"\nImports successful")
print(f"   Coefficients: humanity={cfg.humanity_coeff}, diversity={cfg.diversity_coeff}")

# Create agent
print(f"\nCreating PPO agent...")
agent = PPOAgent(
    obs_dim=100,  # Simplified
    action_dim=949,  # Full action space
    learning_rate=cfg.adam_lr,
    standardize_advantages=False,  # Master default
    use_parametric_softmax=True,
    n_hidden_channels=600,
    parametric_segments=((),(12,3,26),(12,)),  # Simplified networking schema
)
print(f"   Agent created")

# Generate fake batch data
print(f"\n📦 Generating fake batch (64 experiences)...")
batch_size = 64
fake_obs = np.random.randn(batch_size, 100).astype(np.float32)
fake_actions = np.random.randint(0, 949, size=batch_size).astype(np.int32)
fake_rewards = np.random.randn(batch_size).astype(np.float32) * 0.1  # Small rewards like training
fake_values = np.random.randn(batch_size).astype(np.float32) * 0.1
fake_log_probs = np.random.randn(batch_size).astype(np.float32) * 0.1
fake_dones = np.zeros(batch_size, dtype=bool)

print(f"   Obs shape: {fake_obs.shape}")
print(f"   Actions shape: {fake_actions.shape}")
print(f"   Rewards: mean={np.mean(fake_rewards):.4f}, std={np.std(fake_rewards):.4f}")

# Compute GAE advantages
print(f"\nComputing GAE advantages...")
advantages = np.zeros_like(fake_rewards)
returns = np.zeros_like(fake_rewards)
last_gae = 0

for t in reversed(range(batch_size)):
    if t == batch_size - 1:
        next_value = 0
        next_non_terminal = 1.0
    else:
        next_value = fake_values[t + 1]
        next_non_terminal = 1.0
    
    delta = fake_rewards[t] + cfg.ppo_gamma * next_value * next_non_terminal - fake_values[t]
    advantages[t] = last_gae = delta + cfg.ppo_gamma * cfg.ppo_lambda * next_non_terminal * last_gae
    returns[t] = advantages[t] + fake_values[t]

print(f"   Advantages: mean={np.mean(advantages):.4f}, std={np.std(advantages):.4f}")
print(f"              min={np.min(advantages):.4f}, max={np.max(advantages):.4f}")
print(f"   Returns: mean={np.mean(returns):.4f}, std={np.std(returns):.4f}")

# Get policy parameters before
print(f"\n📸 Capturing policy state BEFORE update...")
policy_params_before = [v.numpy().copy() for v in agent.policy.trainable_variables]
value_params_before = [v.numpy().copy() for v in agent.value_net.trainable_variables]

# Perform ONE training step
print(f"\nPerforming ONE training step...")
try:
    agent.train_step(
        tf.convert_to_tensor(fake_obs),
        tf.convert_to_tensor(fake_actions),
        tf.convert_to_tensor(fake_log_probs),
        tf.convert_to_tensor(advantages),
        tf.convert_to_tensor(returns)
    )
    print(f"   Training step completed")
except Exception as e:
    print(f"   Training step FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Get policy parameters after
print(f"\n📸 Capturing policy state AFTER update...")
policy_params_after = [v.numpy() for v in agent.policy.trainable_variables]
value_params_after = [v.numpy() for v in agent.value_net.trainable_variables]

# Check if parameters changed
print(f"\nCHECKING IF PARAMETERS CHANGED...")
policy_changed = False
value_changed = False

for i, (before, after) in enumerate(zip(policy_params_before, policy_params_after)):
    diff = np.abs(before - after).max()
    if diff > 1e-10:
        policy_changed = True
        print(f"   Policy param {i}: max_diff = {diff:.2e}")
    else:
        print(f"   Policy param {i}: NO CHANGE (diff = {diff:.2e})")

for i, (before, after) in enumerate(zip(value_params_before, value_params_after)):
    diff = np.abs(before - after).max()
    if diff > 1e-10:
        value_changed = True
        print(f"   Value param {i}: max_diff = {diff:.2e}")
    else:
        print(f"   Value param {i}: NO CHANGE (diff = {diff:.2e})")

print(f"\n" + "=" * 60)
if policy_changed and value_changed:
    print("SUCCESS: Both policy and value networks updated!")
    print("   The training loop is working correctly.")
elif policy_changed:
    print("PARTIAL: Policy updated but value network DID NOT")
    print("   This suggests an issue with value loss or gradients.")
elif value_changed:
    print("PARTIAL: Value updated but policy DID NOT")
    print("   This suggests an issue with policy loss or gradients.")
else:
    print("FAILURE: NO PARAMETERS CHANGED")
    print("   The model is NOT learning at all!")
    print("   Likely causes:")
    print("   - Zero gradients")
    print("   - Learning rate = 0")
    print("   - Loss computation issue")
print("=" * 60)

