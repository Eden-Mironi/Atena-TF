#!/usr/bin/env python
"""Test script to verify configuration is loaded correctly"""

import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'Configuration'))
sys.path.append(os.path.dirname(__file__))

print("=" * 60)
print("TESTING CONFIGURATION")
print("=" * 60)

# Import config
import Configuration.config as cfg

print(f"\nREWARD COEFFICIENTS:")
print(f"  - humanity_coeff: {cfg.humanity_coeff}")
print(f"  - diversity_coeff: {cfg.diversity_coeff}")
print(f"  - kl_coeff: {cfg.kl_coeff}")
print(f"  - compaction_coeff: {cfg.compaction_coeff}")
print(f"  - entropy_coef: {cfg.entropy_coef}")
print(f"  - reward_scale_factor: {cfg.reward_scale_factor}")

print(f"\nPPO PARAMETERS:")
print(f"  - adam_lr: {cfg.adam_lr}")
print(f"  - ppo_gamma: {cfg.ppo_gamma}")
print(f"  - ppo_lambda: {cfg.ppo_lambda}")
print(f"  - epochs: {cfg.epochs}")

print(f"\nPOLICY CONFIGURATION:")
print(f"  - USE_PARAMETRIC_SOFTMAX_POLICY: {cfg.USE_PARAMETRIC_SOFTMAX_POLICY}")
print(f"  - BETA_TEMPERATURE: {cfg.BETA_TEMPERATURE}")
print(f"  - n_hidden_channels: {cfg.n_hidden_channels}")

print(f"\nEXPECTED VALUES (Master-Exact):")
print(f"  - humanity_coeff: 1.0 {'' if cfg.humanity_coeff == 1.0 else 'WRONG!'}")
print(f"  - diversity_coeff: 2.0 {'' if cfg.diversity_coeff == 2.0 else 'WRONG!'}")
print(f"  - kl_coeff: 1.5 {'' if cfg.kl_coeff == 1.5 else 'WRONG!'}")
print(f"  - entropy_coef: 0.0 {'' if cfg.entropy_coef == 0.0 else 'WRONG!'}")
print(f"  - reward_scale_factor: 0.01 {'' if cfg.reward_scale_factor == 0.01 else 'WRONG!'}")

print("\n" + "=" * 60)
if (cfg.humanity_coeff == 1.0 and cfg.diversity_coeff == 2.0 and 
    cfg.kl_coeff == 1.5 and cfg.entropy_coef == 0.0):
    print("ALL COEFFICIENTS CORRECT!")
else:
    print("SOME COEFFICIENTS ARE WRONG!")
print("=" * 60)

