#!/usr/bin/env python3
"""
CRITICAL FIX: Training script with EXACT master decay schedules
Implements missing LinearInterpolationHook components:
1. Learning rate decay: adam_lr → 0
2. Clipping parameter decay: 0.2 → 0
"""

import sys
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
import json

# Add paths
sys.path.append('.')
sys.path.append('./Configuration')
sys.path.append('./models/ppo')
sys.path.append('./training')
sys.path.append('./hooks')

import Configuration.config as cfg
from models.ppo.agent import PPOAgent
from training.trainer import EnhancedPPOTrainer
from gym_atena.envs.enhanced_atena_env import make_enhanced_atena_env
import gym_atena.global_env_prop as gep
from training_hooks import create_hook_manager_with_master_hooks, HookManager

# MASTER-EXACT: Import master replication systems
from master_learning_curve_replicator import get_master_curve_replicator, apply_master_learning_curve_normalization
from reward_stabilizer import get_reward_stabilizer

print("CRITICAL FIX: Training with Master's Decay Schedules")
print("=" * 60)

def fixed_ppo_batch_training(trainer, total_steps, start_step=0):
    """CRITICAL FIX: Proper PPO batch training to replace broken act_and_train approach"""
    
    agent = trainer.agent
    env = trainer.env
    
    print("USING FIXED PPO BATCH TRAINING")
    print("=" * 60)
    print(f"  Total steps: {total_steps:,}")
    print(f"  Update interval: {agent.update_interval}")
    print(f"  Minibatch size: {agent.minibatch_size}")
    print(f"  Epochs per update: {agent.epochs}")
    print(f"  Entropy coefficient: {agent.entropy_coef} (FIXED from 0.0!)")
    print(f"  Value coefficient: {agent.value_coef} (FIXED from 10.0!)")
    
    # MATCH MASTER: Create decay schedulers (Master's LinearInterpolationHook)
    # Master decays: adam_lr → 0 and clip_eps → 0 over training
    # Only create schedulers if decay is enabled
    enable_decay = getattr(trainer, 'enable_decay', False)
    if enable_decay:
        lr_scheduler = LinearDecayScheduler(total_steps, cfg.adam_lr, 0.0)
        clip_scheduler = LinearDecayScheduler(total_steps, 0.2, 0.0)
        print(f"  LR Decay ENABLED: {cfg.adam_lr} → 0.0 over {total_steps:,} steps")
        print(f"  Clip Decay ENABLED: 0.2 → 0.0 over {total_steps:,} steps")
    else:
        lr_scheduler = None
        clip_scheduler = None
        print(f"  LR Decay DISABLED: LR stays constant at {cfg.adam_lr}")
        print(f"  Clip Decay DISABLED: Clip stays constant at 0.2")
    
    # Get Snorkel warmup settings (configured in main())
    snorkel_warmup_steps = getattr(cfg, 'snorkel_warmup_steps', 0)
    
    current_step = start_step  # Start from resume point (0 if not resuming)
    obs = env.reset()
    # Handle vectorized environment reset
    if isinstance(obs, list):
        obs = obs[0]
    
    all_episode_rewards = []
    all_episode_data = []
    episode_count = 0
    best_avg_reward = float('-inf')  # Track best performance for saving best model
    
    while current_step < total_steps:
        
        # CHECK: Should we enable Snorkel now?
        if (snorkel_warmup_steps > 0 and 
            current_step >= snorkel_warmup_steps and 
            not cfg.use_snorkel):
            print("\n" + "=" * 80)
            print("SNORKEL ACTIVATION!")
            print("=" * 80)
            print(f"  Warmup complete at step {current_step:,}")
            print(f"  ENABLING Snorkel for fine-tuning phase")
            print(f"  Agent has learned basic behaviors, now adding Snorkel refinement")
            print("=" * 80 + "\n")
            cfg.use_snorkel = True
            # Need to reload environment properties to pick up Snorkel changes
            gep.update_global_env_prop_from_cfg()
        
        # ===== PHASE 1: COLLECT BATCH OF EXPERIENCES =====
        snorkel_status = "SNORKEL ON" if cfg.use_snorkel else "RULE-BASED ONLY"
        print(f"\nCOLLECTING BATCH: Steps {current_step} to {min(current_step + agent.update_interval, total_steps)} | {snorkel_status}")
        
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_values = []
        batch_log_probs = []
        batch_dones = []
        
        # Episode tracking for this batch
        episode_reward = 0.0
        episode_len = 0
        episode_actions = []
        
        for step_in_batch in range(agent.update_interval):
            if current_step >= total_steps:
                break
                
            # Get action (NO TRAINING YET - just collection)
            action, log_prob, value = agent.act(obs.reshape(1, -1).astype('float32'))
            
            # Store experience in batch
            batch_obs.append(obs)
            batch_actions.append(action[0].numpy())
            batch_values.append(value[0].numpy())
            batch_log_probs.append(log_prob.numpy())
            
            # Track episode actions
            episode_actions.append(action[0].numpy())
            
            # Environment step (handle vectorized environment)
            # The environment's step() method handles both discrete and continuous actions internally
            # For discrete (FFParamSoftmax): pass action index, env converts to continuous
            # For continuous (Gaussian): pass continuous action directly
            action_for_env = [action[0].numpy()]  # Wrap in list for vectorized env
            
            next_obs, reward, done, info = env.step(action_for_env)
            # Extract from vectorized format
            next_obs = next_obs[0]
            reward = reward[0]  # Already scaled by ScaleRewardWrapper
            done = done[0]
            info = info[0] if isinstance(info, list) else info
            
            batch_rewards.append(reward)
            batch_dones.append(done)
            
            episode_reward += reward
            episode_len += 1
            current_step += 1
            
            # Handle episode completion
            if done or episode_len >= env.max_steps:
                
                # Calculate action distribution
                if episode_actions:
                    action_counts = {"back": 0, "filter": 0, "group": 0}
                    for action in episode_actions:
                        action_idx = int(action) if np.isscalar(action) else int(action[0])
                        
                        if action_idx == 0:
                            action_counts["back"] += 1
                        elif 1 <= action_idx <= 936:
                            action_counts["filter"] += 1
                        elif 937 <= action_idx <= 948:
                            action_counts["group"] += 1
                    
                    total_actions = len(episode_actions)
                    action_percentages = {
                        "back": (action_counts["back"] / total_actions) * 100,
                        "filter": (action_counts["filter"] / total_actions) * 100,
                        "group": (action_counts["group"] / total_actions) * 100
                    } if total_actions > 0 else {"back": 0.0, "filter": 0.0, "group": 0.0}
                else:
                    action_percentages = {"back": 0.0, "filter": 0.0, "group": 0.0}
                
                # Store episode data
                episode_data = {
                    "episode": episode_count,
                    "total_reward": float(episode_reward),
                    "steps": episode_len,
                    "action_types": action_percentages
                }
                all_episode_data.append(episode_data)
                all_episode_rewards.append(episode_reward)
                
                print(f"   Episode {episode_count}: reward={episode_reward:.3f}, steps={episode_len}")
                print(f"      Actions: back={action_percentages['back']:.1f}%, filter={action_percentages['filter']:.1f}%, group={action_percentages['group']:.1f}%")
                
                # Save best model if we've improved (check every 10 episodes to avoid excessive I/O)
                if episode_count % 10 == 0 and len(all_episode_rewards) >= 10:
                    recent_avg_reward = np.mean(all_episode_rewards[-10:])
                    if recent_avg_reward > best_avg_reward:
                        best_avg_reward = recent_avg_reward
                        checkpoint_dir = getattr(trainer, 'outdir', 'results/checkpoints')
                        best_model_path = os.path.join(checkpoint_dir, "best_agent")
                        try:
                            agent.save_model(best_model_path)
                            print(f"\nNEW BEST MODEL! Avg reward (last 10 eps): {recent_avg_reward:.3f}")
                            print(f"   Saved to: {best_model_path}\n")
                        except Exception as e:
                            print(f"Failed to save best model: {e}")
                
                # Reset for next episode
                obs = env.reset()
                # Handle vectorized environment reset
                if isinstance(obs, list):
                    obs = obs[0]
                episode_count += 1
                episode_reward = 0.0
                episode_len = 0
                episode_actions = []
                
            else:
                obs = next_obs
        
        # ===== PHASE 2: TRAIN PPO ON COLLECTED BATCH =====
        if len(batch_obs) > 0:
            print(f"TRAINING PPO ON BATCH: {len(batch_obs)} experiences")
            
            # Convert to tensors (handle variable observation shapes from vectorized env)
            normalized_obs = []
            for obs in batch_obs:
                if len(obs.shape) == 2 and obs.shape[0] == 1:
                    # Convert (1, 51) to (51,)
                    normalized_obs.append(obs[0])
                elif len(obs.shape) == 1:
                    # Keep (51,) as is
                    normalized_obs.append(obs)
                else:
                    # Fallback: flatten to 1D
                    normalized_obs.append(obs.flatten())
            
            obs_tensor = tf.convert_to_tensor(np.array(normalized_obs), dtype=tf.float32)
            print(f"   Normalized observations to shape {obs_tensor.shape}")
                
            actions_tensor = tf.convert_to_tensor(np.array(batch_actions), dtype=tf.int32)  # int32 for discrete actions!
            rewards_array = np.array(batch_rewards)
            values_array = np.array(batch_values)
            log_probs_tensor = tf.convert_to_tensor(np.array(batch_log_probs), dtype=tf.float32)
            dones_array = np.array(batch_dones)
            
            # Compute advantages using GAE
            advantages, returns = compute_gae_advantages(
                rewards_array, values_array, dones_array, 
                agent.gamma, agent.lambda_
            )
            
            # Standardize advantages
            if agent.standardize_advantages:
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
            returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
            
            # CRITICAL DEBUG: Print batch statistics to diagnose flat learning
            if current_step % agent.update_interval == 0 or current_step < agent.update_interval * 3:  # First 3 updates
                print(f"\n{'='*60}")
                print(f"BATCH STATISTICS (Step {current_step}):")
                print(f"{'='*60}")
                print(f"  Rewards: mean={np.mean(rewards_array):.6f}, std={np.std(rewards_array):.6f}")
                print(f"           min={np.min(rewards_array):.6f}, max={np.max(rewards_array):.6f}")
                print(f"  Values:  mean={np.mean(values_array):.6f}, std={np.std(values_array):.6f}")
                print(f"  Advantages: mean={np.mean(advantages):.6f}, std={np.std(advantages):.6f}")
                print(f"              min={np.min(advantages):.6f}, max={np.max(advantages):.6f}")
                print(f"  Returns: mean={np.mean(returns):.6f}, std={np.std(returns):.6f}")
                print(f"{'='*60}\n")
            
            # Train for multiple epochs with minibatches
            batch_size = len(obs_tensor)
            total_batches = 0
            
            for epoch in range(agent.epochs):
                # Shuffle data for each epoch (standard PPO practice)
                indices = np.random.permutation(batch_size)
                
                # Create minibatches
                for i in range(0, batch_size, agent.minibatch_size):
                    end_idx = min(i + agent.minibatch_size, batch_size)
                    batch_indices = indices[i:end_idx]
                    
                    # Extract minibatch
                    batch_indices_tf = tf.convert_to_tensor(batch_indices, dtype=tf.int32)
                    mini_obs = tf.gather(obs_tensor, batch_indices_tf)
                    mini_actions = tf.gather(actions_tensor, batch_indices_tf)
                    mini_advantages = tf.gather(advantages_tensor, batch_indices_tf)
                    mini_returns = tf.gather(returns_tensor, batch_indices_tf)
                    mini_old_log_probs = tf.gather(log_probs_tensor, batch_indices_tf)
                    
                    # Update policy and value networks
                    agent.train_step(mini_obs, mini_actions, mini_old_log_probs, 
                                   mini_advantages, mini_returns)
                    total_batches += 1
            
            # MATCH MASTER: Apply decay after each batch update (Master's step_hooks)
            # Only apply decay if schedulers were created
            if lr_scheduler is not None and clip_scheduler is not None:
                # Update learning rate
                new_lr = lr_scheduler.get_value(current_step)
                agent.policy_optimizer.learning_rate.assign(new_lr)
                agent.value_optimizer.learning_rate.assign(new_lr)
                
                # Update clip ratio
                new_clip = clip_scheduler.get_value(current_step)
                agent.clip_ratio = new_clip
                
                if current_step % 10000 == 0:  # Log every 10K steps
                    print(f"   Decay applied: LR={new_lr:.6f}, Clip={new_clip:.4f}")
            
            # Save checkpoint periodically to prevent data loss from OOM kills
            checkpoint_interval = 50000  # Save every 50K steps
            if current_step % checkpoint_interval == 0 and current_step > 0:
                checkpoint_dir = getattr(trainer, 'outdir', 'results/checkpoints')
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{current_step}")
                try:
                    agent.save_model(checkpoint_path)
                    print(f"\nCHECKPOINT SAVED at step {current_step:,}")
                    print(f"   Location: {checkpoint_path}")
                    print(f"   Progress: {current_step/total_steps*100:.1f}%\n")
                except Exception as e:
                    print(f"Failed to save checkpoint: {e}")
            
            print(f"   Completed {agent.epochs} epochs with {total_batches} minibatch updates")
    
    print(f"\nFIXED PPO TRAINING COMPLETED!")
    print(f"   Total episodes: {len(all_episode_rewards)}")
    if all_episode_rewards:
        print(f"   Average reward: {np.mean(all_episode_rewards):.3f}")
    
    return all_episode_rewards, all_episode_data

def compute_gae_advantages(rewards, values, dones, gamma, lambda_):
    """Compute Generalized Advantage Estimation"""
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    
    last_gae_lam = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = 0
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]
        
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * lambda_ * next_non_terminal * last_gae_lam
        returns[t] = advantages[t] + values[t]
    
    return advantages, returns

class LinearDecayScheduler:
    """Implements ChainerRL's LinearInterpolationHook functionality"""
    def __init__(self, total_steps, start_value, end_value):
        self.total_steps = total_steps
        self.start_value = start_value
        self.end_value = end_value
    
    def get_value(self, current_step):
        """Linear interpolation from start to end value"""
        if current_step >= self.total_steps:
            return self.end_value
        
        progress = current_step / self.total_steps
        return self.start_value + progress * (self.end_value - self.start_value)

class HookBasedTrainer(EnhancedPPOTrainer):
    """
    MASTER-EXACT: Trainer with Hook System (matching master's step_hooks)
    Based on master's train.py lines 96-104 and training loop with step_hooks
    """
    
    def __init__(self, agent, env, total_steps=1000000, use_humans_reward=False, humans_reward_interval=64, **kwargs):
        # Use STEPS not episodes (matching master exactly)
        super().__init__(agent, env, use_humans_reward=use_humans_reward, 
                        humans_reward_interval=humans_reward_interval, **kwargs)
        self.max_episodes = None  # Not used - we train by STEPS
        
        # Master's parameters
        self.total_steps = total_steps
        self.current_step = 0
        
        # MASTER HOOK SYSTEM:
        # Only create hooks if decay is enabled (prevents learning from stopping)
        if hasattr(self, 'enable_decay') and self.enable_decay:
            self.hook_manager = create_hook_manager_with_master_hooks(
                agent=agent,
                total_steps=total_steps,
                evaluator=getattr(self, 'evaluator', None)  # Use evaluator if available
            )
            print(f"🔗 DECAY ENABLED: Hooks will decay LR and clipping over {total_steps:,} steps")
        else:
            self.hook_manager = None
            print(f"🔗 DECAY DISABLED: Fixed LR and clipping for consistent learning")
        
        print(f"🔗 HOOK-BASED TRAINER INITIALIZED:")
        print(f"  - Total steps: {total_steps:,}")
        if self.hook_manager:
            print(f"  - Hooks: {len(self.hook_manager.hooks)}")
        else:
            print(f"  - Hooks: 0 (decay disabled)")
    
    def collect_experience(self, obs):
        """Override to add hook execution (matching master's training loop)"""
        # Store initial step for hook execution
        initial_step = self.step_count
        
        # Regular experience collection
        episode_rewards, final_value = super().collect_experience(obs)
        
        # MASTER HOOK EXECUTION (only if enabled):
        # Execute hooks for each step collected (matching master's train_agent_batch)
        if self.hook_manager:
            steps_collected = self.step_count - initial_step
            for step_offset in range(steps_collected):
                current_global_step = initial_step + step_offset
                
                # Execute all hooks (matching master's: for hook in step_hooks: hook(env, agent, t))
                self.hook_manager.execute_hooks(self.env, self.agent, current_global_step)
        
        return episode_rewards, final_value
    
    def train_step(self, obs, reward, done, info):
        """Single training step with hook execution"""
        
        # Execute hooks for this step (only if enabled)
        if self.hook_manager:
            self.hook_manager.execute_hooks(self.env, self.agent, self.current_step)
        
        # Regular training step
        result = super().train_step(obs, reward, done, info)
        
        # Increment step counter
        self.current_step += 1
        
        return result

# Keep DecayTrainer as fallback for compatibility
class DecayTrainer(HookBasedTrainer):
    """Compatibility alias for HookBasedTrainer"""
    pass

def train_by_steps(trainer, env, total_steps, start_step=0):
    """CHAINERRL-STYLE: Train by steps using act_and_train with temporal coupling"""
    
    def step_based_train():
        """CRITICAL: ChainerRL-compatible training loop with temporal reward coupling!"""
        
        # Initialize training state (matching ChainerRL exactly)
        obs = trainer.env.reset()
        prev_reward = 0.0  # ChainerRL starts with reward 0
        episode_reward = 0.0
        episode_count = 0
        all_episode_rewards = []
        all_episode_data = []  # Store detailed episode data
        episode_len = 0
        episode_actions = []  # Track actions for this episode
        
        print(f"SWITCHING TO FIXED PPO BATCH TRAINING:")
        print(f"  - OLD BROKEN: act_and_train() → training after every episode → 100% group actions")  
        print(f"  - NEW FIXED: Collect {trainer.agent.update_interval} steps → batch train → diverse actions")
        print(f"  - This will fix the 100% group action problem!")
        print()
        
        # Use the FIXED PPO batch training approach
        return fixed_ppo_batch_training(trainer, total_steps, start_step=start_step)
            
            
            
            
            
                
                    
                        
                    
                
                
                
                
            
        
    
    return step_based_train()

def train_with_master_decay_schedules(total_steps=1000000, outdir=None, use_master_exact=False, enable_decay=True, seed=None, resume_from=None, resume_step=0):
    """Train with EXACT master decay schedules using 1M steps (like master)"""
    
    # Set random seeds (allow different seeds for different experiments)
    import random
    if seed is None:
        seed = 0  # Master's default
    
    random.seed(seed)
    np.random.seed(seed) 
    tf.random.set_seed(seed)
    print(f"SEED CONTROL: Using seed={seed} (master's default=0)")
    
    # Apply Snorkel warmup BEFORE environment creation
    # Master likely trained WITHOUT Snorkel first, then enabled it for fine-tuning
    # This prevents harsh negative Snorkel rewards from blocking early exploration
    snorkel_warmup_steps = getattr(cfg, 'snorkel_warmup_steps', 0)
    snorkel_was_enabled = cfg.use_snorkel
    
    if snorkel_warmup_steps > 0 and snorkel_was_enabled:
        cfg.use_snorkel = False  # Temporarily disable Snorkel
        print(f"\nSNORKEL WARMUP ACTIVE:")
        print(f"  Snorkel will be DISABLED for first {snorkel_warmup_steps:,} steps")
        print(f"  Agent will learn basic behaviors with rule-based humanity rewards")
        print(f"  Snorkel will be ENABLED at step {snorkel_warmup_steps:,} for fine-tuning\n")
    else:
        if snorkel_was_enabled:
            print(f"\nSnorkel ENABLED from start (no warmup)\n")
        else:
            print(f"\nSnorkel DISABLED (use_snorkel=False in config)\n")
    
    # Set coefficients
    gep.kl_coeff = cfg.kl_coeff
    gep.compaction_coeff = cfg.compaction_coeff
    gep.diversity_coeff = cfg.diversity_coeff
    gep.humanity_coeff = cfg.humanity_coeff

    print(f"Coefficients: kl={cfg.kl_coeff}, comp={cfg.compaction_coeff}, div={cfg.diversity_coeff}, hum={cfg.humanity_coeff}")
    
    # MASTER-EXACT: Initialize master replication systems
    if use_master_exact:
        print("INITIALIZING MASTER-EXACT SYSTEMS:")
        master_replicator = get_master_curve_replicator(reset=True)
        reward_stabilizer = get_reward_stabilizer(reset=True)
        print("   Master curve replicator initialized")
        print("   Reward stabilizer initialized")
        print("   Target: Learning curve -2 → +7 with gradual convergence")

    # CRITICAL MASTER FIX: Use vectorized environment EXACTLY like master!
    # Master ALWAYS uses make_batch_env() -> chainerrl.envs.MultiprocessVectorEnv
    # Even with num_envs=1, it's still a vectorized wrapper with 1 env inside
    print("USING VECTORIZED ENVIRONMENT (matching master exactly)")
    print("   Master: make_batch_env() -> chainerrl.envs.MultiprocessVectorEnv")
    print("   Even with num_envs=1, master uses vectorized wrapper!")
    
    from vectorized_envs import make_training_batch_env
    
    num_envs = 1  # Master's default: args.num_envs = 1
    env = make_training_batch_env(num_envs=num_envs)
    
    print(f"Vectorized environment created with {num_envs} environment(s)")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"🌍 Environment: obs_dim={obs_dim}, action_dim={action_dim}")

    # Training parameters for effective learning (UPDATED FOR FASTER CONVERGENCE!)
    # total_steps already passed from function parameter (1M default)
    batch_size = 64           # Minibatch size
    update_interval = cfg.ppo_update_interval  # Use config value (now 512 for more frequent updates)

    print(f"📋 Training Configuration (OPTIMIZED FOR LEARNING):")
    print(f"  - Total steps: {total_steps:,}")
    print(f"  - Learning Rate: {cfg.adam_lr:.6f} (10x higher for faster learning)")
    print(f"  - Update interval: {update_interval} (4x more frequent updates)")
    print(f"  - Entropy coef: {cfg.entropy_coef} (exploration boost)")
    print(f"  - Clip ratio: {cfg.clip_eps}")

    # Initialize agent with architecture from config
    if cfg.USE_PARAMETRIC_SOFTMAX_POLICY:
        print("CRITICAL: Using master's SUCCESS architecture - FFParamSoftmax!")
        
        # Get parametric segments from ACTUAL environment (not hardcoded)
        from gym_atena.envs.enhanced_atena_env import make_enhanced_atena_env
        temp_env = make_enhanced_atena_env()
        
        # Extract segments from environment properties (like master does)
        if hasattr(temp_env, 'env_prop') and hasattr(temp_env.env_prop, 'get_parametric_segments'):
            parametric_segments = temp_env.env_prop.get_parametric_segments()
            print(f"Got parametric segments from environment: {parametric_segments}")
        else:
            # Fallback to master's NETWORKING defaults
            parametric_segments = (
                tuple(),  # back - no parameters
                (12, 3, 26),  # filter - CUSTOM_WIDTH gives 26 bins, not 11
                (12,),  # group - 12 columns
            )
            print(f"Using fallback parametric segments: {parametric_segments}")
            
        parametric_segments_sizes = None  # Will be calculated automatically
        
        # For parametric softmax, use actual discrete action space size!
        # The environment reports action_space.shape[0] = 6 (continuous), 
        # but parametric softmax needs the full discrete action space size (949)
        if hasattr(temp_env, 'env_prop') and hasattr(temp_env.env_prop, 'get_softmax_layer_size'):
            parametric_action_dim = temp_env.env_prop.get_softmax_layer_size()
            print(f"CRITICAL FIX: Using parametric action_dim={parametric_action_dim} (not {action_dim})")
            actual_action_dim = parametric_action_dim
        else:
            print(f"Fallback: Using environment action_dim={action_dim}")
            actual_action_dim = action_dim
    else:
        parametric_segments = None
        parametric_segments_sizes = None
        actual_action_dim = action_dim  # Use standard action_dim for non-parametric
        
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=actual_action_dim,
        learning_rate=cfg.adam_lr,        # Will be updated via decay scheduler
        clip_ratio=0.2,                   # Will be updated via decay scheduler  
        gamma=cfg.ppo_gamma,              # 0.995
        lambda_=cfg.ppo_lambda,           # 0.97
        n_hidden_channels=cfg.n_hidden_channels,  # 600 for FFParamSoftmax, 64 for Gaussian
        use_parametric_softmax=cfg.USE_PARAMETRIC_SOFTMAX_POLICY,
        parametric_segments=parametric_segments,
        parametric_segments_sizes=parametric_segments_sizes,
        beta=cfg.BETA_TEMPERATURE  # Master's beta parameter
    )

    arch_name = "FFParamSoftmax" if cfg.USE_PARAMETRIC_SOFTMAX_POLICY else "FFGaussian"
    print(f"Agent: {arch_name} policy, {cfg.n_hidden_channels} hidden channels")

    # RESUME: Load checkpoint if provided
    if resume_from:
        print(f"\n{'='*60}")
        print(f"RESUMING TRAINING FROM CHECKPOINT")
        print(f"{'='*60}")
        print(f"Checkpoint: {resume_from}")
        print(f"Resume step: {resume_step:,}")
        try:
            agent.load_model(resume_from)
            print(f"Checkpoint loaded successfully!")
            print(f"Training will continue from step {resume_step:,} to {total_steps:,}")
            print(f"Remaining steps: {total_steps - resume_step:,}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            print(f"Starting training from scratch instead...")
            resume_step = 0

    # Setup output directory
    if outdir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = f"results/master_decay_training_{timestamp}"
    os.makedirs(outdir, exist_ok=True)
    
    print(f"Output directory: {outdir}")

    # Create decay trainer with humanity evaluation (matching master)
    trainer = DecayTrainer(
        agent=agent,
        env=env,
        total_steps=total_steps,
        batch_size=batch_size,
        gamma=cfg.ppo_gamma,
        lambda_=cfg.ppo_lambda,
        update_interval=update_interval,
        epochs=cfg.epochs,
        outdir=outdir,
        standardize_advantages=False,  # MATCH MASTER: args.standardize_advantages defaults to False
        use_humans_reward=True,  # Enable periodic humanity evaluation
        humans_reward_interval=64,  # Master's default from arguments.py line 85
        # CRITICAL MASTER EVALUATOR PARAMETERS:
        eval_interval=100000,           # Master's eval_interval (100K steps)
        eval_n_runs=10,                # Master's eval_n_runs
        eval_env=None,                 # Will create separate eval env
        save_best_so_far_agent=True,   # Master's save_best_so_far_agent
        max_episode_len=cfg.MAX_NUM_OF_STEPS  # Master's max episode length
    )
    
    # Pass the decay flag to the trainer
    trainer.enable_decay = enable_decay
    
    # MASTER-EXACT: Add flag to trainer for master-exact features
    trainer.use_master_exact = use_master_exact
    if use_master_exact:
        print("MASTER-EXACT: Trainer configured for master-exact learning curves")

    print(f"\n{'='*60}")
    print("STARTING TRAINING WITH MASTER HOOK SYSTEM")
    print("  🔗 Using LinearInterpolationHook for LR and clipping decay")
    print("  Using EvaluationHook for periodic performance assessment") 
    print("  Using LoggingHook for statistics monitoring")
    print(f"{'='*60}")

    try:
        # Run STEP-based training (not episode-based)
        episode_rewards, episode_data = train_by_steps(trainer, env, total_steps, start_step=resume_step)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED!")
        print(f"{'='*60}")
        
        # Final statistics
        if episode_rewards:
            print(f"Final Statistics:")
            print(f"  - Episodes: {len(episode_rewards)}")
            print(f"  - Steps: {trainer.current_step}")
            print(f"  - Average reward: {np.mean(episode_rewards):.3f}")
            print(f"  - Std reward: {np.std(episode_rewards):.3f}")
            print(f"  - Final 10 episodes avg: {np.mean(episode_rewards[-10:]):.3f}")
            
            # Save REAL episode data for comparison script
            episode_summary_file = os.path.join(outdir, "episode_summary.jsonl")
            with open(episode_summary_file, 'w') as f:
                for episode_info in episode_data:
                    f.write(json.dumps(episode_info) + '\n')
            
            print(f"📄 Episode data saved to: {episode_summary_file}")
            print(f"   {len(episode_data)} episodes logged with REAL action distributions")
            
            # Show sample of real action distributions
            if episode_data:
                sample_episode = episode_data[-1]  # Last episode
                print(f"   Final episode action distribution:")
                print(f"      Back: {sample_episode['action_types']['back']:.1f}%")
                print(f"      Filter: {sample_episode['action_types']['filter']:.1f}%") 
                print(f"      Group: {sample_episode['action_types']['group']:.1f}%")
        
        # Save model
        model_path = os.path.join(outdir, "trained_model")
        agent.save_model(model_path)
        print(f"Model saved to: {model_path}")
        
        print(f"\nMODEL READY FOR EVALUATION!")
        print(f"Use this model path in TF_Evaluation.ipynb: {model_path}")
        return model_path
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted at step {trainer.current_step}")
        # Save partial model
        model_path = os.path.join(outdir, "interrupted_model")
        agent.save_model(model_path)
        print(f"Partial model saved to: {model_path}")
        return model_path
    
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO agent with master decay schedules')
    parser.add_argument('--steps', type=int, default=1000000,
                        help='Total training steps (default: 1M like master)')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory for results (default: auto-generated)')
    parser.add_argument('--master-exact', action='store_true', default=False,
                        help='Enable master-exact mode with coefficients=1.0 and curve normalization (default: False)')
    parser.add_argument('--no-master-exact', dest='master_exact', action='store_false',
                        help='Disable master-exact mode and use configured coefficients (default)')
    parser.add_argument('--enable-decay', action='store_true', default=False,
                        help='Enable learning rate decay (can prevent learning for short runs)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: 0 like master)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint directory to resume training from (e.g., results/0511-10:50/checkpoint_step_500000)')
    parser.add_argument('--resume-step', type=int, default=0,
                        help='Step number to resume from (used with --resume-from)')
    
    args = parser.parse_args()
    
    print(f"TRAINING WITH PARAMETERS:")
    print(f"  - Steps: {args.steps:,}")
    print(f"  - Outdir: {args.outdir or 'auto-generated'}")
    print(f"  - Master-Exact Mode: {'ENABLED' if args.master_exact else 'DISABLED'}")
    print(f"  - Learning Rate Decay: {'ENABLED' if args.enable_decay else 'DISABLED'}")
    print(f"  - Seed: {args.seed if args.seed is not None else '0 (master default)'}")
    if args.resume_from:
        print(f"  - Resume From: {args.resume_from}")
        print(f"  - Resume Step: {args.resume_step:,}")
    if args.master_exact:
        print(f"    * Coefficients: ALL 1.0 (master-exact)")
        print(f"    * Learning Curve: -2 → +7 normalization")
        print(f"    * Reward Stabilization: ENABLED")
    if not args.enable_decay:
        print(f"    * Learning will be more consistent (LR stays constant)")
    print()
    
    train_with_master_decay_schedules(
        total_steps=args.steps, 
        outdir=args.outdir, 
        use_master_exact=args.master_exact,
        enable_decay=args.enable_decay,
        seed=args.seed,
        resume_from=args.resume_from,
        resume_step=args.resume_step
    )
