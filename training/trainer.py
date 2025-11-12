# -*- coding: utf-8 -*-
"""
Enhanced PPO Trainer matching ATENA-master functionality
Includes missing reward components and detailed logging
"""

import tensorflow as tf
import numpy as np
import os
import json
import pickle
from collections import deque, defaultdict
from datetime import datetime
import sys
# Fix import paths to work from any directory
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.insert(0, project_root)

# Now import with proper module path
import Configuration.config as cfg
# Fix other imports to use proper paths
try:
    from humanity_reward import eval_agent_humanity, load_human_behavior_data
except ImportError:
    # Fallback: add project root to find the module
    sys.path.append(project_root)
    from humanity_reward import eval_agent_humanity, load_human_behavior_data

# Import evaluator with fallback paths
try:
    from evaluation.evaluator import TrainingEvaluator
except ImportError:
    try:
        from evaluator import TrainingEvaluator
    except ImportError:
        # Last fallback - direct import
        import os
        import sys
        evaluator_path = os.path.join(os.path.dirname(__file__), '..', 'evaluation')
        sys.path.insert(0, evaluator_path)
        from evaluator import TrainingEvaluator
# Fix remaining imports
try:
    from master_action_processing import MasterActionProcessor
except ImportError:
    sys.path.append(project_root)
    from master_action_processing import MasterActionProcessor

try:
    from human_rules_tracker import HumanRulesTracker
except ImportError:
    sys.path.append(project_root) 
    from human_rules_tracker import HumanRulesTracker


class RewardTracker:
    """Track detailed reward components like original ATENA-master"""
    
    def __init__(self):
        self.recent_rewards = defaultdict(lambda: deque(maxlen=cfg.window_size))
        self.action_types_counter = defaultdict(int)  # Support any action type
        self.human_rules_stats = {}
        self.episode_reward_details = []
    
    def update(self, reward_info, action_type, step_idx):
        """Update reward tracking with detailed breakdown"""
        
        # Track individual reward components
        for reward_type, value in reward_info.items():
            if hasattr(reward_info, 'items'):
                # Handle StepReward object
                items = reward_info.items() if hasattr(reward_info, 'items') else []
                for reward_type, value in items:
                    if (value != 0 
                        or reward_type in {"back", "same_display_seen_already", "empty_display", "empty_groupings",
                                         "humanity", "snorkel_humanity"}
                        or (action_type == "group" and reward_type in {"compaction_gain", "diversity"})
                        or (action_type == "filter" and reward_type in {"kl_distance", "diversity"})):
                        self.recent_rewards[reward_type].append(value)
            
        # Track action types
        self.action_types_counter[action_type] += 1
        
        # Store episode details
        self.episode_reward_details.append({
            'step': step_idx,
            'action_type': action_type,
            'reward_breakdown': dict(reward_info.items()) if hasattr(reward_info, 'items') else {}
        })
    
    def get_reward_summary(self):
        """Get summary of reward components"""
        summary = {}
        for reward_type, values in self.recent_rewards.items():
            if len(values) > 0:
                # Filter out non-numeric values (like dictionaries)
                numeric_values = []
                for val in values:
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        numeric_values.append(val)
                
                if numeric_values:
                    summary[f'avg_{reward_type}'] = np.mean(numeric_values)
                    summary[f'std_{reward_type}'] = np.std(numeric_values)
        return summary


class EnhancedPPOTrainer:
    """Enhanced PPO Trainer with full ATENA-master compatibility"""
    
    def __init__(self, agent, env, batch_size=64, gamma=0.995, lambda_=0.97, 
                 update_interval=2048, epochs=10, outdir='results', standardize_advantages=True,
                 use_humans_reward=False, humans_reward_interval=64,
                 # CRITICAL MASTER EVALUATOR PARAMETERS:
                 eval_interval=100000,          # Master's eval_interval (100K steps)
                 eval_n_runs=10,               # Master's eval_n_runs  
                 eval_env=None,                # Separate evaluation environment
                 save_best_so_far_agent=True,  # Master's save_best_so_far_agent
                 max_episode_len=None):        # Master's eval max episode length
        
        # Core PPO parameters (matching original)
        self.agent = agent
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.lambda_ = lambda_
        self.update_interval = update_interval
        self.epochs = epochs
        self.outdir = outdir
        self.standardize_advantages = standardize_advantages  # Master uses this
        
        # Periodic Humanity Evaluation System (matching master)
        self.use_humans_reward = use_humans_reward
        self.humans_reward_interval = humans_reward_interval
        self.human_displays_actions_clusters = {}
        self.human_obss = []
        self.prev_human_eval_num_of_episodes = 0
        self.agent_humanity_factor = 1.0  # Current humanity factor affecting rewards
        
        # Load human behavior data
        if self.use_humans_reward:
            self.human_displays_actions_clusters = load_human_behavior_data()
            if self.human_displays_actions_clusters:
                # Extract observations from human behavior data
                self.human_obss = list(self.human_displays_actions_clusters.keys())
                print(f"HUMANITY EVALUATION ENABLED: {len(self.human_obss)} human observations loaded")
                print(f"   Evaluation interval: every {humans_reward_interval} episodes")
            else:
                print("Humanity evaluation disabled - no human behavior data available")
                self.use_humans_reward = False
        
        # Create output directory
        os.makedirs(self.outdir, exist_ok=True)
        
        # Experience buffer
        self.reset_buffers()
        
        # Statistics tracking (matching original)
        self.episode_count = 0
        self.step_count = 0
        
        # CRITICAL MASTER RETURNS TRACKING SYSTEM:
        # Based on train_agent_chainerrl.py lines 405-407
        return_window_size = 100  # Master's default return_window_size
        
        # Master's exact tracking deques (lines 405-407):
        self.recent_returns = deque(maxlen=return_window_size)                    # Total returns (with humanity)
        self.recent_returns_without_humanity = deque(maxlen=return_window_size)   # Returns without humanity components
        self.recent_rewards = defaultdict(lambda: deque(maxlen=return_window_size))  # Individual reward components
        
        print(f"Master Returns Tracking initialized:")
        print(f"   Return window size: {return_window_size}")
        print(f"   Tracking returns with/without humanity separately")
        
        # CRITICAL ENTROPY TRACKING (matching master's entropy_values deque)
        # Master line 410: entropy_values = deque(maxlen=100)
        self.entropy_values = deque(maxlen=100)  # Track action distribution entropy
        self.action_types_cntr = {"back": 0, "filter": 0, "group": 0}  # Master line 418
        self.actions_cntr = defaultdict(int)  # Master line 422 (action function counters)
        self.human_obss_cntr = {}  # Will be initialized when human data is loaded
        
        # Human observation encounter counter (master lines 387-388)  
        if self.use_humans_reward and self.human_displays_actions_clusters:
            self.human_obss_cntr = {tuple_obs: 0 for tuple_obs in self.human_displays_actions_clusters.keys()}
        
        # Reward tracking
        self.reward_tracker = RewardTracker()
        
        # CRITICAL MASTER EVALUATOR SYSTEM:
        # Create evaluator matching master's train_agent_batch_with_evaluation (lines 789-798)
        self.eval_interval = eval_interval
        self.eval_n_runs = eval_n_runs
        self.save_best_so_far_agent = save_best_so_far_agent
        
        # Create evaluation environment (separate from training env like master)
        if eval_env is None and hasattr(env, 'num_envs'):
            # Create evaluation environment matching training env but for evaluation
            from vectorized_envs import make_evaluation_batch_env
            eval_env = make_evaluation_batch_env(num_envs=1)  # Single env for evaluation
            print("Created separate evaluation environment")
        
        self.evaluator = TrainingEvaluator(
            agent=agent,
            n_episodes=eval_n_runs,
            eval_interval=eval_interval,
            outdir=outdir,
            max_episode_len=max_episode_len,
            env=eval_env,
            save_best_so_far_agent=save_best_so_far_agent,
            logger=None  # We can add logging later
        )
        
        print(f"Training Evaluator initialized:")
        print(f"   Evaluation interval: {eval_interval:,} steps") 
        print(f"   Episodes per evaluation: {eval_n_runs}")
        print(f"   Save best agent: {save_best_so_far_agent}")
        
        # CRITICAL MASTER ACTION PROCESSING SYSTEM:
        # Create action processor for explicit action processing like master (lines 522-531)
        env_prop = None
        if hasattr(env, 'env_prop'):
            env_prop = env.env_prop
        elif hasattr(env, 'envs') and len(env.envs) > 0 and hasattr(env.envs[0], 'env_prop'):
            env_prop = env.envs[0].env_prop  # For vectorized environments
        
        # Determine architecture from agent configuration
        architecture = "FFGaussian"  # Default
        if hasattr(agent, 'use_parametric_softmax') and agent.use_parametric_softmax:
            architecture = "FFParamSoftmax"
        elif hasattr(agent, 'is_discrete') and agent.is_discrete:
            architecture = "FFParamSoftmax"  # Assume ParamSoftmax for discrete
        
        if env_prop is not None:
            self.action_processor = MasterActionProcessor(env_prop, architecture=architecture)
            print(f"Master Action Processor initialized with env_prop and architecture: {architecture}")
        else:
            print("Warning: Could not find env_prop, creating basic action processor")
            # Create a mock env_prop for basic functionality
            class MockEnvProp:
                def compressed2full_range(self, action_vec, continuous_filter_term=True):
                    # Basic fallback - just return the action as-is
                    return action_vec
            self.action_processor = MasterActionProcessor(MockEnvProp(), architecture=architecture)
        
        # CRITICAL MASTER HUMAN RULES TRACKING SYSTEM:
        # Create human rules tracker matching master (lines 412-415)
        self.human_rules_tracker = HumanRulesTracker(schema_name=cfg.schema)
        print("Human Rules Tracker initialized for comprehensive rule analysis")
        
        # Logging setup
        self.setup_logging()
        
        print("Enhanced PPO Trainer initialized with ATENA-master configuration:")
        print(f"  - batch_size: {batch_size}")
        print(f"  - gamma: {gamma}")
        print(f"  - lambda: {lambda_}")
        print(f"  - update_interval: {update_interval}")
        print(f"  - epochs: {epochs}")
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        
        # Main training log
        self.training_log_path = os.path.join(self.outdir, 'training_detailed.log')
        
        # Reward analysis log (matching original)
        self.reward_log_path = os.path.join(self.outdir, 'reward_analysis.jsonl')
        
        # Episode summary log
        self.episode_log_path = os.path.join(self.outdir, 'episode_summary.jsonl')
        
        # Session log (matching original format)
        self.session_log_path = os.path.join(self.outdir, 'session_log.txt')
        
        # Initialize log files
        with open(self.training_log_path, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write("="*80 + "\n")
            
        with open(self.session_log_path, 'w') as f:
            f.write("")  # Clear session log
    
    def reset_buffers(self):
        """Reset experience buffers"""
        self.obs_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.values_buffer = []
        self.log_probs_buffer = []
        self.dones_buffer = []
    
    def evaluate_agent_humanity(self):
        """
        CRITICAL: Periodic Humanity Evaluation - exact match to master's logic
        Based on ATENA-master/train_agent_chainerrl.py lines 444-483
        
        Evaluates agent's similarity to human behavior patterns and updates humanity_factor
        Called every humans_reward_interval episodes during training
        """
        if not self.use_humans_reward or not self.human_obss:
            return
        
        print(f"\nHUMANITY EVALUATION (Episode {self.episode_count})")
        print("=" * 50)
        
        # Create a copy of human observations for batch processing
        human_obss_copy = deepcopy(self.human_obss)
        
        agent_actions = []
        pad_length = 0
        num_envs = 1  # Our implementation uses single environment
        
        # Process observations in batches (matching master's logic)
        while human_obss_copy:
            batch_eval_obss = human_obss_copy[:num_envs]
            
            # Pad batch to be a multiple of num_envs (master's line 460-464)
            pad_length = num_envs - len(batch_eval_obss)
            for i in range(pad_length):
                if self.human_displays_actions_clusters:
                    first_obs_key = list(self.human_displays_actions_clusters.keys())[0]
                    batch_eval_obss.append(np.array(first_obs_key, dtype=np.float32))
            
            # Get deterministic agent actions using batch_act_with_mean
            # This is exactly what master does on line 467
            agent_actions_for_batch_obs = self.agent.batch_act_with_mean(batch_eval_obss)
            agent_actions.extend(agent_actions_for_batch_obs)
            
            # Remove already evaluated observations
            human_obss_copy = human_obss_copy[num_envs:]
        
        # Evaluate agent actions against human behavior patterns
        # Master's line 474-476
        agent_humanity_rate, agent_humanity_info = eval_agent_humanity(
            self.human_displays_actions_clusters,
            list(self.human_obss), agent_actions, pad_length)
        
        # Calculate humanity factor exactly like master (line 478)
        # agent_humanity_factor = (1.0 + agent_humanity_rate * 1.0) ** 10 / 10 * cfg.humanity_coeff
        self.agent_humanity_factor = (1.0 + agent_humanity_rate * 1.0) ** 10 / 10 * cfg.humanity_coeff
        
        # Log results (matching master's line 481-482)
        print(f"Agent humanity rate: {agent_humanity_rate:.4f}")
        print(f"Agent humanity factor: {self.agent_humanity_factor:.4f}")
        print(f"Success/Failure breakdown:")
        
        # Log detailed breakdown
        for action_type, count in agent_humanity_info['success_count_per_action_type'].items():
            print(f"   {action_type}: {count} successes")
        for action_type, count in agent_humanity_info['failure_count_per_action_type'].items():
            print(f"   {action_type}: {count} failures")
        
        print("=" * 50)
    
    def log_training_step(self, obs, action, reward, info, step_count):
        """Log training step with original ATENA-master format"""
        
        # Extract action info
        action_desc = info.get('action', f"Action: {action}")
        reward_info = info.get('reward_info', {})
        
        # Session log (matching original format exactly)
        log_lines = []
        log_lines.append(f"{obs}\n")
        log_lines.append(f"{action_desc}\n")
        log_lines.append(f"reward:{reward}\n")
        
        # Reward info breakdown
        if hasattr(reward_info, 'items'):
            log_lines.append(f"{list(reward_info.items())}\n")
        else:
            log_lines.append(f"{{}}\n")
        log_lines.append("---------------------------------------------------\n")
        
        log_str = ''.join(log_lines)
        
        # Write to session log
        with open(self.session_log_path, 'a') as f:
            f.write(log_str)
        
        # Console output (reduced frequency to match original)
        if step_count % 100 == 0 or reward != 0:
            print(log_str, end='')
        
        # Helper function to convert numpy types to JSON-serializable types
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            else:
                return obj
        
        # Detailed reward analysis log
        reward_analysis = {
            'step': int(step_count),
            'episode': int(self.episode_count),
            'obs': convert_numpy_types(obs),
            'action': convert_numpy_types(action),
            'action_desc': str(action_desc),
            'reward': float(reward),
            'reward_breakdown': convert_numpy_types(dict(reward_info.items()) if hasattr(reward_info, 'items') else {}),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.reward_log_path, 'a') as f:
            f.write(json.dumps(reward_analysis) + '\n')
    
    def compute_advantages(self, rewards, values, next_value, done):
        """Compute GAE advantages with original hyperparameters"""
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        
        # GAE calculation (matching original: gamma=0.995, lambda=0.97)
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if done else next_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.lambda_ * last_gae_lam
        
        # ChainerRL-style advantage standardization
        if hasattr(self.agent, 'standardize_advantages') and self.agent.standardize_advantages:
            advantages = self._standardize_advantages(advantages)
        
        returns = advantages + values
        return advantages, returns
        
    def _standardize_advantages(self, advantages):
        """CRITICAL FIX: ChainerRL-style advantage standardization to fix backwards learning"""
        advantages = advantages - np.mean(advantages)
        advantages = advantages / (np.std(advantages) + 1e-8)
        return advantages
    
    def collect_experience(self, obs):
        """Collect experience with enhanced logging"""
        self.reset_buffers()
        
        step_count = 0
        episode_rewards = []
        episode_rewards_without_humanity = []
        current_episode_reward = 0
        current_episode_reward_without_humanity = 0
        
        print(f"\n{'='*80}")
        print(f"COLLECTING EXPERIENCE - UPDATE {self.step_count // self.update_interval + 1}")
        print(f"{'='*80}")
        
        while step_count < self.update_interval:
            # CRITICAL MASTER ARCHITECTURE: Use REAL vectorized environments!
            # Master uses chainerrl.envs.MultiprocessVectorEnv with ACTUAL batches
            
            # Check if we're using vectorized environment (REAL batch) or single environment
            # Master uses vectorized env even with num_envs=1!
            if hasattr(self.env, 'num_envs'):
                # REAL VECTORIZED ENVIRONMENT (matching master exactly)
                batch_actions, action_distrib = self.agent.batch_act_and_train(obs)
                
                # Master's EXACT entropy tracking (line 501)
                # Master: entropy_values.append(actions_distrib.entropy.data[0])
                entropy_val = action_distrib.entropy().numpy()[0]  # Simple and direct like master!
                self.entropy_values.append(entropy_val)
                
                # CRITICAL MASTER ACTION PROCESSING:
                # Process actions explicitly like master (train_agent_chainerrl.py lines 522-531)
                processed_actions = []
                for i, action in enumerate(batch_actions):
                    processed_action_info = self.action_processor.process_action(action)
                    processed_actions.append(processed_action_info)
                
                # Log entropy for monitoring (every 100 steps)
                if step_count % 100 == 0:
                    avg_entropy = np.mean(list(self.entropy_values)[-10:]) if len(self.entropy_values) >= 10 else entropy_val
                    print(f"   Step {step_count}: Entropy = {entropy_val:.4f}, Avg(10) = {avg_entropy:.4f}")
                    # Log action processing statistics
                    self.action_processor.log_statistics(prefix="   ")
                
                # Step vectorized environment (REAL batch step like master) - use raw actions with master-exact parameters
                # Master expects: step(action, compressed=False, filter_by_field=True, continuous_filter_term=True, filter_term=None)
                next_obs, batch_rewards, batch_dones, batch_infos = self.env.step(
                    batch_actions, 
                    compressed=True,  # Master uses compressed=True for agent actions
                    filter_by_field=True,  # Master default
                    continuous_filter_term=True,  # Master default
                    filter_term=None  # Master default
                )
                
                # Process batch results (master processes all environments in batch)
                for env_idx in range(self.env.num_envs):
                    # Extract individual environment data
                    single_obs = obs[env_idx] if len(obs.shape) > 1 else obs
                    single_action = batch_actions[env_idx]
                    single_reward = batch_rewards[env_idx]
                    single_done = batch_dones[env_idx]
                    single_info = batch_infos[env_idx]
                    single_next_obs = next_obs[env_idx] if len(next_obs.shape) > 1 else next_obs
                    
                    # Get value and log_prob for this environment
                    _, log_prob, value = self.agent.act(single_obs)
                    
                    # Store experience for this environment (ensure consistent shape)
                    obs_normalized = np.array(single_obs, dtype=np.float32).flatten()  # Always flatten to 1D
                    self.obs_buffer.append(obs_normalized)
                    self.actions_buffer.append(single_action)
                    self.rewards_buffer.append(single_reward)
                    self.values_buffer.append(value)
                    self.log_probs_buffer.append(log_prob)
                    self.dones_buffer.append(single_done)
                    
                    # Update rewards
                    current_episode_reward += single_reward
                    current_episode_reward_without_humanity += single_reward  # TODO: separate humanity
                    
                    # Update tracking
                    reward_info = single_info.get('reward_info', {})
                    action_type = self.extract_action_type(single_action, single_info)
                    self.reward_tracker.update(reward_info, action_type, self.step_count + step_count)
                    
                    # Master statistics tracking
                    self.update_master_statistics(single_obs, single_action, action_type, reward_info)
                    
                    # Log individual environment step
                    self.log_training_step(single_obs, single_action, single_reward, single_info, self.step_count + step_count)
                    
                    # Handle episode completion for this environment
                    if single_done:
                        episode_rewards.append(current_episode_reward)
                        self.recent_returns.append(current_episode_reward)
                        
                        # Episode completion logic
                        self.episode_count += 1
                        if (self.use_humans_reward and 
                            self.episode_count > self.prev_human_eval_num_of_episodes and
                            self.episode_count % self.humans_reward_interval == 0):
                            self.prev_human_eval_num_of_episodes = self.episode_count
                            self.evaluate_agent_humanity()
                        
                        self.log_episode_completion(current_episode_reward, current_episode_reward_without_humanity)
                        self.log_master_statistics()
                        
                        current_episode_reward = 0
                        current_episode_reward_without_humanity = 0
                
                # Update obs for next iteration (batch)
                obs = next_obs
                step_count += self.env.num_envs  # Multiple environments stepped simultaneously
                
            else:
                # FALLBACK: Single environment (temporary compatibility)
                print("Warning: Using single environment fallback - should use vectorized!")
                
                # Get action using batch_act_and_train for entropy tracking
                batch_obs = [obs]  # Temporary single-env adaptation
                batch_actions, action_distrib = self.agent.batch_act_and_train(batch_obs)
                action = batch_actions[0]
                
                # Master's EXACT entropy tracking (same as vectorized)
                # Master: entropy_values.append(actions_distrib.entropy.data[0])
                entropy_val = action_distrib.entropy().numpy()[0]  # Simple and direct like master!
                self.entropy_values.append(entropy_val)
                
                # CRITICAL MASTER ACTION PROCESSING (single env path):
                # Process action explicitly like master (same as vectorized path)
                processed_action_info = self.action_processor.process_action(action)
                
                # Get value and log_prob
                _, log_prob, value = self.agent.act(obs)
                
                # Step single environment (convert to batch format for vectorized env)
                if hasattr(self.env, 'num_envs') and self.env.num_envs == 1:
                    # Vectorized env with 1 env expects batch format
                    next_obs, rewards, dones, infos = self.env.step(np.array([action]), compressed=True)
                    next_obs, reward, done, info = next_obs[0], rewards[0], dones[0], infos[0]
                else:
                    # Regular single environment
                    next_obs, reward, done, info = self.env.step(action, compressed=True)
                
                # Extract reward components (if available)
                reward_info = info.get('reward_info', {})
                
                # Calculate reward without humanity component
                reward_without_humanity = reward
                if hasattr(reward_info, 'humanity') and hasattr(reward_info, 'snorkel_humanity'):
                    humanity_component = getattr(reward_info, 'humanity', 0) + getattr(reward_info, 'snorkel_humanity', 0)
                    reward_without_humanity -= humanity_component
                
                # Store experience (ensure consistent shape)
                obs_normalized = np.array(obs, dtype=np.float32).flatten()  # Always flatten to 1D
                
                self.obs_buffer.append(obs_normalized)
                self.actions_buffer.append(action)
                self.rewards_buffer.append(reward)
                self.values_buffer.append(value)
                self.log_probs_buffer.append(log_prob)
                self.dones_buffer.append(done)
                
                # Update rewards
                current_episode_reward += reward
                current_episode_reward_without_humanity += reward_without_humanity
                
                # Update reward tracking
                action_type = self.extract_action_type(action, info)
                self.reward_tracker.update(reward_info, action_type, self.step_count + step_count)
                
                # Master statistics tracking
                self.update_master_statistics(obs, action, action_type, reward_info)
                
                # Logging
                self.log_training_step(obs, action, reward, info, self.step_count + step_count)
                
                step_count += 1
                
                if done:
                    episode_rewards.append(current_episode_reward)
                    episode_rewards_without_humanity.append(current_episode_reward_without_humanity)
                    self.recent_returns.append(current_episode_reward)
                    self.recent_returns_without_humanity.append(current_episode_reward_without_humanity)
                    
                    # Episode completion logic
                    self.episode_count += 1
                    if (self.use_humans_reward and 
                        self.episode_count > self.prev_human_eval_num_of_episodes and
                        self.episode_count % self.humans_reward_interval == 0):
                        self.prev_human_eval_num_of_episodes = self.episode_count
                        self.evaluate_agent_humanity()
                    
                    self.log_episode_completion(current_episode_reward, current_episode_reward_without_humanity)
                    self.log_master_statistics()
                    
                    current_episode_reward = 0
                    current_episode_reward_without_humanity = 0
                    obs = self.env.reset()
                else:
                    obs = next_obs
        
        # Get final value
        _, _, final_value = self.agent.act(obs)
        
        self.step_count += step_count
        
        # CRITICAL MASTER EVALUATOR: Call evaluation if necessary
        # Based on master's train_agent.py lines 231-236 and evaluator.evaluate_if_necessary
        if hasattr(self, 'evaluator') and self.evaluator is not None:
            eval_results = self.evaluator.evaluate_if_necessary(
                t=self.step_count, 
                episodes=self.episode_count
            )
            if eval_results is not None:
                # Log evaluation results
                print(f"Evaluation completed at step {self.step_count:,}")
                print(f"   Mean return: {eval_results['mean_return']:.3f}")
                if eval_results.get('new_best', False):
                    print(f"   New best agent saved!")
        
        return np.array(episode_rewards), final_value
    
    def extract_action_type(self, action, info):
        """Extract action type from action vector or discrete index"""
        try:
            # Handle both continuous actions (6D arrays) and discrete actions (scalars)
            if isinstance(action, np.ndarray):
                if action.shape == ():  # Scalar numpy array (discrete action)
                    discrete_idx = int(action.item())
                elif len(action) > 0:  # Vector array (continuous action) 
                    action_type_idx = int(action[0])
                    action_types = {0: "back", 1: "filter", 2: "group"}
                    return action_types.get(action_type_idx, "unknown")
                else:
                    return "unknown"
            else:
                # Plain int/scalar (discrete action)
                discrete_idx = int(action)
                
            # For discrete actions, decode from action index to action type
            if 'discrete_idx' in locals():
                if discrete_idx == 0:
                    return "back"
                elif discrete_idx < 937:  # 1 + 936 filter actions 
                    return "filter"
                else:
                    return "group"
                    
        except (TypeError, AttributeError, ValueError):
            return "unknown"
        return "unknown"
    
    def log_episode_completion(self, reward, reward_without_humanity):
        """Log episode completion with detailed statistics"""
        self.episode_count += 1
        
        # CRITICAL MASTER RETURNS TRACKING:
        # Add episode returns to master's tracking deques (lines 405-407)
        self.recent_returns.append(float(reward))                      # Total episode return (with humanity)
        self.recent_returns_without_humanity.append(float(reward_without_humanity))  # Episode return without humanity
        
        # Track individual reward components in recent_rewards deque
        reward_summary = self.reward_tracker.get_reward_summary()
        for component, value in reward_summary.items():
            if isinstance(value, (int, float)):
                self.recent_rewards[component].append(float(value))
        
        # Extract reward components for TensorBoard-style logging
        reward_components = {}
        for component, value in reward_summary.items():
            if isinstance(value, (int, float)):
                reward_components[component] = float(value)
        
        episode_data = {
            'episode': self.episode_count,
            'total_reward': float(reward),
            'reward_without_humanity': float(reward_without_humanity),
            'step': self.step_count,
            'timestamp': datetime.now().isoformat(),
            'reward_summary': reward_summary,
            'reward_components': reward_components,  # NEW: Separate reward components for CSV export
            'action_types': dict(self.reward_tracker.action_types_counter),
            'steps': self.step_count  # NEW: Add steps for compatibility
        }
        
        # Write to episode log
        with open(self.episode_log_path, 'a') as f:
            f.write(json.dumps(episode_data) + '\n')
        
        # Console output
        print(f"\nEpisode {self.episode_count} completed:")
        print(f"  Total Reward: {reward:.3f}")
        print(f"  Reward (w/o humanity): {reward_without_humanity:.3f}")
        print(f"  Steps: {self.step_count}")

    def log_master_statistics(self):
        """
        CRITICAL: Comprehensive statistics logging matching master's output
        Based on master's logging system throughout train_agent_chainerrl.py
        """
        # Entropy statistics (policy collapse detection)
        if len(self.entropy_values) > 0:
            recent_entropy = list(self.entropy_values)[-10:]
            avg_entropy = np.mean(recent_entropy)
            min_entropy = np.min(recent_entropy)
            max_entropy = np.max(recent_entropy)
            
            print(f"   ENTROPY STATS: Avg={avg_entropy:.4f}, Min={min_entropy:.4f}, Max={max_entropy:.4f}")
            
            # Policy collapse warning (low entropy indicates collapsed policy)
            if avg_entropy < 0.1:
                print(f"   WARNING: Low entropy ({avg_entropy:.4f}) - possible policy collapse!")
        
        # CRITICAL MASTER ACTION PROCESSING STATISTICS:
        # Log action processing statistics from master's explicit processing
        if hasattr(self, 'action_processor'):
            action_stats = self.action_processor.get_statistics()
            if action_stats['total_action_types'] > 0:
                print(f"   MASTER ACTION PROCESSING:")
                print(f"      Total processed: {action_stats['total_actions']}")
                for action_type, percentage in action_stats['action_type_distribution'].items():
                    count = action_stats['action_types_cntr'][action_type]
                    print(f"      {action_type}: {count} ({percentage:.1%})")
        
        # Action type distribution (master's action_types_cntr) - fallback
        total_actions = sum(self.action_types_cntr.values())
        if total_actions > 0:
            action_dist = {k: (v/total_actions*100) for k, v in self.action_types_cntr.items()}
            print(f"   ACTION TYPES: {', '.join([f'{k}={v:.1f}%' for k, v in action_dist.items() if v > 0])}")
        
        # Human observation encounters (if humanity evaluation enabled)
        if self.use_humans_reward and self.human_obss_cntr:
            total_human_obs = sum(self.human_obss_cntr.values())
            unique_encountered = sum(1 for count in self.human_obss_cntr.values() if count > 0)
            total_human_obs_available = len(self.human_obss_cntr)
            
            if total_human_obs > 0:
                coverage = (unique_encountered / total_human_obs_available) * 100
                print(f"   👥 HUMAN OBS: {total_human_obs} encounters, {unique_encountered}/{total_human_obs_available} obs ({coverage:.1f}% coverage)")
        
        # CRITICAL MASTER RETURNS TRACKING STATISTICS:
        # Report separated returns like master (based on recent_returns and recent_returns_without_humanity)
        if len(self.recent_returns) > 0 and len(self.recent_returns_without_humanity) > 0:
            recent_total = list(self.recent_returns)[-10:]  # Last 10 episodes
            recent_without_humanity = list(self.recent_returns_without_humanity)[-10:]
            
            avg_total = np.mean(recent_total)
            avg_without_humanity = np.mean(recent_without_humanity)
            humanity_impact = avg_total - avg_without_humanity
            
            print(f"   RETURNS: Total={avg_total:.2f}, Base={avg_without_humanity:.2f}, Humanity={humanity_impact:.2f}")
            
            if len(self.recent_returns) >= 2:
                # Show trend
                recent_trend = self.recent_returns[-1] - self.recent_returns[-2]
                trend_symbol = "" if recent_trend > 0 else "" if recent_trend < 0 else "➡️"
                print(f"   {trend_symbol} TREND: {recent_trend:+.2f} from last episode")
        
        # Recent reward components summary
        if len(self.recent_rewards) > 0:
            active_rewards = {k: np.mean(list(v)[-10:]) for k, v in self.recent_rewards.items() if len(v) > 0}
            if active_rewards:
                top_rewards = sorted(active_rewards.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                print(f"   💰 TOP REWARDS: {', '.join([f'{k}={v:.3f}' for k, v in top_rewards])}")
        
        # CRITICAL MASTER HUMAN RULES STATISTICS:
        # Log human rules effectiveness like master (every 10 episodes)
        if hasattr(self, 'human_rules_tracker') and self.episode_count % 10 == 0:
            self.human_rules_tracker.log_rule_statistics(prefix="   ", top_n=5)

    def update_master_statistics(self, obs, action, action_type, reward_info):
        """
        CRITICAL: Update master's comprehensive statistics tracking
        Based on master's statistical tracking throughout train_agent_chainerrl.py
        """
        # Track action types (master line 521-530)
        if action_type in self.action_types_cntr:
            self.action_types_cntr[action_type] += 1
        
        # Track human observation encounters (master lines 491-494)
        if self.use_humans_reward and self.human_obss_cntr:
            # Convert numpy array to hashable tuple - flatten first if multi-dimensional
            if hasattr(obs, 'flatten'):
                obs_tuple = tuple(obs.flatten())
            else:
                obs_tuple = tuple(obs)
            if obs_tuple in self.human_obss_cntr:
                self.human_obss_cntr[obs_tuple] += 1
        
        # Track individual reward components (master's recent_rewards tracking)
        if hasattr(reward_info, 'items'):
            for reward_name, reward_val in reward_info.items():
                if isinstance(reward_val, (int, float)):
                    self.recent_rewards[reward_name].append(reward_val)
        
        # CRITICAL MASTER HUMAN RULES TRACKING:
        # Track human rules statistics (master lines 412-415)
        if hasattr(reward_info, 'rules_reward_info'):
            # Extract rule-based rewards from the reward info
            rules_reward_info = getattr(reward_info, 'rules_reward_info', None)
            if rules_reward_info:
                self.human_rules_tracker.track_rules_from_reward_info(rules_reward_info)
        elif hasattr(reward_info, 'rules_based_humanity') and hasattr(reward_info, 'items'):
            # Look for rule-based humanity in reward breakdown
            for reward_name, reward_val in reward_info.items():
                if 'rule' in reward_name.lower() and isinstance(reward_val, (int, float)) and reward_val != 0:
                    # Try to map reward names to rules (basic mapping)
                    # This is a fallback when detailed rule info is not available
                    try:
                        # Look for rule enum in the tracker based on reward name
                        for rule in self.human_rules_tracker.HumanRule:
                            if rule.name.lower() in reward_name.lower():
                                self.human_rules_tracker.track_rule(rule, reward_val)
                                break
                    except (AttributeError, KeyError):
                        pass
    
    def train_epoch(self):
        """Train for one epoch with detailed statistics"""
        # Convert to tensors
        obs = tf.convert_to_tensor(self.obs_buffer, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.actions_buffer, dtype=tf.int32)  # int32 for discrete actions!
        rewards = np.array(self.rewards_buffer)
        values = np.array(self.values_buffer)
        log_probs = tf.convert_to_tensor(self.log_probs_buffer, dtype=tf.float32)
        dones = np.array(self.dones_buffer)
        
        # Compute advantages with correct next_value (CRITICAL PPO FIX)
        # If episode is not done, we need the value of the next state
        if dones[-1]:  # Episode finished
            next_value = 0.0
        else:  # Episode ongoing - use last observation to estimate next value
            # Use the last observation in buffer to approximate next state value
            last_obs = tf.identity(obs[-1:])  # Get last observation (TF tensor copy)
            last_obs_norm = self.agent._normalize_obs(last_obs, update=False) 
            next_value = float(self.agent.value_net(last_obs_norm).numpy()[0])
            
        advantages, returns = self.compute_advantages(rewards, values, next_value, dones[-1])
        
        # Normalize advantages (matching master's standardize_advantages parameter)
        if self.standardize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        # Training statistics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_batches = 0
        
        # Train for multiple epochs (matching original)
        batch_size = len(obs)
        indices = np.arange(batch_size)
        
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            
            for start in range(0, batch_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_obs = tf.gather(obs, batch_indices)
                batch_actions = tf.gather(actions, batch_indices)
                batch_advantages = tf.gather(advantages, batch_indices)
                batch_returns = tf.gather(returns, batch_indices)
                batch_log_probs = tf.gather(log_probs, batch_indices)
                
                # Training step
                loss_info = self.agent.train_step(
                    batch_obs, batch_actions, batch_log_probs,
                    batch_advantages, batch_returns
                )
                
                total_policy_loss += loss_info['policy_loss']
                total_value_loss += loss_info['value_loss']
                total_entropy += loss_info.get('entropy_loss', 0)
                num_batches += 1
        
        # Return average losses
        return {
            'policy_loss': total_policy_loss / num_batches if num_batches > 0 else 0,
            'value_loss': total_value_loss / num_batches if num_batches > 0 else 0,
            'entropy': -total_entropy / num_batches if num_batches > 0 else 0
        }
    
    def train(self, max_episodes=1000):
        """Main training loop with comprehensive logging"""
        obs = self.env.reset()
        all_episode_rewards = []
        
        print(f"\n{'='*80}")
        print(f"STARTING ENHANCED ATENA TRAINING")
        print(f"Max Episodes: {max_episodes}")
        print(f"Update Interval: {self.update_interval}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Epochs per Update: {self.epochs}")
        print(f"Output Directory: {self.outdir}")
        print(f"{'='*80}")
        
        # Training loop
        update_count = 0
        while self.episode_count < max_episodes:
            update_count += 1
            
            # Collect experience
            episode_rewards, final_value = self.collect_experience(obs)
            all_episode_rewards.extend(episode_rewards)
            
            # Training
            loss_info = self.train_epoch()
            
            # Logging and statistics
            self.log_training_update(update_count, episode_rewards, loss_info)
            
            # Update obs for next collection
            obs = self.env.reset()
            
            # Save progress periodically
            if update_count % 10 == 0:
                self.save_progress(update_count)
                self.save_training_statistics(update_count)
        
        # Final logging
        self.finalize_training(all_episode_rewards)
        
        return all_episode_rewards
    
    def save_training_statistics(self, update_count):
        """
        CRITICAL: Save training statistics like master
        Based on master's statistics saving system
        """
        # Save human rules statistics (like master's human_rules_stats.pickle)
        if hasattr(self, 'human_rules_tracker'):
            human_rules_path = os.path.join(self.outdir, 'human_rules_stats.pickle')
            self.human_rules_tracker.save_statistics(human_rules_path)
        
        # Save action processor statistics
        if hasattr(self, 'action_processor'):
            action_stats = self.action_processor.get_statistics()
            action_stats_path = os.path.join(self.outdir, f'action_stats_update_{update_count}.json')
            with open(action_stats_path, 'w') as f:
                json.dump(action_stats, f, indent=2)
        
        # Save comprehensive training stats
        training_stats = {
            'update_count': update_count,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'recent_returns_avg': np.mean(list(self.recent_returns)[-10:]) if len(self.recent_returns) > 0 else 0,
            'recent_returns_without_humanity_avg': np.mean(list(self.recent_returns_without_humanity)[-10:]) if len(self.recent_returns_without_humanity) > 0 else 0,
            'entropy_avg': np.mean(list(self.entropy_values)[-10:]) if len(self.entropy_values) > 0 else 0,
            'action_types_distribution': dict(self.action_types_cntr),
            'timestamp': datetime.now().isoformat()
        }
        
        training_stats_path = os.path.join(self.outdir, f'training_stats_update_{update_count}.json')
        with open(training_stats_path, 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        print(f"Training statistics saved (update {update_count})")
    
    def log_training_update(self, update_count, episode_rewards, loss_info):
        """Log training update with detailed statistics"""
        avg_episode_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0
        avg_recent_reward = np.mean(self.recent_returns) if len(self.recent_returns) > 0 else 0
        
        # Agent statistics
        agent_stats = self.agent.get_statistics()
        
        log_message = (
            f"\n{'='*80}\n"
            f"UPDATE {update_count} - Episode {self.episode_count}\n"
            f"{'='*80}\n"
            f"Episode Rewards (this update): {avg_episode_reward:.3f}\n"
            f"Recent Average Reward: {avg_recent_reward:.3f}\n"
            f"Policy Loss: {loss_info['policy_loss']:.6f}\n"
            f"Value Loss: {loss_info['value_loss']:.6f}\n"
            f"Entropy: {loss_info['entropy']:.6f}\n"
            f"Agent Statistics: {agent_stats}\n"
            f"Action Types: {dict(self.reward_tracker.action_types_counter)}\n"
            f"Steps: {self.step_count}\n"
            f"{'='*80}\n"
        )
        
        print(log_message)
        
        # Write to training log
        with open(self.training_log_path, 'a') as f:
            f.write(log_message)
    
    def save_progress(self, update_count):
        """Save training progress and statistics"""
        progress_data = {
            'update_count': update_count,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'recent_returns': list(self.recent_returns),
            'recent_returns_without_humanity': list(self.recent_returns_without_humanity),
            'action_types_counter': dict(self.reward_tracker.action_types_counter),
            'reward_summary': self.reward_tracker.get_reward_summary(),
            'timestamp': datetime.now().isoformat()
        }
        
        progress_path = os.path.join(self.outdir, f'progress_update_{update_count}.json')
        with open(progress_path, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def finalize_training(self, all_episode_rewards):
        """Finalize training with comprehensive analysis"""
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED!")
        print(f"{'='*80}")
        print(f"Total Episodes: {self.episode_count}")
        print(f"Total Steps: {self.step_count}")
        print(f"Average Reward: {np.mean(all_episode_rewards):.3f}")
        print(f"Final Recent Average: {np.mean(self.recent_returns):.3f}")
        print(f"Action Types Distribution: {dict(self.reward_tracker.action_types_counter)}")
        print(f"{'='*80}")
        
        # Save final results
        final_results = {
            'total_episodes': self.episode_count,
            'total_steps': self.step_count,
            'all_episode_rewards': all_episode_rewards,
            'average_reward': np.mean(all_episode_rewards),
            'std_reward': np.std(all_episode_rewards),
            'final_recent_average': np.mean(self.recent_returns) if self.recent_returns else 0,
            'action_types_distribution': dict(self.reward_tracker.action_types_counter),
            'reward_summary': self.reward_tracker.get_reward_summary(),
            'training_completed_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.outdir, 'final_results.json'), 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Save the trained model
        model_path = os.path.join(self.outdir, 'trained_model')
        self.agent.save_model(model_path)
        print(f"Trained model saved to: {model_path}")
        
        # Also save as CSV for Excel analysis (matching original)
        self.save_results_for_excel_analysis(all_episode_rewards)
    
    def save_results_for_excel_analysis(self, all_episode_rewards):
        """Save results in Excel-compatible format for analysis"""
        import pandas as pd
        
        try:
            # Episode summary (matching original rewards_summary.xlsx)
            episode_data = []
            for i, reward in enumerate(all_episode_rewards):
                episode_data.append({
                    'episode': i + 1,
                    'total_reward': reward,
                    'avg_reward_per_action': reward / cfg.MAX_NUM_OF_STEPS,
                    'num_of_actions': cfg.MAX_NUM_OF_STEPS
                })
            
            episode_df = pd.DataFrame(episode_data)
            episode_df.to_csv(os.path.join(self.outdir, 'episode_rewards_summary.csv'), index=False)
            
            # Detailed reward analysis (matching original rewards_analysis.xlsx)
            if hasattr(self.reward_tracker, 'episode_reward_details'):
                detailed_df = pd.DataFrame(self.reward_tracker.episode_reward_details)
                detailed_df.to_csv(os.path.join(self.outdir, 'detailed_rewards_analysis.csv'), index=False)
            
            print(f"Results saved to CSV files in {self.outdir}")
            
        except ImportError:
            print("pandas not available, saving as JSON instead")
            # Fallback to JSON
            with open(os.path.join(self.outdir, 'episode_rewards_summary.json'), 'w') as f:
                json.dump(episode_data, f, indent=2)
