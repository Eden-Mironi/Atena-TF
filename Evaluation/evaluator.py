"""
CRITICAL_FIXED: Master-Exact Evaluator Class
Based on chainerrl.experiments.evaluator.Evaluator used in ATENA-master

Provides periodic performance evaluation during training with:
- Performance evaluation every eval_interval steps
- Best agent saving (save_best_so_far_agent)
- Separate evaluation environment
- Episode return statistics
- Model checkpointing
"""

import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Optional, Any, Dict, List
import tensorflow as tf

# Import our ATENA evaluation functions
import sys
sys.path.append('../')
from vectorized_envs import make_evaluation_batch_env


class TrainingEvaluator:
    """
    MASTER-EXACT: TensorFlow equivalent of chainerrl.experiments.evaluator.Evaluator
    
    Based on ATENA-master usage:
    - train_agent_chainerrl.py lines 789-798
    - train.py lines 84-104
    """
    
    def __init__(self, 
                 agent,
                 n_steps: Optional[int] = None,
                 n_episodes: int = 10,          # Master's eval_n_runs
                 eval_interval: int = 100000,   # Master's eval_interval (100K steps)
                 outdir: str = 'results',
                 max_episode_len: Optional[int] = None,
                 env=None,                      # Evaluation environment
                 step_offset: int = 0,
                 save_best_so_far_agent: bool = True,
                 logger=None):
        """
        Initialize evaluator matching master's parameters exactly
        
        Args:
            agent: PPOAgent to evaluate
            n_steps: Number of steps to run (None for episode-based evaluation)
            n_episodes: Number of episodes per evaluation (master's eval_n_runs)
            eval_interval: Evaluation interval in steps (master's 100000)
            outdir: Output directory for saving models and logs
            max_episode_len: Maximum episode length for evaluation
            env: Evaluation environment (separate from training env)
            step_offset: Step offset for evaluation timing
            save_best_so_far_agent: Whether to save best performing agent
            logger: Logger instance
        """
        self.agent = agent
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.eval_interval = eval_interval
        self.outdir = outdir
        self.max_episode_len = max_episode_len
        self.env = env
        self.step_offset = step_offset
        self.save_best_so_far_agent = save_best_so_far_agent
        self.logger = logger
        
        # Performance tracking
        self.max_score = float('-inf')
        self.evaluation_count = 0
        self.last_eval_step = -1
        
        # Evaluation history
        self.eval_history = []
        
        # Create directories
        os.makedirs(self.outdir, exist_ok=True)
        if self.save_best_so_far_agent:
            self.best_agent_path = os.path.join(self.outdir, 'best_agent')
            os.makedirs(self.best_agent_path, exist_ok=True)
        
        # Setup evaluation logging
        self.eval_log_path = os.path.join(self.outdir, 'evaluation_log.jsonl')
        
        print(f"EVALUATOR INITIALIZED:")
        print(f"   Evaluation interval: {eval_interval:,} steps")
        print(f"   Episodes per evaluation: {n_episodes}")
        print(f"   Max episode length: {max_episode_len}")
        print(f"   Save best agent: {save_best_so_far_agent}")
        print(f"   Output directory: {outdir}")
    
    def evaluate_if_necessary(self, t: int, episodes: Optional[int] = None) -> Optional[Dict]:
        """
        CRITICAL_FIXED: Master's exact evaluation trigger logic
        Based on chainerrl.experiments.evaluator.Evaluator.evaluate_if_necessary
        
        Args:
            t: Current training step
            episodes: Current episode count (optional)
            
        Returns:
            Evaluation results dictionary if evaluation was performed, None otherwise
        """
        # Adjust for step offset
        adjusted_step = t - self.step_offset
        
        # Check if it's time to evaluate
        if (adjusted_step > 0 and 
            adjusted_step % self.eval_interval == 0 and 
            adjusted_step != self.last_eval_step):
            
            self.last_eval_step = adjusted_step
            return self.evaluate(t, episodes)
        
        return None
    
    def evaluate(self, t: int, episodes: Optional[int] = None) -> Dict:
        """
        CRITICAL_FIXED: Perform evaluation matching master's methodology
        
        Args:
            t: Current training step
            episodes: Current episode count
            
        Returns:
            Dictionary with evaluation results
        """
        self.evaluation_count += 1
        
        print(f"\n{'='*60}")
        print(f"EVALUATION #{self.evaluation_count} (Step {t:,})")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Run evaluation episodes
        episode_returns = []
        episode_lengths = []
        
        for episode_idx in range(self.n_episodes):
            episode_return, episode_length = self._run_evaluation_episode(episode_idx)
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            
            if episode_idx % 5 == 0 or episode_idx == self.n_episodes - 1:
                print(f"   Episode {episode_idx + 1}/{self.n_episodes}: R={episode_return:.2f}, L={episode_length}")
        
        # Calculate statistics
        mean_return = np.mean(episode_returns)
        median_return = np.median(episode_returns)
        std_return = np.std(episode_returns)
        mean_length = np.mean(episode_lengths)
        
        eval_time = time.time() - start_time
        
        # Create evaluation results
        eval_results = {
            'step': t,
            'episodes': episodes,
            'evaluation_count': self.evaluation_count,
            'n_episodes': self.n_episodes,
            'mean_return': float(mean_return),
            'median_return': float(median_return),
            'std_return': float(std_return),
            'mean_length': float(mean_length),
            'episode_returns': [float(r) for r in episode_returns],
            'episode_lengths': [int(l) for l in episode_lengths],
            'eval_time': eval_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log results
        print(f"EVALUATION RESULTS:")
        print(f"   Mean return: {mean_return:.3f} ± {std_return:.3f}")
        print(f"   Median return: {median_return:.3f}")
        print(f"   Mean length: {mean_length:.1f}")
        print(f"   Evaluation time: {eval_time:.1f}s")
        
        # Save best agent if performance improved
        if self.save_best_so_far_agent and mean_return > self.max_score:
            self.max_score = mean_return
            print(f"NEW BEST SCORE: {mean_return:.3f} (previous: {self.max_score:.3f})")
            self._save_best_agent(t, mean_return)
            eval_results['new_best'] = True
        else:
            eval_results['new_best'] = False
        
        # Store evaluation history
        self.eval_history.append(eval_results)
        
        # Write to log file
        self._write_eval_log(eval_results)
        
        print(f"{'='*60}\n")
        
        return eval_results
    
    def _run_evaluation_episode(self, episode_idx: int) -> tuple:
        """
        Run a single evaluation episode
        
        Returns:
            (episode_return, episode_length)
        """
        if self.env is None:
            raise ValueError("Evaluation environment not provided")
        
        obs = self.env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False
        
        while not done and (self.max_episode_len is None or episode_length < self.max_episode_len):
            # Get action using most probable (deterministic) policy
            if hasattr(self.agent, 'act_most_probable'):
                action, _, _ = self.agent.act_most_probable(obs)
            else:
                action, _, _ = self.agent.act(obs)
            
            # Step environment
            if hasattr(self.env, 'step') and hasattr(self.env, 'num_envs'):
                # Vectorized environment
                if self.env.num_envs == 1:
                    next_obs, rewards, dones, infos = self.env.step(np.array([action]), compressed=True)
                    next_obs = next_obs[0]
                    reward = rewards[0]
                    done = dones[0]
                    info = infos[0]
                else:
                    raise ValueError("Evaluation environment should have num_envs=1")
            else:
                # Single environment
                next_obs, reward, done, info = self.env.step(action, compressed=True)
            
            episode_return += reward
            episode_length += 1
            obs = next_obs
        
        return episode_return, episode_length
    
    def _save_best_agent(self, step: int, score: float):
        """Save the best performing agent"""
        if not self.save_best_so_far_agent:
            return
        
        try:
            # Save policy and value networks
            policy_path = os.path.join(self.best_agent_path, 'policy')
            value_path = os.path.join(self.best_agent_path, 'value_net')
            
            self.agent.policy.save_weights(policy_path)
            self.agent.value_net.save_weights(value_path)
            
            # Save agent metadata
            metadata = {
                'step': step,
                'score': score,
                'timestamp': datetime.now().isoformat(),
                'evaluation_count': self.evaluation_count
            }
            
            metadata_path = os.path.join(self.best_agent_path, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Best agent saved to {self.best_agent_path}")
            
        except Exception as e:
            print(f"Failed to save best agent: {e}")
    
    def _write_eval_log(self, eval_results: Dict):
        """Write evaluation results to log file"""
        try:
            with open(self.eval_log_path, 'a') as f:
                f.write(json.dumps(eval_results) + '\n')
        except Exception as e:
            print(f"Failed to write evaluation log: {e}")
    
    def load_best_agent(self) -> bool:
        """
        Load the best saved agent
        
        Returns:
            True if successful, False otherwise
        """
        if not self.save_best_so_far_agent:
            return False
        
        try:
            policy_path = os.path.join(self.best_agent_path, 'policy')
            value_path = os.path.join(self.best_agent_path, 'value_net')
            metadata_path = os.path.join(self.best_agent_path, 'metadata.json')
            
            if not (os.path.exists(policy_path + '.index') and 
                    os.path.exists(value_path + '.index') and 
                    os.path.exists(metadata_path)):
                print("Best agent files not found")
                return False
            
            # Load weights
            self.agent.policy.load_weights(policy_path)
            self.agent.value_net.load_weights(value_path)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"Best agent loaded (step {metadata['step']}, score {metadata['score']:.3f})")
            return True
            
        except Exception as e:
            print(f"Failed to load best agent: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get evaluation statistics"""
        if not self.eval_history:
            return {}
        
        recent_returns = [eval_result['mean_return'] for eval_result in self.eval_history[-10:]]
        
        return {
            'evaluation_count': self.evaluation_count,
            'max_score': self.max_score,
            'recent_mean_return': np.mean(recent_returns) if recent_returns else 0.0,
            'recent_std_return': np.std(recent_returns) if recent_returns else 0.0,
            'last_eval_step': self.last_eval_step
        }
