#!/usr/bin/env python3
"""
ATENA-TF Evaluation System
Comprehensive evaluation framework for ATENA-TF models

Based on the original ATENA-master evaluation system but adapted for TensorFlow implementation.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from enum import Enum
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'Configuration'))
sys.path.append(os.path.join(project_root, 'models'))
sys.path.append(os.path.join(project_root, 'gym_atena/envs'))

import Configuration.config as cfg
from gym_atena.envs.enhanced_atena_env import make_enhanced_atena_env
from live_recommender_agent import TFRecommenderAgent, find_latest_trained_model


logger = logging.getLogger(__name__)


class EvaluationMetric(Enum):
    """Evaluation metrics supported by the TF evaluation system"""
    TOTAL_REWARD = 'total_reward'
    AVG_REWARD_PER_STEP = 'avg_reward_per_step'
    REWARD_COMPONENTS = 'reward_components'
    ACTION_DISTRIBUTION = 'action_distribution'
    SESSION_LENGTH = 'session_length'
    SUCCESS_RATE = 'success_rate'
    DIVERSITY_SCORE = 'diversity_score'
    HUMANITY_SCORE = 'humanity_score'
    INTERESTINGNESS_SCORE = 'interestingness_score'


class EvaluationResult:
    """Container for evaluation results"""
    
    def __init__(self, agent_type: str, dataset_number: int):
        self.agent_type = agent_type
        self.dataset_number = dataset_number
        self.timestamp = datetime.now()
        
        # Core metrics
        self.total_rewards: List[float] = []
        self.session_lengths: List[int] = []
        self.action_sequences: List[List] = []
        self.reward_breakdowns: List[Dict] = []
        
        # Component scores
        self.diversity_scores: List[float] = []
        self.humanity_scores: List[float] = []
        self.interestingness_scores: List[float] = []
        
        # Session info
        self.session_infos: List[Dict] = []
        
    def add_session_result(self, total_reward: float, actions: List, 
                          reward_breakdown: Dict, session_info: Dict):
        """Add results from a single session"""
        self.total_rewards.append(total_reward)
        self.session_lengths.append(len(actions))
        self.action_sequences.append(actions.copy())
        self.reward_breakdowns.append(reward_breakdown.copy())
        self.session_infos.append(session_info.copy())
        
        # Extract component scores
        if 'diversity_reward' in reward_breakdown:
            self.diversity_scores.append(reward_breakdown['diversity_reward'])
        if 'humanity_reward' in reward_breakdown:
            self.humanity_scores.append(reward_breakdown['humanity_reward'])
        if 'interestingness_reward' in reward_breakdown:
            self.interestingness_scores.append(reward_breakdown['interestingness_reward'])
    
    @property
    def avg_total_reward(self) -> float:
        return np.mean(self.total_rewards) if self.total_rewards else 0.0
    
    @property
    def avg_session_length(self) -> float:
        return np.mean(self.session_lengths) if self.session_lengths else 0.0
    
    @property
    def avg_diversity_score(self) -> float:
        return np.mean(self.diversity_scores) if self.diversity_scores else 0.0
    
    @property
    def avg_humanity_score(self) -> float:
        return np.mean(self.humanity_scores) if self.humanity_scores else 0.0
    
    @property
    def avg_interestingness_score(self) -> float:
        return np.mean(self.interestingness_scores) if self.interestingness_scores else 0.0


class ATENATFEvaluator:
    """
    Comprehensive evaluation system for ATENA-TF models
    
    Evaluates trained models across multiple datasets and provides detailed analysis
    similar to the original ATENA-master evaluation system.
    """
    
    def __init__(self, model_path: Optional[str] = None, schema: str = 'NETWORKING'):
        self.model_path = model_path or find_latest_trained_model()
        self.schema = schema
        self.results: Dict[str, EvaluationResult] = {}
        
        # Load model if available
        if self.model_path:
            logger.info(f"Loading model from: {self.model_path}")
        else:
            logger.warning("No model path provided - some evaluations will be skipped")
    
    def run_agent_session(self, dataset_number: int, max_steps: int = None) -> Tuple[List, float, Dict]:
        """
        Run a single session with the TF agent on specified dataset
        
        Returns:
            Tuple of (actions_taken, total_reward, reward_breakdown)
        """
        if not self.model_path:
            raise ValueError("No model available for evaluation")
        
        # Create agent
        agent = TFRecommenderAgent(
            model_path=self.model_path,
            dataset_number=dataset_number,
            schema=self.schema
        )
        
        # Run episode
        actions = []
        total_reward = 0.0
        reward_breakdown = defaultdict(float)
        
        obs = agent.env.reset()
        done = False
        step_count = 0
        max_steps = max_steps or cfg.MAX_NUM_OF_STEPS
        
        session_info = {
            'dataset_number': dataset_number,
            'initial_state': obs.copy(),
            'steps_taken': 0,
            'terminated_reason': 'unknown'
        }
        
        while not done and step_count < max_steps:
            # Get action from agent - USE DETERMINISTIC for evaluation consistency
            action, _, _ = agent.act_most_probable(obs)  # Updated to handle new return format
            actions.append(action.tolist() if hasattr(action, 'tolist') else action)
            
            # Take step in environment
            obs, reward, done, info = agent.env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Extract reward components if available
            if 'reward_info' in info:
                reward_info = info['reward_info']
                for key, value in reward_info.__dict__.items():
                    if isinstance(value, (int, float)):
                        reward_breakdown[key] += value
        
        session_info['steps_taken'] = step_count
        session_info['final_state'] = obs.copy()
        session_info['terminated_reason'] = 'done' if done else 'max_steps'
        
        return actions, total_reward, dict(reward_breakdown), session_info
    
    def evaluate_agent_on_dataset(self, dataset_number: int, n_sessions: int = 5) -> EvaluationResult:
        """
        Evaluate agent on a specific dataset across multiple sessions
        """
        logger.info(f"Evaluating agent on dataset {dataset_number} for {n_sessions} sessions")
        
        result = EvaluationResult(agent_type='tf_ppo', dataset_number=dataset_number)
        
        for session_idx in range(n_sessions):
            try:
                actions, total_reward, reward_breakdown, session_info = self.run_agent_session(dataset_number)
                result.add_session_result(total_reward, actions, reward_breakdown, session_info)
                logger.info(f"Session {session_idx + 1}/{n_sessions}: Reward = {total_reward:.3f}")
            except Exception as e:
                logger.error(f"Error in session {session_idx + 1}: {e}")
                continue
        
        self.results[f'dataset_{dataset_number}'] = result
        return result
    
    def evaluate_all_datasets(self, n_sessions_per_dataset: int = 5, dataset_range: Tuple[int, int] = None) -> Dict[str, EvaluationResult]:
        """
        Evaluate agent across all available datasets
        """
        # Determine dataset range
        if dataset_range is None:
            if self.schema == 'NETWORKING':
                dataset_range = (0, 4)  # Datasets 0-3 for networking
            else:
                dataset_range = (0, 2)  # Adjust based on schema
        
        start_idx, end_idx = dataset_range
        
        logger.info(f"Starting comprehensive evaluation on datasets {start_idx}-{end_idx}")
        
        all_results = {}
        for dataset_idx in range(start_idx, end_idx):
            try:
                result = self.evaluate_agent_on_dataset(dataset_idx, n_sessions_per_dataset)
                all_results[f'dataset_{dataset_idx}'] = result
            except Exception as e:
                logger.error(f"Failed to evaluate dataset {dataset_idx}: {e}")
                continue
        
        return all_results
    
    def compare_with_master_data(self, master_data_path: str = None) -> Dict[str, Any]:
        """
        Compare evaluation results with ATENA-master reference data
        """
        comparison_results = {}
        
        # Load master data if available
        master_rewards = None
        if master_data_path and os.path.exists(master_data_path):
            try:
                if master_data_path.endswith('.xlsx'):
                    master_df = pd.read_excel(master_data_path)
                    master_rewards = master_df['reward'].values if 'reward' in master_df.columns else None
                elif master_data_path.endswith('.json'):
                    with open(master_data_path, 'r') as f:
                        master_data = json.load(f)
                        master_rewards = master_data.get('rewards', [])
            except Exception as e:
                logger.error(f"Error loading master data: {e}")
        
        # Compare each dataset result
        for dataset_key, result in self.results.items():
            dataset_comparison = {
                'tf_avg_reward': result.avg_total_reward,
                'tf_reward_std': np.std(result.total_rewards),
                'tf_avg_session_length': result.avg_session_length,
                'tf_n_sessions': len(result.total_rewards)
            }
            
            if master_rewards is not None:
                dataset_comparison.update({
                    'master_avg_reward': np.mean(master_rewards),
                    'master_reward_std': np.std(master_rewards),
                    'reward_difference': result.avg_total_reward - np.mean(master_rewards),
                    'accuracy_percentage': (result.avg_total_reward / np.mean(master_rewards)) * 100
                })
            
            comparison_results[dataset_key] = dataset_comparison
        
        return comparison_results
    
    def generate_evaluation_report(self, output_dir: str = "evaluation_results") -> str:
        """
        Generate comprehensive evaluation report
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"tf_evaluation_report_{timestamp}.json")
        
        # Compile report data
        report = {
            'timestamp': timestamp,
            'model_path': self.model_path,
            'schema': self.schema,
            'evaluation_summary': {},
            'dataset_results': {},
            'overall_statistics': {}
        }
        
        # Add dataset results
        total_rewards = []
        for dataset_key, result in self.results.items():
            total_rewards.extend(result.total_rewards)
            
            report['dataset_results'][dataset_key] = {
                'avg_total_reward': result.avg_total_reward,
                'reward_std': np.std(result.total_rewards),
                'avg_session_length': result.avg_session_length,
                'n_sessions': len(result.total_rewards),
                'avg_diversity_score': result.avg_diversity_score,
                'avg_humanity_score': result.avg_humanity_score,
                'avg_interestingness_score': result.avg_interestingness_score,
                'reward_range': [min(result.total_rewards), max(result.total_rewards)] if result.total_rewards else [0, 0]
            }
        
        # Overall statistics
        if total_rewards:
            report['overall_statistics'] = {
                'total_sessions_evaluated': len(total_rewards),
                'overall_avg_reward': np.mean(total_rewards),
                'overall_reward_std': np.std(total_rewards),
                'overall_reward_range': [min(total_rewards), max(total_rewards)],
                'datasets_evaluated': len(self.results)
            }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to: {report_path}")
        return report_path
    
    def plot_evaluation_results(self, output_dir: str = "evaluation_results"):
        """
        Generate visualization plots for evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Plot 1: Reward distribution across datasets
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ATENA-TF Evaluation Results', fontsize=16)
        
        # Reward distribution
        dataset_names = []
        reward_means = []
        reward_stds = []
        
        for dataset_key, result in self.results.items():
            dataset_names.append(dataset_key.replace('dataset_', 'Dataset '))
            reward_means.append(result.avg_total_reward)
            reward_stds.append(np.std(result.total_rewards))
        
        axes[0, 0].bar(dataset_names, reward_means, yerr=reward_stds, capsize=5)
        axes[0, 0].set_title('Average Total Reward by Dataset')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Session length distribution
        session_lengths = []
        for result in self.results.values():
            session_lengths.extend(result.session_lengths)
        
        if session_lengths:
            axes[0, 1].hist(session_lengths, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Session Length Distribution')
            axes[0, 1].set_xlabel('Session Length (steps)')
            axes[0, 1].set_ylabel('Frequency')
        
        # Reward components comparison
        if any(result.diversity_scores for result in self.results.values()):
            component_data = []
            for dataset_key, result in self.results.items():
                if result.diversity_scores:
                    component_data.append({
                        'Dataset': dataset_key.replace('dataset_', 'D'),
                        'Diversity': np.mean(result.diversity_scores),
                        'Humanity': np.mean(result.humanity_scores) if result.humanity_scores else 0,
                        'Interestingness': np.mean(result.interestingness_scores) if result.interestingness_scores else 0
                    })
            
            if component_data:
                df_components = pd.DataFrame(component_data)
                x_pos = np.arange(len(df_components))
                width = 0.25
                
                axes[1, 0].bar(x_pos - width, df_components['Diversity'], width, label='Diversity')
                axes[1, 0].bar(x_pos, df_components['Humanity'], width, label='Humanity')
                axes[1, 0].bar(x_pos + width, df_components['Interestingness'], width, label='Interestingness')
                
                axes[1, 0].set_title('Reward Components by Dataset')
                axes[1, 0].set_xlabel('Dataset')
                axes[1, 0].set_ylabel('Average Score')
                axes[1, 0].set_xticks(x_pos)
                axes[1, 0].set_xticklabels(df_components['Dataset'])
                axes[1, 0].legend()
        
        # Overall reward distribution
        all_rewards = []
        for result in self.results.values():
            all_rewards.extend(result.total_rewards)
        
        if all_rewards:
            axes[1, 1].hist(all_rewards, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Overall Reward Distribution')
            axes[1, 1].set_xlabel('Total Reward')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].axvline(np.mean(all_rewards), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(all_rewards):.3f}')
            axes[1, 1].legend()
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"tf_evaluation_plots_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation plots saved to: {plot_path}")


def evaluate_model(model_path: Optional[str] = None, 
                   datasets: List[int] = None,
                   n_episodes: int = 5,
                   schema: str = 'NETWORKING',
                   output_dir: str = 'evaluation_results') -> Dict[int, EvaluationResult]:
    """
    Convenience function for evaluating a model from notebooks
    
    Args:
        model_path: Path to trained model (None = auto-detect latest)
        datasets: List of dataset IDs to evaluate (None = [0,1,2,3])
        n_episodes: Number of episodes per dataset
        schema: Dataset schema ('NETWORKING', 'FLIGHTS', etc.)
        output_dir: Where to save evaluation results
        
    Returns:
        Dictionary mapping dataset_id -> EvaluationResult
    """
    # Auto-detect model if not provided
    if model_path is None:
        try:
            model_path = find_latest_trained_model()
            print(f"Auto-detected model: {model_path}")
        except FileNotFoundError:
            print("No trained model found!")
            raise
    
    # Default to all datasets if not specified
    if datasets is None:
        datasets = list(range(4))  # Datasets 0-3
    
    print(f"Evaluating ATENA-TF Model")
    print(f"Model: {model_path}")
    print(f"Datasets: {datasets}")
    print(f"🔢 Episodes per dataset: {n_episodes}")
    print("="*60)
    
    # Create evaluator
    evaluator = ATENATFEvaluator(model_path=model_path, schema=schema)
    
    # Run evaluation on specified datasets
    results = {}
    for dataset_id in datasets:
        print(f"\nEvaluating Dataset {dataset_id}...")
        dataset_results = []
        
        for episode in range(n_episodes):
            try:
                actions, reward, breakdown = evaluator.run_agent_session(
                    dataset_number=dataset_id,
                    max_steps=cfg.MAX_NUM_OF_STEPS
                )
                dataset_results.append({
                    'actions': actions,
                    'total_reward': reward,
                    'reward_breakdown': breakdown,
                    'episode': episode
                })
                print(f"  Episode {episode+1}/{n_episodes}: Reward={reward:.2f}, Steps={len(actions)}")
            except Exception as e:
                print(f"  Episode {episode+1} failed: {e}")
                continue
        
        # Aggregate results
        if dataset_results:
            avg_reward = np.mean([r['total_reward'] for r in dataset_results])
            avg_steps = np.mean([len(r['actions']) for r in dataset_results])
            
            results[dataset_id] = {
                'episodes': dataset_results,
                'avg_reward': avg_reward,
                'avg_steps': avg_steps,
                'n_episodes': len(dataset_results)
            }
            
            print(f"  Dataset {dataset_id}: Avg Reward={avg_reward:.2f}, Avg Steps={avg_steps:.1f}")
        else:
            print(f"  Dataset {dataset_id}: No successful episodes")
    
    print("\n" + "="*60)
    print(f"Evaluation Complete!")
    print(f"Successfully evaluated {len(results)} datasets")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"evaluation_{timestamp}.json")
    
    # Convert to JSON-serializable format
    json_results = {}
    for dataset_id, data in results.items():
        json_results[str(dataset_id)] = {
            'avg_reward': float(data['avg_reward']),
            'avg_steps': float(data['avg_steps']),
            'n_episodes': int(data['n_episodes'])
        }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    return results


def main():
    """Main evaluation script"""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create evaluator
    evaluator = ATENATFEvaluator(schema='NETWORKING')
    
    if not evaluator.model_path:
        print("No trained model found!")
        print("Please run training first: python main.py --episodes 100 --outdir results")
        return
    
    print(f"Starting ATENA-TF Evaluation")
    print(f"Using model: {evaluator.model_path}")
    
    # Run evaluation
    try:
        results = evaluator.evaluate_all_datasets(n_sessions_per_dataset=3, dataset_range=(0, 2))
        
        print(f"\nEvaluation completed!")
        print(f"Evaluated {len(results)} datasets")
        
        # Generate report
        report_path = evaluator.generate_evaluation_report()
        print(f"📄 Report saved: {report_path}")
        
        # Generate plots
        evaluator.plot_evaluation_results()
        print(f"Plots generated in evaluation_results/")
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for dataset_key, result in results.items():
            print(f"\n{dataset_key.upper()}:")
            print(f"  Average Reward: {result.avg_total_reward:.3f}")
            print(f"  Sessions: {len(result.total_rewards)}")
            print(f"  Avg Length: {result.avg_session_length:.1f} steps")
            if result.diversity_scores:
                print(f"  Diversity: {result.avg_diversity_score:.3f}")
                print(f"  Humanity: {result.avg_humanity_score:.3f}")
                print(f"  Interestingness: {result.avg_interestingness_score:.3f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
