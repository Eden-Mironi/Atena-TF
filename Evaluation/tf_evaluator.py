#!/usr/bin/env python3
"""
ATENA-TF Evaluator - Comprehensive evaluation system for TensorFlow models
Based on ATENA-master evaluation methodology, streamlined for TF implementation
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'Configuration'))
sys.path.append(os.path.join(parent_dir, 'models/ppo'))

import config as cfg
from models.ppo.agent import PPOAgent
from gym_atena.envs.enhanced_atena_env import make_enhanced_atena_env
import gym_atena.global_env_prop as gep

logger = logging.getLogger(__name__)


class EvaluationResult:
    """Container for evaluation results of a single dataset"""
    
    def __init__(self, dataset_id: int):
        self.dataset_id = dataset_id
        self.rewards = []
        self.session_lengths = []
        self.actions_taken = []
        self.reward_components = []
        
    def add_session(self, total_reward: float, session_length: int, 
                   actions: List, reward_info: Any = None):
        """Add results from a single evaluation session"""
        self.rewards.append(total_reward)
        self.session_lengths.append(session_length)
        self.actions_taken.append(actions)
        if reward_info:
            self.reward_components.append(reward_info)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for this dataset"""
        return {
            'dataset_id': self.dataset_id,
            'num_sessions': len(self.rewards),
            'avg_reward': np.mean(self.rewards) if self.rewards else 0.0,
            'max_reward': np.max(self.rewards) if self.rewards else 0.0,
            'min_reward': np.min(self.rewards) if self.rewards else 0.0,
            'std_reward': np.std(self.rewards) if self.rewards else 0.0,
            'avg_session_length': np.mean(self.session_lengths) if self.session_lengths else 0.0,
            'max_session_length': np.max(self.session_lengths) if self.session_lengths else 0.0,
        }


class ATENATFEvaluator:
    """Comprehensive evaluator for ATENA-TF models"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 results_dir: str = "evaluation_results",
                 num_sessions_per_dataset: int = 10):
        """
        Initialize the evaluator
        
        Args:
            model_path: Path to trained model (if None, finds latest)
            results_dir: Directory to save evaluation results
            num_sessions_per_dataset: Number of evaluation sessions per dataset
        """
        self.model_path = model_path
        self.results_dir = results_dir
        self.num_sessions_per_dataset = num_sessions_per_dataset
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize components
        self.agent = None
        self.env = None
        self.evaluation_results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
    def find_latest_model(self) -> Optional[str]:
        """Find the latest trained model"""
        # Look for models in various locations
        search_patterns = [
            "results/*/trained_model*",
            "results/*/final_model*", 
            "*_model_*/trained_model*",
            "evaluation_model_*/trained_model*",
        ]
        
        all_models = []
        for pattern in search_patterns:
            all_models.extend(glob.glob(pattern))
            
        if not all_models:
            logger.warning("No trained models found!")
            return None
            
        # Return the most recent one
        latest_model = max(all_models, key=lambda x: os.path.getmtime(x))
        logger.info(f"Found latest model: {latest_model}")
        return latest_model
        
    def load_model(self, model_path: str) -> bool:
        """Load the trained model"""
        try:
            # Initialize agent
            obs_dim = 51  # Standard ATENA observation dimension
            action_dim = 6  # Standard ATENA action dimension
            
            self.agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim)
            
            # Load model
            success = self.agent.load_model(model_path)
            if success:
                logger.info(f"Successfully loaded model from {model_path}")
                return True
            else:
                logger.error(f"Failed to load model from {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
            
    def evaluate_dataset(self, dataset_id: int, verbose: bool = True) -> EvaluationResult:
        """Evaluate agent performance on a specific dataset"""
        if verbose:
            print(f"\nEvaluating Dataset {dataset_id}")
            print("-" * 50)
            
        result = EvaluationResult(dataset_id)
        
        # Create environment for this dataset
        env = make_enhanced_atena_env(max_steps=cfg.MAX_NUM_OF_STEPS)
        
        for session_idx in range(self.num_sessions_per_dataset):
            if verbose:
                print(f"  Session {session_idx + 1}/{self.num_sessions_per_dataset}", end=" ")
                
            try:
                # Reset environment with specific dataset
                obs = env.reset(dataset_number=dataset_id)
                
                total_reward = 0.0
                session_actions = []
                step_count = 0
                done = False
                
                while not done and step_count < cfg.MAX_NUM_OF_STEPS:
                    # Get agent action
                    action_probs = self.agent.get_action_probabilities(obs)
                    action = np.argmax(action_probs)
                    action_vector = env.translate_action_to_vector(action)
                    session_actions.append(action_vector)
                    
                    # Take step
                    obs, reward, done, info = env.step(action_vector)
                    total_reward += reward
                    step_count += 1
                    
                # Add session results
                result.add_session(total_reward, step_count, session_actions, info)
                
                if verbose:
                    print(f"→ Reward: {total_reward:.3f}, Steps: {step_count}")
                    
            except Exception as e:
                logger.error(f"Error in session {session_idx} for dataset {dataset_id}: {e}")
                if verbose:
                    print(f"→ ERROR: {str(e)[:50]}")
                    
        env.close()
        return result
        
    def evaluate_all_datasets(self, datasets: List[int] = [0, 1, 2, 3]) -> Dict[int, EvaluationResult]:
        """Evaluate agent on all specified datasets"""
        print(f"\nATENA-TF Model Evaluation")
        print("=" * 60)
        print(f"Model: {self.model_path}")
        print(f"Sessions per dataset: {self.num_sessions_per_dataset}")
        print(f"Datasets: {datasets}")
        
        results = {}
        for dataset_id in datasets:
            try:
                result = self.evaluate_dataset(dataset_id)
                results[dataset_id] = result
                self.evaluation_results[dataset_id] = result
            except Exception as e:
                logger.error(f"Failed to evaluate dataset {dataset_id}: {e}")
                
        return results
        
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        if not self.evaluation_results:
            return {}
            
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'evaluation_config': {
                'num_sessions_per_dataset': self.num_sessions_per_dataset,
                'max_steps': cfg.MAX_NUM_OF_STEPS,
                'reward_coefficients': {
                    'humanity_coeff': cfg.humanity_coeff,
                    'diversity_coeff': cfg.diversity_coeff, 
                    'kl_coeff': cfg.kl_coeff,
                    'compaction_coeff': cfg.compaction_coeff,
                }
            },
            'datasets': {},
            'overall_summary': {}
        }
        
        # Per-dataset summaries
        all_rewards = []
        all_lengths = []
        
        for dataset_id, result in self.evaluation_results.items():
            summary = result.get_summary()
            report['datasets'][dataset_id] = summary
            all_rewards.extend(result.rewards)
            all_lengths.extend(result.session_lengths)
            
        # Overall summary
        if all_rewards:
            report['overall_summary'] = {
                'total_sessions': len(all_rewards),
                'avg_reward': np.mean(all_rewards),
                'max_reward': np.max(all_rewards),
                'min_reward': np.min(all_rewards),
                'std_reward': np.std(all_rewards),
                'avg_session_length': np.mean(all_lengths),
                'reward_distribution': {
                    'percentile_25': np.percentile(all_rewards, 25),
                    'percentile_50': np.percentile(all_rewards, 50),
                    'percentile_75': np.percentile(all_rewards, 75),
                }
            }
            
        return report
        
    def save_results(self, filename: str = None) -> str:
        """Save evaluation results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
            
        filepath = os.path.join(self.results_dir, filename)
        
        report = self.generate_summary_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Results saved to: {filepath}")
        return filepath
        
    def generate_visualizations(self, save_plots: bool = True) -> Dict[str, Any]:
        """Generate visualization plots"""
        if not self.evaluation_results:
            logger.warning("No evaluation results to visualize")
            return {}
            
        plots_info = {}
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Rewards by Dataset
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ATENA-TF Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # Subplot 1: Average Rewards by Dataset
        dataset_ids = list(self.evaluation_results.keys())
        avg_rewards = [self.evaluation_results[d].get_summary()['avg_reward'] for d in dataset_ids]
        
        axes[0, 0].bar(dataset_ids, avg_rewards, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Average Reward by Dataset')
        axes[0, 0].set_xlabel('Dataset ID')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(avg_rewards):
            axes[0, 0].text(dataset_ids[i], v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
            
        # Subplot 2: Reward Distribution 
        all_rewards = []
        dataset_labels = []
        for dataset_id, result in self.evaluation_results.items():
            all_rewards.extend(result.rewards)
            dataset_labels.extend([f'Dataset {dataset_id}'] * len(result.rewards))
            
        df_rewards = pd.DataFrame({'Dataset': dataset_labels, 'Reward': all_rewards})
        sns.boxplot(data=df_rewards, x='Dataset', y='Reward', ax=axes[0, 1])
        axes[0, 1].set_title('Reward Distribution by Dataset')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Subplot 3: Session Lengths
        avg_lengths = [self.evaluation_results[d].get_summary()['avg_session_length'] for d in dataset_ids]
        axes[1, 0].bar(dataset_ids, avg_lengths, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('Average Session Length by Dataset')
        axes[1, 0].set_xlabel('Dataset ID') 
        axes[1, 0].set_ylabel('Average Steps')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(avg_lengths):
            axes[1, 0].text(dataset_ids[i], v + 0.1, f'{v:.1f}', ha='center', fontweight='bold')
            
        # Subplot 4: Performance Summary Table
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        # Create summary table data
        table_data = []
        for dataset_id in dataset_ids:
            summary = self.evaluation_results[dataset_id].get_summary()
            table_data.append([
                f"Dataset {dataset_id}",
                f"{summary['avg_reward']:.2f}",
                f"{summary['max_reward']:.2f}",
                f"{summary['avg_session_length']:.1f}"
            ])
            
        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=['Dataset', 'Avg Reward', 'Max Reward', 'Avg Length'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Performance Summary')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.results_dir, f"evaluation_plots_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plots_info['main_plot'] = plot_path
            logger.info(f"Plots saved to: {plot_path}")
            
        plt.show()
        
        return plots_info
        
    def compare_with_master_data(self, master_results_path: str = None) -> Dict[str, Any]:
        """Compare results with ATENA-master data if available"""
        comparison = {
            'tf_results': self.generate_summary_report(),
            'master_results': None,
            'comparison': None
        }
        
        # Try to load master results from Excel files
        master_files = [
            "rewards_analysis.xlsx",
            "rewards_summary.xlsx", 
            "../ATENA-master/rewards_analysis.xlsx",
            "../ATENA-master/rewards_summary.xlsx"
        ]
        
        for master_file in master_files:
            if os.path.exists(master_file):
                try:
                    df = pd.read_excel(master_file)
                    logger.info(f"Found master results: {master_file}")
                    # Process master data here
                    break
                except Exception as e:
                    logger.warning(f"Could not load {master_file}: {e}")
                    
        return comparison
        
    def run_complete_evaluation(self, datasets: List[int] = [0, 1, 2, 3]) -> str:
        """Run complete evaluation pipeline"""
        print(f"\nSTARTING COMPLETE ATENA-TF EVALUATION")
        print("=" * 80)
        
        # 1. Find and load model
        if self.model_path is None:
            self.model_path = self.find_latest_model()
            
        if self.model_path is None:
            raise ValueError("No trained model found!")
            
        if not self.load_model(self.model_path):
            raise ValueError(f"Failed to load model: {self.model_path}")
            
        # 2. Run evaluation
        print(f"\nEVALUATING ACROSS {len(datasets)} DATASETS")
        results = self.evaluate_all_datasets(datasets)
        
        # 3. Generate report
        print(f"\n📋 GENERATING COMPREHENSIVE REPORT")
        report_path = self.save_results()
        
        # 4. Create visualizations
        print(f"\nCREATING VISUALIZATIONS")
        plots_info = self.generate_visualizations()
        
        # 5. Print summary
        summary = self.generate_summary_report()['overall_summary']
        if summary:
            print(f"\nEVALUATION SUMMARY")
            print("-" * 40)
            print(f"Total Sessions: {summary['total_sessions']}")
            print(f"Average Reward: {summary['avg_reward']:.3f}")
            print(f"Max Reward: {summary['max_reward']:.3f}")
            print(f"Average Length: {summary['avg_session_length']:.1f} steps")
            
        print(f"\nEVALUATION COMPLETE!")
        print(f"Results saved to: {report_path}")
        
        return report_path


if __name__ == "__main__":
    # Quick test run
    evaluator = ATENATFEvaluator(num_sessions_per_dataset=3)  # Quick test
    try:
        evaluator.run_complete_evaluation([0, 1])  # Test on first two datasets
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
