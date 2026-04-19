"""
CRITICAL: Algorithm Selection System
Based on ATENA-master's algorithm selection and argument parsing

Implements master's extensible algorithm selection:
- AlgoName enum for different algorithms
- Agent factory for creating different agent types
- Configuration system for algorithm-specific parameters
"""

import sys
from enum import Enum
from typing import Any, Dict, Optional, Tuple
import argparse

sys.path.append('./Configuration')
sys.path.append('./models/ppo')

import config as cfg
from models.ppo.agent import PPOAgent

class AlgoName(Enum):
    """
    MASTER-EXACT: Algorithm names enum
    Based on ATENA-master arguments.py AlgoName
    """
    PPO = 'PPO'
    CHAINERRL_PPO = 'CHAINERRL_PPO'  # Master's main algorithm
    A2C = 'A2C'
    SAC = 'SAC'
    DDPG = 'DDPG'
    
    def __str__(self):
        return self.value

class ArchName(Enum):
    """
    MASTER-EXACT: Architecture names enum
    Based on ATENA-master arguments.py ArchName
    """
    FF_GAUSSIAN = 'FF_GAUSSIAN'          # Master's main architecture  
    FF_SOFTMAX = 'FF_SOFTMAX'
    FF_PARAM_SOFTMAX = 'FF_PARAM_SOFTMAX'
    
    def __str__(self):
        return self.value

class SchemaName(Enum):
    """
    MASTER-EXACT: Schema names enum
    Based on ATENA-master arguments.py SchemaName
    """
    NETWORKING = 'NETWORKING'
    FLIGHTS = 'FLIGHTS'  
    BIG_FLIGHTS = 'BIG_FLIGHTS'
    WIDE_FLIGHTS = 'WIDE_FLIGHTS'
    WIDE12_FLIGHTS = 'WIDE12_FLIGHTS'
    
    def __str__(self):
        return self.value

class AlgorithmConfiguration:
    """
    Configuration class for algorithm-specific parameters
    """
    
    def __init__(self, 
                 algo: AlgoName = AlgoName.PPO,
                 arch: ArchName = ArchName.FF_GAUSSIAN,
                 schema: SchemaName = SchemaName.NETWORKING):
        self.algo = algo
        self.arch = arch  
        self.schema = schema
        
        # Algorithm-specific parameters
        self.algo_params = self._get_default_algo_params()
        self.arch_params = self._get_default_arch_params()
        self.schema_params = self._get_default_schema_params()
    
    def _get_default_algo_params(self) -> Dict[str, Any]:
        """Get default parameters for the selected algorithm"""
        
        if self.algo == AlgoName.PPO or self.algo == AlgoName.CHAINERRL_PPO:
            return {
                'learning_rate': cfg.adam_lr,           # 3e-4
                'clip_ratio': 0.2,
                'value_coef': 0.5,
                'entropy_coef': cfg.entropy_coef,       # 0.0
                'update_interval': 2048,                # Master's update_interval
                'minibatch_size': cfg.batchsize,        # 64
                'epochs': cfg.epochs,                   # 10
                'gamma': cfg.ppo_gamma,                 # 0.995
                'lambda_': cfg.ppo_lambda,              # 0.97
                'clip_eps_vf': None,                   # Master's clip_eps_vf
                'standardize_advantages': True,         # Master uses this
            }
        elif self.algo == AlgoName.A2C:
            return {
                'learning_rate': 3e-4,
                'value_coef': 0.5,
                'entropy_coef': 0.01,
                'gamma': 0.99,
            }
        # Add other algorithms as needed
        else:
            return {}
    
    def _get_default_arch_params(self) -> Dict[str, Any]:
        """Get default parameters for the selected architecture"""
        
        if self.arch == ArchName.FF_GAUSSIAN:
            return {
                'n_hidden_channels': 64,        # Master's hidden layer size
                'n_hidden_layers': 2,           # Master's hidden layers  
                'bound_mean': True,             # Master uses bound_mean=True
                'activation': 'tanh',           # Master's activation
                'mean_wscale': 1.0,            # Master's mean weight scale
            }
        elif self.arch == ArchName.FF_SOFTMAX:
            return {
                'n_hidden_channels': 600,       # Master's softmax hidden size
                'n_hidden_layers': 2,
                'activation': 'tanh',
            }
        elif self.arch == ArchName.FF_PARAM_SOFTMAX:
            return {
                'n_hidden_channels': 600,       # Master's param softmax hidden size
                'n_hidden_layers': 2,
                'activation': 'tanh',
            }
        else:
            return {}
    
    def _get_default_schema_params(self) -> Dict[str, Any]:
        """Get default parameters for the selected schema"""
        
        if self.schema == SchemaName.NETWORKING:
            return {
                'max_steps': cfg.MAX_NUM_OF_STEPS,
                'eval_datasets': ['dataset_0', 'dataset_1', 'dataset_2'],
                'human_rules_type': 'NetHumanRule',
            }
        elif self.schema == SchemaName.FLIGHTS:
            return {
                'max_steps': cfg.MAX_NUM_OF_STEPS,
                'eval_datasets': ['flights_dataset'],
                'human_rules_type': 'FlightsHumanRule',
            }
        # Add other schemas as needed
        else:
            return {}

class AgentFactory:
    """
    MASTER-EXACT: Agent factory for creating different agent types
    Based on master's agent initialization patterns
    """
    
    @staticmethod
    def create_agent(config: AlgorithmConfiguration, 
                    obs_dim: int, 
                    action_dim: int,
                    **kwargs) -> Any:
        """
        Create agent based on configuration
        
        Args:
            config: Algorithm configuration
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            **kwargs: Additional agent parameters
            
        Returns:
            Configured agent instance
        """
        
        if config.algo == AlgoName.PPO or config.algo == AlgoName.CHAINERRL_PPO:
            # Combine configuration parameters with kwargs
            agent_params = {**config.algo_params, **kwargs}
            
            return PPOAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                **agent_params
            )
        
        elif config.algo == AlgoName.A2C:
            # Could implement A2C agent here
            raise NotImplementedError(f"A2C agent not implemented yet")
        
        elif config.algo == AlgoName.SAC:
            # Could implement SAC agent here
            raise NotImplementedError(f"SAC agent not implemented yet")
        
        else:
            raise ValueError(f"Unknown algorithm: {config.algo}")

def create_argument_parser():
    """
    MASTER-EXACT: Create argument parser matching master's arguments.py
    Based on master's command-line argument structure
    """
    
    parser = argparse.ArgumentParser(description='ATENA Training with Algorithm Selection')
    
    # Algorithm selection
    parser.add_argument('--algo', type=str, default='PPO',
                       choices=[algo.value for algo in AlgoName],
                       help='Algorithm to use')
    
    parser.add_argument('--arch', type=str, default='FF_GAUSSIAN', 
                       choices=[arch.value for arch in ArchName],
                       help='Network architecture')
    
    parser.add_argument('--schema', type=str, default='NETWORKING',
                       choices=[schema.value for schema in SchemaName], 
                       help='Dataset schema')
    
    # Training parameters
    parser.add_argument('--steps', type=int, default=1000000,
                       help='Total training steps')
    
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    
    parser.add_argument('--clip', type=float, default=0.2,
                       help='PPO clipping parameter')
    
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size')
    
    parser.add_argument('--epochs', type=int, default=10,
                       help='Training epochs per update')
    
    # Evaluation parameters
    parser.add_argument('--eval-interval', type=int, default=100000,
                       help='Evaluation interval')
    
    parser.add_argument('--eval-n-runs', type=int, default=10,
                       help='Number of evaluation episodes')
    
    # Output
    parser.add_argument('--outdir', type=str, default='results',
                       help='Output directory')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser

def parse_args_and_create_config(args=None):
    """
    Parse command-line arguments and create algorithm configuration
    
    Args:
        args: Command-line arguments (None for sys.argv)
        
    Returns:
        Tuple of (AlgorithmConfiguration, parsed_args)
    """
    
    parser = create_argument_parser()
    parsed_args = parser.parse_args(args)
    
    # Create configuration
    config = AlgorithmConfiguration(
        algo=AlgoName(parsed_args.algo),
        arch=ArchName(parsed_args.arch), 
        schema=SchemaName(parsed_args.schema)
    )
    
    # Override default parameters with command-line arguments
    if hasattr(parsed_args, 'lr'):
        config.algo_params['learning_rate'] = parsed_args.lr
    if hasattr(parsed_args, 'clip'):
        config.algo_params['clip_ratio'] = parsed_args.clip
    if hasattr(parsed_args, 'batch_size'):
        config.algo_params['minibatch_size'] = parsed_args.batch_size
    if hasattr(parsed_args, 'epochs'):
        config.algo_params['epochs'] = parsed_args.epochs
    
    return config, parsed_args

def main():
    """Example usage of algorithm selection system"""
    
    print("ALGORITHM SELECTION SYSTEM DEMO")
    print("="*50)
    
    # Parse command-line arguments
    config, args = parse_args_and_create_config()
    
    print(f"Selected Algorithm: {config.algo}")
    print(f"Selected Architecture: {config.arch}")  
    print(f"Selected Schema: {config.schema}")
    
    print(f"\nAlgorithm Parameters:")
    for key, value in config.algo_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nArchitecture Parameters:")
    for key, value in config.arch_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nSchema Parameters:")
    for key, value in config.schema_params.items():
        print(f"  {key}: {value}")
    
    # Example of creating an agent
    print(f"\nCreating agent with configuration...")
    try:
        agent = AgentFactory.create_agent(
            config=config,
            obs_dim=51,     # Example dimensions
            action_dim=6
        )
        print(f"Agent created successfully: {type(agent).__name__}")
    except Exception as e:
        print(f"Failed to create agent: {e}")

if __name__ == "__main__":
    main()
