"""
CRITICAL: Master's Hook System Implementation
Based on ATENA-master's chainerrl.experiments.LinearInterpolationHook

Implements master's exact hook system for extensible training:
- LinearInterpolationHook for parameter decay
- Step hooks called every training step
- Extensible hook architecture
"""

import sys
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

sys.path.append('../Configuration')
import config as cfg

class TrainingHook(ABC):
    """
    MASTER-EXACT: Base training hook class
    Based on ChainerRL's hook system architecture
    """
    
    @abstractmethod
    def __call__(self, env, agent, step: int):
        """
        Execute hook logic
        
        Args:
            env: Training environment
            agent: Training agent
            step: Current training step
        """
        pass

class LinearInterpolationHook(TrainingHook):
    """
    MASTER-EXACT: Linear interpolation hook for parameter decay
    Based on chainerrl.experiments.LinearInterpolationHook
    
    Exactly matches master's implementation for:
    - Learning rate decay
    - Clipping parameter decay
    - Any linear parameter interpolation
    """
    
    def __init__(self, 
                 total_steps: int,
                 initial_value: float, 
                 final_value: float,
                 parameter_setter: Callable[[float], None],
                 name: str = "LinearInterpolation"):
        """
        Initialize linear interpolation hook
        
        Args:
            total_steps: Total training steps for interpolation
            initial_value: Starting parameter value
            final_value: Ending parameter value  
            parameter_setter: Function to set the parameter value
            name: Hook name for logging
        """
        self.total_steps = total_steps
        self.initial_value = initial_value
        self.final_value = final_value
        self.parameter_setter = parameter_setter
        self.name = name
        
        print(f"🔗 {name} Hook initialized:")
        print(f"   {initial_value:.6f} → {final_value:.6f} over {total_steps:,} steps")
    
    def __call__(self, env, agent, step: int):
        """
        CRITICAL: Master's exact linear interpolation formula
        Based on chainerrl's LinearInterpolationHook implementation
        """
        if step >= self.total_steps:
            # After total steps, use final value
            current_value = self.final_value
        else:
            # Linear interpolation: current = initial + (final - initial) * (step / total_steps)
            progress = step / self.total_steps
            current_value = self.initial_value + (self.final_value - self.initial_value) * progress
        
        # Set the parameter using the provided setter
        self.parameter_setter(current_value)
        
        # Log periodically (every 1000 steps)
        if step % 1000 == 0 and step > 0:
            print(f"   🔗 {self.name}: Step {step:6d} → {current_value:.6f}")

class LoggingHook(TrainingHook):
    """
    Hook for periodic logging and statistics
    """
    
    def __init__(self, log_interval: int = 1000, name: str = "Logging"):
        self.log_interval = log_interval
        self.name = name
        self.last_log_step = -1
    
    def __call__(self, env, agent, step: int):
        if step % self.log_interval == 0 and step != self.last_log_step:
            self.last_log_step = step
            
            # Get basic statistics
            if hasattr(agent, 'entropy_values') and len(agent.entropy_values) > 0:
                avg_entropy = np.mean(list(agent.entropy_values)[-10:])
                print(f"   Step {step:6d}: Entropy = {avg_entropy:.4f}")

class EvaluationHook(TrainingHook):
    """
    Hook for periodic evaluation during training
    """
    
    def __init__(self, evaluator, eval_interval: int = 100000, name: str = "Evaluation"):
        self.evaluator = evaluator
        self.eval_interval = eval_interval
        self.name = name
        self.last_eval_step = -1
    
    def __call__(self, env, agent, step: int):
        if (step % self.eval_interval == 0 and 
            step > 0 and 
            step != self.last_eval_step and
            self.evaluator is not None):
            
            self.last_eval_step = step
            print(f"   {self.name}: Triggering evaluation at step {step}")
            
            # Trigger evaluation
            eval_results = self.evaluator.evaluate_if_necessary(step)
            if eval_results:
                print(f"   Evaluation complete: Mean return = {eval_results['mean_return']:.3f}")

class CustomHook(TrainingHook):
    """
    Custom hook for user-defined training logic
    """
    
    def __init__(self, 
                 hook_function: Callable[[Any, Any, int], None],
                 name: str = "Custom",
                 call_interval: int = 1):
        self.hook_function = hook_function
        self.name = name
        self.call_interval = call_interval
    
    def __call__(self, env, agent, step: int):
        if step % self.call_interval == 0:
            self.hook_function(env, agent, step)

class HookManager:
    """
    MASTER-EXACT: Hook management system
    Based on master's step_hooks usage in training loops
    
    Manages and executes training hooks in the correct order
    """
    
    def __init__(self):
        self.hooks = []
        self.hook_stats = {}
    
    def add_hook(self, hook: TrainingHook):
        """Add a training hook"""
        self.hooks.append(hook)
        self.hook_stats[hook.name] = {'calls': 0, 'last_call_step': -1}
        print(f"Added hook: {hook.name}")
    
    def remove_hook(self, hook_name: str):
        """Remove a hook by name"""
        self.hooks = [h for h in self.hooks if h.name != hook_name]
        if hook_name in self.hook_stats:
            del self.hook_stats[hook_name]
        print(f"Removed hook: {hook_name}")
    
    def execute_hooks(self, env, agent, step: int):
        """
        CRITICAL: Execute all hooks (master's step_hooks execution)
        Based on train_agent_chainerrl.py lines 209-210:
        ```python
        for hook in step_hooks:
            hook(env, agent, t)
        ```
        """
        for hook in self.hooks:
            try:
                hook(env, agent, step)
                
                # Update statistics
                self.hook_stats[hook.name]['calls'] += 1
                self.hook_stats[hook.name]['last_call_step'] = step
                
            except Exception as e:
                print(f"Hook {hook.name} failed at step {step}: {e}")
    
    def get_hook_statistics(self):
        """Get hook execution statistics"""
        return dict(self.hook_stats)
    
    def log_hook_statistics(self):
        """Log hook statistics"""
        print("🔗 HOOK STATISTICS:")
        for hook_name, stats in self.hook_stats.items():
            print(f"   {hook_name}: {stats['calls']} calls, last at step {stats['last_call_step']}")

def create_master_hooks(agent, total_steps: int, initial_lr: float = 3e-4, initial_clip: float = 0.2):
    """
    MASTER-EXACT: Create hooks matching master's training setup
    Based on train.py lines 97-98:
    ```python
    step_hooks=[
        lr_decay_hook,
        clip_eps_decay_hook,
    ]
    ```
    """
    hooks = []
    
    # Learning rate decay hook (master's lr_decay_hook)
    def lr_setter(lr_value):
        if hasattr(agent, 'optimizer') and hasattr(agent.optimizer, 'learning_rate'):
            agent.optimizer.learning_rate.assign(lr_value)
    
    lr_decay_hook = LinearInterpolationHook(
        total_steps=total_steps,
        initial_value=initial_lr,
        final_value=0.0,
        parameter_setter=lr_setter,
        name="LearningRateDecay"
    )
    hooks.append(lr_decay_hook)
    
    # Clipping parameter decay hook (master's clip_eps_decay_hook)  
    def clip_setter(clip_value):
        if hasattr(agent, 'clip_ratio'):
            agent.clip_ratio = clip_value
    
    clip_decay_hook = LinearInterpolationHook(
        total_steps=total_steps,
        initial_value=initial_clip,
        final_value=0.0,
        parameter_setter=clip_setter,
        name="ClippingDecay"
    )
    hooks.append(clip_decay_hook)
    
    return hooks

def create_hook_manager_with_master_hooks(agent, total_steps: int, evaluator=None):
    """
    Create a complete hook manager with master's hooks plus additional useful hooks
    """
    manager = HookManager()
    
    # Add master's core hooks
    master_hooks = create_master_hooks(agent, total_steps)
    for hook in master_hooks:
        manager.add_hook(hook)
    
    # Add logging hook
    logging_hook = LoggingHook(log_interval=1000, name="Statistics")
    manager.add_hook(logging_hook)
    
    # Add evaluation hook if evaluator provided
    if evaluator is not None:
        eval_hook = EvaluationHook(evaluator, eval_interval=100000, name="PerformanceEval")
        manager.add_hook(eval_hook)
    
    print(f"🔗 Hook Manager created with {len(manager.hooks)} hooks")
    return manager
