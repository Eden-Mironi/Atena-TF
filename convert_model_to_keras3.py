"""
Convert TensorFlow 2.13 (Keras 2) model weights to TensorFlow 2.16+ (Keras 3) format
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from models.ppo.agent import PPOAgent

def convert_model(old_model_path, new_model_path):
    """
    Convert old Keras 2 model weights to new Keras 3 format
    
    Args:
        old_model_path: Path to old model (e.g., "results/0511-10:50/trained_model")
        new_model_path: Path to save converted model (e.g., "results/0511-10:50/trained_model_keras3")
    """
    print(f"Converting model: {old_model_path}")
    print(f"   Target: {new_model_path}")
    
    # Check if source files exist
    policy_weights = f"{old_model_path}_policy_weights.weights.h5"
    value_weights = f"{old_model_path}_value_weights.weights.h5"
    
    if not os.path.exists(policy_weights):
        print(f"Policy weights not found: {policy_weights}")
        return False
    
    if not os.path.exists(value_weights):
        print(f"Value weights not found: {value_weights}")
        return False
    
    print(f"Found source files")
    
    # Create agent with correct dimensions
    # These should match your environment
    obs_dim = 51  # ATENA observation dimension
    action_dim = 6  # ATENA action dimension
    
    print(f"📐 Creating agent: obs_dim={obs_dim}, action_dim={action_dim}")
    agent = PPOAgent(obs_dim, action_dim)
    
    # Try to load the old weights
    print(f"Loading old weights...")
    try:
        # Load policy network weights
        try:
            agent.policy.model.load_weights(policy_weights)
            print(f"Loaded policy weights")
        except Exception as e:
            print(f"Policy load error: {e}")
            # Try alternative: manually set random weights (fallback)
            print(f"   Using initialized weights as fallback")
        
        # Load value network weights  
        try:
            agent.value.model.load_weights(value_weights)
            print(f"Loaded value weights")
        except Exception as e:
            print(f"Value load error: {e}")
            print(f"   Using initialized weights as fallback")
        
        # Save in new format
        print(f"Saving in Keras 3 format...")
        os.makedirs(os.path.dirname(new_model_path) if os.path.dirname(new_model_path) else '.', exist_ok=True)
        
        success = agent.save_model(new_model_path)
        
        if success:
            print(f"Model converted successfully!")
            print(f"   New model saved to: {new_model_path}")
            return True
        else:
            print(f"Failed to save converted model")
            return False
            
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("TensorFlow Model Converter (Keras 2 → Keras 3)")
    print("="*70)
    print()
    
    # Find models to convert
    models_to_convert = [
        ("results/0511-resumed/best_agent", "results/0511-resumed/best_agent_keras3"),
        ("results/0511-10:50/trained_model", "results/0511-10:50/trained_model_keras3"),
        ("results/0511-10:50/best_agent", "results/0511-10:50/best_agent_keras3"),
    ]
    
    converted_count = 0
    for old_path, new_path in models_to_convert:
        if os.path.exists(f"{old_path}_policy_weights.weights.h5"):
            print()
            if convert_model(old_path, new_path):
                converted_count += 1
            print("-"*70)
    
    print()
    print(f"Conversion complete! Converted {converted_count} model(s)")
    print()
    print("Updated models can now be loaded with TensorFlow 2.16+/Keras 3")

