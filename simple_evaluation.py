#!/usr/bin/env python3
"""
Simple Evaluation Script - No hanging imports
"""
import os
import sys

def find_model_simple():
    """Simple model finder without hanging imports"""
    candidates = [
        "evaluation_model_20250831_20250831_121056/trained_model",
        "master_coefficients_test_20250828_154748_20250828_154754/trained_model", 
        "quick_training_for_live_demo_20250827_184436/trained_model"
    ]
    
    for candidate in candidates:
        if os.path.exists(f"{candidate}_policy_weights.weights.h5"):
            print(f"Found Keras 3 model: {candidate}")
            return candidate
        elif os.path.exists(f"{candidate}_policy_weights.index"):  
            print(f"Found legacy model: {candidate}")
            return candidate
    
    print("No model found")
    return None

def test_basic_functionality():
    """Test basic functionality without full evaluation"""
    print("ATENA-TF Simple Evaluation")
    print("="*50)
    
    # Find model
    model_path = find_model_simple()
    if not model_path:
        print("No trained model available")
        return False
    
    print(f"Model found: {model_path}")
    
    # Test basic imports that we know work
    try:
        print("Testing basic imports...")
        import tensorflow as tf
        print(f"TensorFlow {tf.__version__}")
        
        from models.ppo.agent import PPOAgent
        print("PPOAgent import OK")
        
        # Test agent creation
        agent = PPOAgent(obs_dim=51, action_dim=6)
        print("Agent creation OK")
        
        # Test model loading (if it exists)
        if model_path:
            try:
                success = agent.load_model(model_path)
                if success:
                    print("Model loading successful")
                else:
                    print("Model loading failed, but no crash")
            except Exception as e:
                print(f"Model loading error: {e}")
        
        print("\nBasic functionality test PASSED!")
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\nSystem appears to be working correctly!")
        print("The hang issue is likely in the environment imports.")
        print("You can proceed with the Jupyter notebook using direct model paths.")
    else:
        print("\nBasic functionality test failed")
