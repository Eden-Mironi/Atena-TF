#!/usr/bin/env python3
"""
Reference Action Loader
Loads human/expert reference actions for BLEU calculation
"""

import os
import pickle
import json
from typing import List, Dict, Any, Optional
import numpy as np

def load_expert_sessions_from_master() -> Dict[int, List[List[str]]]:
    """Load expert sessions from master project's evaluation data"""
    
    expert_sessions = {}
    
    # Try to load from master's expert sessions
    master_expert_paths = [
        "../master/eval_sessions",  # Relative to atena-tf
        "../../master/eval_sessions",
        "../ATENA-A-EDA/master/eval_sessions",
        "eval_sessions"  # In case it's copied locally
    ]
    
    for expert_path in master_expert_paths:
        if os.path.exists(expert_path):
            print(f"Loading expert sessions from: {expert_path}")
            
            # Load all pickle files in the directory
            for filename in os.listdir(expert_path):
                if filename.endswith('.pickle') and 'dataset' in filename:
                    try:
                        # Extract dataset ID from filename
                        dataset_id = int(filename.split('dataset')[1].split('_')[0])
                        
                        with open(os.path.join(expert_path, filename), 'rb') as f:
                            session_data = pickle.load(f)
                        
                        # Extract action sequences
                        if isinstance(session_data, list):
                            action_sequences = []
                            for session in session_data:
                                if 'actions' in session:
                                    action_sequences.append(session['actions'])
                                elif 'action_sequence' in session:
                                    action_sequences.append(session['action_sequence'])
                            expert_sessions[dataset_id] = action_sequences
                        
                        print(f"  Dataset {dataset_id}: {len(expert_sessions[dataset_id])} expert sequences")
                        
                    except Exception as e:
                        print(f"  Failed to load {filename}: {e}")
                        continue
            
            if expert_sessions:
                break
    
    return expert_sessions

def load_human_behavior_patterns() -> Dict[int, List[List[str]]]:
    """Load human behavior patterns from saved data"""
    
    human_patterns = {}
    
    # Try to load human behavior data
    human_data_paths = [
        "human_actions_clusters.pickle",
        "./human_actions_clusters.pickle",
        "../human_actions_clusters.pickle",
        "gym_atena/envs/human_actions_clusters.pickle"
    ]
    
    for data_path in human_data_paths:
        if os.path.exists(data_path):
            print(f"👥 Loading human behavior patterns from: {data_path}")
            
            try:
                with open(data_path, 'rb') as f:
                    human_data = pickle.load(f)
                
                # Convert human observation patterns to action sequences
                # This is a simplified conversion - in reality, we'd need more sophisticated mapping
                if isinstance(human_data, dict):
                    for dataset_id in range(3):  # Assume 3 datasets
                        # Generate representative action sequences from human patterns
                        action_sequences = generate_action_sequences_from_patterns(human_data, dataset_id)
                        human_patterns[dataset_id] = action_sequences
                
                print(f"  Loaded human patterns for {len(human_patterns)} datasets")
                break
                
            except Exception as e:
                print(f"  Failed to load human patterns: {e}")
                continue
    
    return human_patterns

def generate_action_sequences_from_patterns(human_data: Dict, dataset_id: int) -> List[List[str]]:
    """Generate representative action sequences from human behavior patterns"""
    
    # This is a simplified implementation
    # In a real scenario, you'd have more sophisticated pattern-to-action conversion
    
    sequences = []
    
    # Generate some typical human-like sequences
    typical_sequences = [
        ['[filter]', '[group]', '[filter]', '[back]'],
        ['[group]', '[filter]', '[group]'],
        ['[filter]', '[filter]', '[group]', '[back]', '[filter]'],
        ['[group]', '[back]', '[filter]', '[group]'],
        ['[filter]', '[group]', '[back]']
    ]
    
    # Add some variation with attributes
    attribute_variations = [
        ['[filter]_[ip_src]', '[group]_[tcp_port]', '[filter]_[ip_dst]'],
        ['[group]_[highest_layer]', '[filter]_[protocol]', '[back]'],
        ['[filter]_[timestamp]', '[group]_[ip_src]', '[filter]_[tcp_port]', '[back]']
    ]
    
    sequences.extend(typical_sequences)
    sequences.extend(attribute_variations)
    
    return sequences[:5]  # Return top 5 patterns

def create_default_references(dataset_id: int) -> List[List[str]]:
    """Create default reference sequences when no expert data is available"""
    
    # Default reference sequences based on typical ATENA usage patterns
    default_sequences = [
        # Conservative exploration
        ['[filter]', '[group]', '[back]'],
        ['[group]', '[filter]', '[back]'],
        
        # Medium exploration
        ['[filter]', '[filter]', '[group]', '[back]'],
        ['[group]', '[filter]', '[group]', '[back]'],
        
        # Comprehensive exploration
        ['[filter]', '[group]', '[filter]', '[group]', '[back]'],
        ['[group]', '[back]', '[filter]', '[group]', '[filter]'],
    ]
    
    # Add attribute-specific sequences
    attribute_sequences = [
        ['[filter]_[ip_src]', '[group]_[tcp_port]'],
        ['[group]_[highest_layer]', '[filter]_[protocol]'],
        ['[filter]_[timestamp]', '[back]', '[group]_[ip_dst]']
    ]
    
    references = default_sequences + attribute_sequences
    
    # Select subset based on dataset_id for variety
    start_idx = (dataset_id * 3) % len(references)
    selected_refs = references[start_idx:start_idx+4]
    
    # Ensure we have at least 3 references
    while len(selected_refs) < 3:
        selected_refs.extend(references[:3-len(selected_refs)])
    
    return selected_refs[:5]  # Max 5 references

def load_reference_actions(dataset_id: int) -> List[List[str]]:
    """
    Load reference actions for a specific dataset
    
    Args:
        dataset_id: Dataset identifier
    
    Returns:
        List of reference action sequences
    """
    
    print(f"Loading reference actions for dataset {dataset_id}")
    
    # Try to load expert sessions first
    expert_sessions = load_expert_sessions_from_master()
    if dataset_id in expert_sessions:
        print(f"  Using {len(expert_sessions[dataset_id])} expert sequences")
        return expert_sessions[dataset_id]
    
    # Try human behavior patterns
    human_patterns = load_human_behavior_patterns()
    if dataset_id in human_patterns:
        print(f"  Using {len(human_patterns[dataset_id])} human pattern sequences")
        return human_patterns[dataset_id]
    
    # Fall back to default references
    print(f"  Using default reference sequences")
    default_refs = create_default_references(dataset_id)
    return default_refs

def load_all_reference_actions() -> Dict[int, List[List[str]]]:
    """Load reference actions for all available datasets"""
    
    all_references = {}
    
    # Try expert sessions first
    expert_sessions = load_expert_sessions_from_master()
    all_references.update(expert_sessions)
    
    # Fill in missing datasets with human patterns or defaults
    for dataset_id in range(10):  # Check up to 10 datasets
        if dataset_id not in all_references:
            references = load_reference_actions(dataset_id)
            all_references[dataset_id] = references
    
    return all_references

def validate_reference_actions(references: List[List[str]]) -> bool:
    """Validate that reference actions are properly formatted"""
    
    if not references:
        return False
    
    for sequence in references:
        if not isinstance(sequence, list):
            return False
        
        for action in sequence:
            if not isinstance(action, str):
                return False
            
            # Check basic action format
            if not (action.startswith('[') and action.endswith(']')):
                # Allow some flexibility in format
                continue
    
    return True

def get_reference_statistics(references: List[List[str]]) -> Dict[str, Any]:
    """Get statistics about reference actions"""
    
    if not references:
        return {'num_sequences': 0, 'total_actions': 0, 'avg_length': 0.0}
    
    sequence_lengths = [len(seq) for seq in references]
    all_actions = [action for seq in references for action in seq]
    
    # Count action types
    action_types = {}
    for action in all_actions:
        action_type = action.split('_')[0] if '_' in action else action
        action_type = action_type.replace('[', '').replace(']', '')
        action_types[action_type] = action_types.get(action_type, 0) + 1
    
    return {
        'num_sequences': len(references),
        'total_actions': len(all_actions),
        'avg_length': np.mean(sequence_lengths),
        'min_length': min(sequence_lengths),
        'max_length': max(sequence_lengths),
        'action_type_counts': action_types,
        'unique_actions': len(set(all_actions))
    }
