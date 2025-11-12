#!/usr/bin/env python3
"""
BLEU Score Calculation Module
Master-exact BLEU calculation for ATENA evaluation
"""

import re
from collections import Counter
from typing import List, Dict, Any
import numpy as np

def extract_action_type(action_str: str) -> str:
    """Extract action type from action string"""
    # Remove brackets and extract type
    clean_action = action_str.replace('[', '').replace(']', '')
    
    # Split on underscore and take first part (action type)
    if '_' in clean_action:
        return clean_action.split('_')[0]
    else:
        return clean_action

def extract_action_and_attribute(action_str: str) -> str:
    """Extract full action + attribute string"""
    # Keep the original format but clean brackets
    return action_str.replace('[', '').replace(']', '')

def calculate_ngram_precision(candidate_tokens: List[str], reference_tokens_list: List[List[str]], n: int) -> float:
    """Calculate n-gram precision for BLEU"""
    
    # Get candidate n-grams
    candidate_ngrams = []
    for i in range(len(candidate_tokens) - n + 1):
        ngram = tuple(candidate_tokens[i:i+n])
        candidate_ngrams.append(ngram)
    
    if not candidate_ngrams:
        return 0.0
    
    # Count candidate n-grams
    candidate_ngram_counts = Counter(candidate_ngrams)
    
    # Find maximum reference counts for each n-gram
    max_ref_counts = Counter()
    for ref_tokens in reference_tokens_list:
        ref_ngrams = []
        for i in range(len(ref_tokens) - n + 1):
            ngram = tuple(ref_tokens[i:i+n])
            ref_ngrams.append(ngram)
        
        ref_ngram_counts = Counter(ref_ngrams)
        
        # Update maximum counts
        for ngram, count in ref_ngram_counts.items():
            max_ref_counts[ngram] = max(max_ref_counts[ngram], count)
    
    # Calculate clipped counts
    clipped_counts = 0
    total_counts = 0
    
    for ngram, count in candidate_ngram_counts.items():
        clipped_count = min(count, max_ref_counts.get(ngram, 0))
        clipped_counts += clipped_count
        total_counts += count
    
    if total_counts == 0:
        return 0.0
    
    return clipped_counts / total_counts

def calculate_brevity_penalty(candidate_length: int, reference_lengths: List[int]) -> float:
    """Calculate brevity penalty for BLEU"""
    
    # Find closest reference length
    closest_ref_length = min(reference_lengths, key=lambda x: abs(x - candidate_length))
    
    if candidate_length > closest_ref_length:
        return 1.0
    elif candidate_length == 0:
        return 0.0
    else:
        return np.exp(1 - closest_ref_length / candidate_length)

def calculate_sentence_bleu(candidate_tokens: List[str], reference_tokens_list: List[List[str]], 
                          weights: List[float] = [0.25, 0.25, 0.25, 0.25]) -> float:
    """Calculate sentence-level BLEU score"""
    
    if not candidate_tokens or not reference_tokens_list:
        return 0.0
    
    # Calculate n-gram precisions
    precisions = []
    for n in range(1, len(weights) + 1):
        precision = calculate_ngram_precision(candidate_tokens, reference_tokens_list, n)
        precisions.append(precision)
    
    # Skip if any precision is 0 (geometric mean would be 0)
    if any(p == 0 for p in precisions):
        return 0.0
    
    # Calculate geometric mean of precisions
    log_precisions = [w * np.log(p) for w, p in zip(weights, precisions) if p > 0]
    if not log_precisions:
        return 0.0
    
    geometric_mean = np.exp(sum(log_precisions))
    
    # Calculate brevity penalty
    candidate_length = len(candidate_tokens)
    reference_lengths = [len(ref) for ref in reference_tokens_list]
    brevity_penalty = calculate_brevity_penalty(candidate_length, reference_lengths)
    
    # Final BLEU score
    bleu_score = brevity_penalty * geometric_mean
    
    return bleu_score

def calculate_corpus_bleu(candidate_sentences: List[List[str]], 
                         reference_sentences: List[List[List[str]]],
                         weights: List[float] = [0.25, 0.25, 0.25, 0.25]) -> float:
    """Calculate corpus-level BLEU score"""
    
    if len(candidate_sentences) != len(reference_sentences):
        raise ValueError("Number of candidate and reference sentences must match")
    
    # Accumulate counts across all sentences
    total_clipped_counts = [0] * len(weights)
    total_candidate_counts = [0] * len(weights)
    total_candidate_length = 0
    total_reference_length = 0
    
    for candidate_tokens, reference_tokens_list in zip(candidate_sentences, reference_sentences):
        if not candidate_tokens or not reference_tokens_list:
            continue
        
        # Accumulate n-gram counts
        for n in range(1, len(weights) + 1):
            # Candidate n-grams
            candidate_ngrams = []
            for i in range(len(candidate_tokens) - n + 1):
                ngram = tuple(candidate_tokens[i:i+n])
                candidate_ngrams.append(ngram)
            
            candidate_ngram_counts = Counter(candidate_ngrams)
            
            # Maximum reference counts
            max_ref_counts = Counter()
            for ref_tokens in reference_tokens_list:
                ref_ngrams = []
                for i in range(len(ref_tokens) - n + 1):
                    ngram = tuple(ref_tokens[i:i+n])
                    ref_ngrams.append(ngram)
                
                ref_ngram_counts = Counter(ref_ngrams)
                for ngram, count in ref_ngram_counts.items():
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], count)
            
            # Clipped counts
            clipped_counts = sum(min(count, max_ref_counts.get(ngram, 0)) 
                               for ngram, count in candidate_ngram_counts.items())
            
            total_clipped_counts[n-1] += clipped_counts
            total_candidate_counts[n-1] += len(candidate_ngrams)
        
        # Length statistics
        total_candidate_length += len(candidate_tokens)
        reference_lengths = [len(ref) for ref in reference_tokens_list]
        closest_ref_length = min(reference_lengths, key=lambda x: abs(x - len(candidate_tokens)))
        total_reference_length += closest_ref_length
    
    # Calculate corpus-level precisions
    precisions = []
    for clipped, total in zip(total_clipped_counts, total_candidate_counts):
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped / total)
    
    # Skip if any precision is 0
    if any(p == 0 for p in precisions):
        return 0.0
    
    # Geometric mean
    log_precisions = [w * np.log(p) for w, p in zip(weights, precisions) if p > 0]
    if not log_precisions:
        return 0.0
    
    geometric_mean = np.exp(sum(log_precisions))
    
    # Brevity penalty
    if total_candidate_length > total_reference_length:
        brevity_penalty = 1.0
    elif total_candidate_length == 0:
        brevity_penalty = 0.0
    else:
        brevity_penalty = np.exp(1 - total_reference_length / total_candidate_length)
    
    return brevity_penalty * geometric_mean

def calculate_bleu_score(candidate_actions: List[str], 
                        reference_actions: List[List[str]], 
                        mode: str = 'action_type') -> float:
    """
    Calculate BLEU score for ATENA actions
    
    Args:
        candidate_actions: List of agent action strings
        reference_actions: List of reference action lists (can be multiple refs)
        mode: 'action_type' for type-only, 'full' for type+attribute
    
    Returns:
        BLEU score (0.0 to 1.0)
    """
    
    if not candidate_actions or not reference_actions:
        return 0.0
    
    # Process candidate actions
    if mode == 'action_type':
        candidate_tokens = [extract_action_type(action) for action in candidate_actions]
    else:  # 'full' mode
        candidate_tokens = [extract_action_and_attribute(action) for action in candidate_actions]
    
    # Process reference actions
    reference_token_lists = []
    for ref_list in reference_actions:
        if mode == 'action_type':
            ref_tokens = [extract_action_type(action) for action in ref_list]
        else:  # 'full' mode
            ref_tokens = [extract_action_and_attribute(action) for action in ref_list]
        reference_token_lists.append(ref_tokens)
    
    # Calculate sentence-level BLEU
    bleu_score = calculate_sentence_bleu(candidate_tokens, reference_token_lists)
    
    return bleu_score

def batch_calculate_bleu_scores(candidates_list: List[List[str]], 
                               references_list: List[List[List[str]]],
                               mode: str = 'action_type') -> Dict[str, float]:
    """Calculate BLEU scores for multiple candidate-reference pairs"""
    
    individual_scores = []
    
    for candidate_actions, reference_actions in zip(candidates_list, references_list):
        score = calculate_bleu_score(candidate_actions, reference_actions, mode)
        individual_scores.append(score)
    
    # Prepare data for corpus BLEU
    candidate_sentences = []
    reference_sentences = []
    
    for candidate_actions, reference_actions in zip(candidates_list, references_list):
        # Process candidate
        if mode == 'action_type':
            candidate_tokens = [extract_action_type(action) for action in candidate_actions]
        else:
            candidate_tokens = [extract_action_and_attribute(action) for action in candidate_actions]
        
        # Process references
        reference_token_lists = []
        for ref_list in reference_actions:
            if mode == 'action_type':
                ref_tokens = [extract_action_type(action) for action in ref_list]
            else:
                ref_tokens = [extract_action_and_attribute(action) for action in ref_list]
            reference_token_lists.append(ref_tokens)
        
        candidate_sentences.append(candidate_tokens)
        reference_sentences.append(reference_token_lists)
    
    # Calculate corpus BLEU
    corpus_bleu = calculate_corpus_bleu(candidate_sentences, reference_sentences)
    
    return {
        'individual_scores': individual_scores,
        'average_sentence_bleu': np.mean(individual_scores) if individual_scores else 0.0,
        'corpus_bleu': corpus_bleu,
        'num_sentences': len(individual_scores)
    }
