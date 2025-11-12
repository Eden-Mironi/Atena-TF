#!/usr/bin/env python3
"""
Test script to verify gradient flow through ParametricSoftmaxDistribution
"""
import tensorflow as tf
from models.ppo.networks import ParametricSoftmaxDistribution
import numpy as np

def test_gradient_flow():
    print('Testing gradient flow through ParametricSoftmaxDistribution...')
    
    # Create test data
    batch_size = 4
    logits = tf.Variable(tf.random.normal((batch_size, 53)), trainable=True)
    segments = ((), (12, 3, 26), (12,))
    segment_sizes = [1, 936, 12]

    # Test sampling and log_prob with gradient tape
    with tf.GradientTape() as tape:
        tape.watch(logits)
        dist = ParametricSoftmaxDistribution(logits, segments, segment_sizes)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        loss = tf.reduce_mean(log_probs)

    # Check gradients
    gradients = tape.gradient(loss, logits)
    if gradients is not None:
        print('SUCCESS: Gradients flow correctly!')
        print(f'   Gradient shape: {gradients.shape}')
        print(f'   Gradient stats: mean={tf.reduce_mean(tf.abs(gradients)):.6f}')
        return True
    else:
        print('FAILURE: Gradients are still None!')
        return False

if __name__ == "__main__":
    success = test_gradient_flow()
    exit(0 if success else 1)
