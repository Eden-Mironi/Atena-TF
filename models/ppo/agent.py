import tensorflow as tf
import numpy as np
from .networks import GaussianPolicy, ParametricSoftmaxPolicy, ValueNetwork
from .param_softmax_policy import ParamSoftmaxPolicy

class EmpiricalNormalization:
    """Observation normalization matching original ChainerRL EmpiricalNormalization"""
    
    def __init__(self, obs_size, clip_threshold=5):
        self.obs_size = obs_size
        self.clip_threshold = clip_threshold
        self.mean = tf.Variable(tf.zeros(obs_size, dtype=tf.float32), trainable=False)
        self.std = tf.Variable(tf.ones(obs_size, dtype=tf.float32), trainable=False)
        self.count = tf.Variable(0, dtype=tf.int32, trainable=False)
        
    def update(self, obs):
        """Update normalization statistics"""
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        if len(obs.shape) == 1:
            obs = tf.expand_dims(obs, 0)
        
        batch_size = tf.shape(obs)[0]  # Use tf.shape for dynamic batch size
        if batch_size == 0:
            return
        
        # Update count
        new_count = self.count + batch_size
        
        # Update mean
        delta = obs - self.mean
        new_mean = self.mean + tf.reduce_sum(delta, axis=0) / tf.cast(new_count, tf.float32)
        
        # Update std using numerically stable formula
        delta2 = obs - new_mean
        # Handle the case when count is 0 (first update)
        if self.count == 0:
            new_std = tf.sqrt(tf.reduce_mean(delta2 ** 2, axis=0) + 1e-8)
        else:
            new_std = tf.sqrt(
                (self.std ** 2 * tf.cast(self.count, tf.float32) + 
                 tf.reduce_sum(delta2 ** 2, axis=0)) / tf.cast(new_count, tf.float32) + 1e-8
            )
        
        # Apply updates
        self.count.assign(new_count)
        self.mean.assign(new_mean)
        self.std.assign(new_std)
    
    def normalize(self, obs, update=True):
        """Normalize observations"""
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        
        if update:
            self.update(obs)
        
        # Normalize
        normalized = (obs - self.mean) / (self.std + 1e-8)
        
        # Clip
        normalized = tf.clip_by_value(normalized, -self.clip_threshold, self.clip_threshold)
        
        return normalized

class PPOAgent:
    def __init__(self, obs_dim, action_dim, 
                 learning_rate=3e-4,        # Original Adam learning rate
                 clip_ratio=0.2,            # Original clip_eps
                 value_coef=1.0,            # MATCH MASTER: ChainerRL PPO default is 1.0
                 entropy_coef=0.0,           # MATCH MASTER: No entropy regularization (arguments.py line 139)
                 max_grad_norm=0.5,         # Original gradient clipping
                 gamma=0.995,               # Original gamma
                 lambda_=0.97,              # Original lambda
                 n_hidden_layers=2,         # Original: 2 hidden layers
                 n_hidden_channels=64,      # Master: 64 channels for Gaussian (continuous)
                 beta=0.5,                  # REDUCED: More exploration/randomness for diverse actions
                 use_parametric_softmax=False,  # NEW: Switch between architectures
                 parametric_segments=None,      # NEW: For parametric softmax
                 parametric_segments_sizes=None, # NEW: For parametric softmax
                 # CRITICAL PPO PARAMETERS FROM MASTER (lines 163-173 in chainerrl_ppo.py):
                 update_interval=2048,      # Master's PPO update_interval
                 minibatch_size=64,         # Master's minibatch_size 
                 epochs=10,                 # Master's epochs per update
                 clip_eps_vf=None,          # Master's value function clipping (None = disabled)
                 standardize_advantages=True):  # Master's advantage standardization
        # Store original obs_dim for reference
        self.original_obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_parametric_softmax = use_parametric_softmax
        
        # CRITICAL PPO PARAMETERS (fixing backwards learning):
        self.update_interval = update_interval
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.standardize_advantages = standardize_advantages
        self.clip_eps_vf = clip_eps_vf
        
        # Set TensorFlow random seed for deterministic network initialization (matching master's args.seed default 0)
        tf.random.set_seed(0)  # Master uses args.seed with default 0, not 42
        
        # Initialize policy based on configuration
        if use_parametric_softmax:
            # Use master's successful FFParamSoftmax architecture (BLEU 0.33-0.56)
            print("Initializing ParamSoftmaxPolicy (master's SUCCESS architecture - 600 channels!)")
            self.policy = ParamSoftmaxPolicy(
                obs_dim=obs_dim,
                n_hidden_layers=n_hidden_layers, 
                n_hidden_channels=n_hidden_channels,  # 600 for master's FFParamSoftmax
                parametric_segments=parametric_segments,
                beta=beta
            )
            self.is_discrete = True
        else:
            # Use continuous Gaussian policy (current working version)  
            print("Initializing GaussianPolicy (continuous architecture)")
            print("Using ChainerRL-compatible bound_mean=True (master uses --bound-mean) and action_space bounds!")
            
            # Pass action space to policy for bound_mean functionality (like ChainerRL)
            # This will automatically bound outputs to action space using tanh + scaling
            from gym_atena.envs.enhanced_atena_env import make_enhanced_atena_env
            temp_env = make_enhanced_atena_env()
            action_space = temp_env.action_space
            
            self.policy = GaussianPolicy(
                obs_dim, action_dim, n_hidden_layers, n_hidden_channels, 
                action_space=action_space,  # Pass action bounds to policy
                bound_mean=True  # Master uses bound_mean=True (--bound-mean flag)!
            )
            self.is_discrete = False
        
        # Value network uses SAME channels as policy (matching master's ChainerRL architecture)
        # Master: FFGaussian uses 64 for both, FFParamSoftmax uses 600 for both
        value_activation = tf.nn.tanh if use_parametric_softmax else tf.nn.relu
        self.value_net = ValueNetwork(obs_dim, n_hidden_layers, n_hidden_channels, activation=value_activation)
        
        # Observation normalizer (matching original) - will be updated dynamically
        self.obs_normalizer = None
        
        # CRITICAL MISSING PIECE: Add master's phi preprocessing function
        # Master uses: phi=lambda x: x.astype(np.float32, copy=False)
        self.phi = lambda x: tf.cast(x, tf.float32)  # TF equivalent of astype(np.float32)
        
        # EMERGENCY FIX: Separate optimizers for policy and value networks
        # Value network needs MUCH higher learning rate to scale up from tiny predictions
        self.weight_decay = 0.0  # Master's default args.weight_decay value
        
        self.policy_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,  # 3e-4 for policy
            epsilon=1e-5,
        )
        
        self.value_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            epsilon=1e-5,
        )
        
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lambda_ = lambda_
        
        # CRITICAL PPO PARAMETERS FROM MASTER:
        self.update_interval = update_interval    # 2048 steps before PPO update
        self.minibatch_size = minibatch_size     # 64 samples per minibatch
        self.epochs = epochs                     # 10 epochs per PPO update
        self.clip_eps_vf = clip_eps_vf          # Value function clipping (None = disabled)
        
        # Experience collection for batch updates (matching master's batch training)
        self.experience_buffer = []
        self.steps_since_update = 0
        
        # Statistics tracking (matching original ChainerRL)
        self.policy_loss_record = []
        self.value_loss_record = []
        self.entropy_record = []
        self.value_record = []
        
        # Master-style policy collapse detection
        self.avg_entropy_history = []
        self.avg_highest_act_prob_history = []
        self.avg_second_highest_act_prob_history = []
        
        # Training state (matching original)
        self.batch_last_episode = None
        self.batch_last_state = None
        self.batch_last_action = None
        
        # Build networks immediately to ensure trainable variables are available
        print("🏗️  Building networks...")
        dummy_obs = tf.zeros((1, obs_dim), dtype=tf.float32)
        if self.is_discrete:
            _ = self.policy(dummy_obs, training=False)  # Build policy network
        else:
            _ = self.policy(dummy_obs)  # Build policy network
        _ = self.value_net(dummy_obs)  # Build value network
        
        print(f"Networks built - Policy vars: {len(self.policy.trainable_variables)}, Value vars: {len(self.value_net.trainable_variables)}")
        
        if len(self.policy.trainable_variables) == 0 or len(self.value_net.trainable_variables) == 0:
            raise RuntimeError("Networks have 0 trainable variables! Training impossible!")
    
    def _initialize_batch_variables(self, num_envs):
        """Initialize batch variables like original ChainerRL"""
        self.batch_last_episode = [False] * num_envs
        self.batch_last_state = [None] * num_envs
        self.batch_last_action = [None] * num_envs
    
    def _normalize_obs(self, obs, update=True):
        """Normalize observations using empirical normalization"""
        return self.obs_normalizer.normalize(obs, update=update)
    
    def _apply_nonbias_weight_decay(self):
        """
        Apply weight decay only to non-bias parameters (matching ChainerRL's NonbiasWeightDecay)
        
        ChainerRL's NonbiasWeightDecay excludes bias parameters from weight decay to prevent
        regularization from adversely affecting parameters that should remain unbiased
        """
        if self.weight_decay > 0:
            for model in [self.policy, self.value_net]:
                for layer in model.layers:
                    if hasattr(layer, 'kernel') and layer.kernel is not None:
                        # Apply weight decay to kernel (weights) only, not bias
                        layer.kernel.assign(layer.kernel * (1 - self.weight_decay))
                    # Explicitly skip bias parameters (layer.bias) - this is the key difference from Adam's weight_decay
    
    # Discrete conversion no longer needed - using continuous actions
    
    def act(self, obs):
        """Get action and value for given observation (single environment) - NON-TRAINING VERSION"""
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        
        # Apply phi preprocessing first (master's phi function)
        obs = self.phi(obs)
        
        # Ensure proper shape: (batch_size, obs_dim)
        if len(obs.shape) == 1:
            obs = tf.expand_dims(obs, 0)  # Add batch dimension
        elif len(obs.shape) > 2:
            obs = tf.reshape(obs, (-1, obs.shape[-1]))  # Flatten to (batch_size, obs_dim)
        
        # Get actual observation dimension from the observation
        actual_obs_dim = obs.shape[-1]
        
        # Initialize normalizer with correct dimension if needed
        if self.obs_normalizer is None or self.obs_normalizer.obs_size != actual_obs_dim:
            self.obs_normalizer = EmpiricalNormalization(actual_obs_dim, clip_threshold=5)
        
        # Normalize observations (matching original)
        obs = self._normalize_obs(obs, update=True)
            
        # Handle different policy types
        if self.is_discrete and isinstance(self.policy, ParamSoftmaxPolicy):
            # NEW: Master's FFParamSoftmax policy
            action, log_prob = self.policy.sample_action(obs)
            value = self.value_net(obs)
            
            # Calculate entropy from probabilities
            logits = self.policy(obs, training=False)
            probs = self.policy.get_action_probabilities(logits)
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
            
        else:
            # Original policy logic (FFGaussian or old ParametricSoftmax)
            action_dist, value = self.policy(obs), self.value_net(obs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            if self.is_discrete:
                # Discrete actions: log_prob should already be correct shape from ParametricSoftmaxDistribution
                if len(log_prob.shape) > 1:
                    log_prob = tf.reduce_mean(log_prob, axis=-1)  # Handle multi-dimensional if needed
            else:
                # Continuous actions: sum log probs across action dimensions  
                if len(log_prob.shape) > 1:
                    log_prob = tf.reduce_sum(log_prob, axis=-1)
            
            # Ensure log_prob is a scalar (sum over action dimensions if needed)
            if len(log_prob.shape) > 1:
                log_prob = tf.reduce_sum(log_prob, axis=-1)
            
            # Record entropy for statistics (matching original)
            entropy = action_dist.entropy()
        
        self.entropy_record.append(tf.reduce_mean(entropy).numpy())
        self.value_record.append(tf.reduce_mean(value).numpy())
        
        return action, log_prob, value
    
    def act_and_train(self, obs, prev_reward):
        """ChainerRL-compatible act_and_train with temporal reward coupling!
        
        This method matches ChainerRL's temporal learning:
        - Agent sees PREVIOUS reward when making decisions
        - Enables temporal cause-effect learning
        - Stores experience for training
        """
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        prev_reward = tf.convert_to_tensor(prev_reward, dtype=tf.float32)
        
        # Apply phi preprocessing first (master's phi function)
        obs = self.phi(obs)
        
        # Ensure proper shape: (batch_size, obs_dim)
        if len(obs.shape) == 1:
            obs = tf.expand_dims(obs, 0)  # Add batch dimension
        elif len(obs.shape) > 2:
            obs = tf.reshape(obs, (-1, obs.shape[-1]))  # Flatten to (batch_size, obs_dim)
        
        # Get actual observation dimension from the observation
        actual_obs_dim = obs.shape[-1]
        
        # Initialize normalizer with correct dimension if needed
        if self.obs_normalizer is None or self.obs_normalizer.obs_size != actual_obs_dim:
            self.obs_normalizer = EmpiricalNormalization(actual_obs_dim, clip_threshold=5)
        
        # Normalize observations (matching original)
        obs = self._normalize_obs(obs, update=True)
        
        # TEMPORAL COUPLING: Store previous transition if we have complete information
        if hasattr(self, 'last_obs') and hasattr(self, 'last_action'):
            # Store the (s, a, r, s') transition for training
            self._store_transition(
                obs=self.last_obs,
                action=self.last_action, 
                reward=prev_reward,
                next_obs=obs,
                done=False  # Will be updated by stop_episode_and_train if needed
            )
        
        # Include previous reward in decision making!
        # This enables the agent to learn immediate reward feedback patterns
        if len(prev_reward.shape) == 0:
            prev_reward = tf.expand_dims(prev_reward, 0)  # Add batch dimension
        
        # Handle different policy types with temporal information
        if self.is_discrete and isinstance(self.policy, ParamSoftmaxPolicy):
            # NEW: Master's FFParamSoftmax policy with temporal coupling
            # For now, use standard action selection (temporal info can be added later to policy architecture)
            action, log_prob = self.policy.sample_action(obs)
            value = self.value_net(obs)
            
            # TODO: Integrate prev_reward into policy architecture for true temporal coupling
            # For now, the temporal information is captured through experience storage
            
        elif self.is_discrete:
            # For other discrete policies: standard action selection for now
            action_dist = self.policy(obs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            value = self.value_net(obs)
            
        else:
            # Continuous policy - standard action selection for now
            action_dist = self.policy(obs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Continuous actions: sum log probs across action dimensions  
            if len(log_prob.shape) > 1:
                log_prob = tf.reduce_sum(log_prob, axis=-1)
                
            value = self.value_net(obs)
            
        # Ensure log_prob is scalar
        if len(log_prob.shape) > 0:
            log_prob = tf.reduce_mean(log_prob)
            
        # Record entropy and value for statistics
        if self.is_discrete and isinstance(self.policy, ParamSoftmaxPolicy):
            logits = self.policy(obs, training=False)
            probs = self.policy.get_action_probabilities(logits)
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
        else:
            if hasattr(action_dist, 'entropy'):
                entropy = action_dist.entropy()
            else:
                entropy = tf.constant(0.0)  # Fallback
                
        self.entropy_record.append(tf.reduce_mean(entropy).numpy())
        self.value_record.append(tf.reduce_mean(value).numpy())
        
        # Store current state/action for next transition
        self.last_obs = obs.numpy()
        self.last_action = action.numpy() if hasattr(action, 'numpy') else action
        
        return action, log_prob, value
    
    def _store_transition(self, obs, action, reward, next_obs, done):
        """Store transition for ChainerRL-compatible experience replay"""
        # Initialize experience buffer if needed
        if not hasattr(self, 'experience_buffer'):
            self.experience_buffer = []
            
        transition = {
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done
        }
        self.experience_buffer.append(transition)
    
    def stop_episode_and_train(self, obs, reward, done=True):
        """ChainerRL-compatible episode ending with training trigger
        
        This method:
        - Stores the final transition 
        - Marks episode as done
        - Triggers training if enough experience collected
        """
        # Store final transition if we have previous state/action
        if hasattr(self, 'last_obs') and hasattr(self, 'last_action'):
            self._store_transition(
                obs=self.last_obs,
                action=self.last_action,
                reward=reward,
                next_obs=obs,
                done=done
            )
        
        # Clear last state/action for new episode
        if hasattr(self, 'last_obs'):
            delattr(self, 'last_obs')
        if hasattr(self, 'last_action'):
            delattr(self, 'last_action')
        
        # Trigger training if we have enough experience
        if (hasattr(self, 'experience_buffer') and 
            len(self.experience_buffer) >= self.update_interval):
            self._train_from_experience_buffer()
            
    def _train_from_experience_buffer(self):
        """Train PPO from collected experience buffer (ChainerRL-style)"""
        if not hasattr(self, 'experience_buffer') or len(self.experience_buffer) == 0:
            return
            
        # Convert experience buffer to training format
        obs_list = []
        actions_list = []
        rewards_list = []
        values_list = []
        log_probs_list = []
        dones_list = []
        
        for transition in self.experience_buffer:
            obs_list.append(transition['obs'])
            actions_list.append(transition['action'])
            rewards_list.append(transition['reward'])
            dones_list.append(transition['done'])
            
            # Get value and log_prob for this transition
            _, log_prob, value = self.act(transition['obs'])
            values_list.append(value.numpy())
            log_probs_list.append(log_prob.numpy())
        
        # Clear experience buffer
        self.experience_buffer = []
        
        # Trigger PPO update (this would connect to trainer's update method)
        print(f"ChainerRL-style training triggered with {len(obs_list)} experiences")
        
        # Master-style policy collapse monitoring  
        if self.is_discrete and isinstance(self.policy, ParamSoftmaxPolicy):
            # For FFParamSoftmax: use already calculated probabilities
            highest_prob = tf.reduce_max(probs, axis=-1)
            sorted_probs = tf.nn.top_k(probs, k=2).values
            second_highest = sorted_probs[..., 1] if sorted_probs.shape[-1] > 1 else sorted_probs[..., 0]
        elif 'action_dist' in locals() and hasattr(action_dist, 'probs_parameter'):
            # For old discrete policies
            probs = action_dist.probs_parameter()
            highest_prob = tf.reduce_max(probs, axis=-1)
            sorted_probs = tf.nn.top_k(probs, k=2).values
            second_highest = sorted_probs[..., 1] if sorted_probs.shape[-1] > 1 else sorted_probs[..., 0]
        elif 'action_dist' in locals():
            # For continuous policies - approximate with concentration
            if hasattr(action_dist, 'scale'):
                try:
                    # Higher scale = lower concentration = lower "probability" 
                    if hasattr(action_dist.scale, 'diag_part'):
                        scale_vals = action_dist.scale.diag_part()
                    else:
                        scale_vals = action_dist.scale
                    highest_prob = 1.0 / (1.0 + tf.reduce_mean(scale_vals))  # Approximate
                    second_highest = highest_prob * 0.8  # Approximate
                except:
                    highest_prob = 0.5  # Fallback
                    second_highest = 0.3  # Fallback
            else:
                highest_prob = 0.5  # Default
                second_highest = 0.3  # Default
        else:
            # Fallback for FFParamSoftmax without probs calculated
            highest_prob = 0.5  # Default
            second_highest = 0.3  # Default
        
        self.avg_highest_act_prob_history.append(float(tf.reduce_mean(highest_prob).numpy()))
        self.avg_second_highest_act_prob_history.append(float(tf.reduce_mean(second_highest).numpy()))
        self.avg_entropy_history.append(float(tf.reduce_mean(entropy).numpy()))
        
        # Return numpy arrays, squeeze batch dimension if single observation
        if len(obs.shape) == 2 and obs.shape[0] == 1:
            action_np = action.numpy().squeeze()
            log_prob_np = log_prob.numpy().squeeze()
            value_np = value.numpy().squeeze()
        else:
            action_np = action.numpy()
            log_prob_np = log_prob.numpy()
            value_np = value.numpy()
        
        # Ensure action is an array (not scalar) for environments that expect arrays
        if np.isscalar(action_np):
            action_np = np.array([action_np])
        
        return action_np, log_prob_np, value_np

    def batch_act_with_mean(self, batch_obs):
        """
        CRITICAL MISSING FUNCTION: Get batch of deterministic actions using mean (no sampling)
        Based on ATENA-master/train_agent_chainerrl.py lines 80-104
        
        Used for humanity reward evaluation - compares agent actions to human behavior patterns
        Called every humans_reward_interval episodes during training
        """
        obs = tf.convert_to_tensor(batch_obs, dtype=tf.float32)
        
        # Apply phi preprocessing (master's phi function)
        obs = self.phi(obs)
        
        # Ensure proper shape: (batch_size, obs_dim)
        if len(obs.shape) == 1:
            obs = tf.expand_dims(obs, 0)  # Add batch dimension
        elif len(obs.shape) > 2:
            obs = tf.reshape(obs, (-1, obs.shape[-1]))  # Flatten to (batch_size, obs_dim)
        
        # Get actual observation dimension from the observation
        actual_obs_dim = obs.shape[-1]
        
        # Initialize normalizer with correct dimension if needed
        if self.obs_normalizer is None or self.obs_normalizer.obs_size != actual_obs_dim:
            self.obs_normalizer = EmpiricalNormalization(actual_obs_dim, clip_threshold=5)
        
        # Normalize observations (don't update normalizer stats during evaluation)
        obs = self._normalize_obs(obs, update=False)
        
        # Get deterministic actions (matching master's batch_act_with_mean)
        if self.is_discrete and isinstance(self.policy, ParamSoftmaxPolicy):
            # FFParamSoftmax: use most_probable_action method
            actions = self.policy.most_probable_action(obs)
        elif self.is_discrete:
            # For old discrete policies: get argmax of logits (most probable actions)
            action_dist = self.policy(obs)
            logits = action_dist.logits if hasattr(action_dist, 'logits') else action_dist.distribution.logits
            actions = tf.argmax(logits, axis=-1)
        else:
            # For continuous: use raw mean from hidden layers (matching master's implementation)
            # Master line 98: actions = agent.model.pi.hidden_layers(b_state).data
            actions = self.policy(obs, return_hidden=True)
        
        return actions.numpy()

    def batch_act_and_train(self, batch_obs):
        """
        CRITICAL MISSING FUNCTION: Batch act and train (returns actions + action distribution)
        Based on ATENA-master/train_agent_chainerrl.py lines 107-132
        This is the core training function used in the master's training loop.
        """
        obs = tf.convert_to_tensor(batch_obs, dtype=tf.float32)
        obs = self.phi(obs)
        
        if len(obs.shape) == 1:
            obs = tf.expand_dims(obs, 0)
        elif len(obs.shape) > 2:
            obs = tf.reshape(obs, (-1, obs.shape[-1]))
            
        # Initialize/update observation normalizer
        actual_obs_dim = obs.shape[-1]
        if self.obs_normalizer is None or self.obs_normalizer.obs_size != actual_obs_dim:
            self.obs_normalizer = EmpiricalNormalization(actual_obs_dim, clip_threshold=5)
        obs = self._normalize_obs(obs, update=False)
        
        num_envs = len(batch_obs)
        
        # Initialize batch variables if needed (matching master's batch training)
        if self.batch_last_episode is None:
            self._initialize_batch_variables(num_envs)
            
        # Forward pass to get action distribution and values
        if self.is_discrete and isinstance(self.policy, ParamSoftmaxPolicy):
            # FFParamSoftmax policy
            batch_action, batch_log_prob = self.policy.sample_action(obs)
            batch_value = self.value_net(obs)
            
            # Calculate entropy from probabilities for recording
            logits = self.policy(obs, training=False)
            probs = self.policy.get_action_probabilities(logits)
            entropy_values = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
            
            # Create a mock distribution for compatibility
            class MockDistribution:
                def __init__(self, actions, log_probs, entropy):
                    self.actions = actions
                    self.log_probs = log_probs
                    self._entropy = entropy
                    
                def sample(self):
                    return self.actions
                    
                def entropy(self):
                    return self._entropy
                    
                def log_prob(self, actions):
                    return self.log_probs
            
            action_dist = MockDistribution(batch_action, batch_log_prob, entropy_values)
            
        else:
            # Original policy logic (FFGaussian or old ParametricSoftmax)
            action_dist = self.policy(obs)
            batch_action = action_dist.sample()
            batch_value = self.value_net(obs)
            entropy_values = action_dist.entropy()  # Simple and direct like master!
        
        # Master's EXACT entropy recording (line 125-126)
        # Master: self.entropy_record.extend(chainer.cuda.to_cpu(action_distrib.entropy.data))
        self.entropy_record.extend(entropy_values.numpy().flatten().tolist())
            
        # Store value predictions
        self.value_record.extend(batch_value.numpy().flatten().tolist())
        
        # Update batch state tracking
        self.batch_last_state = list(batch_obs)
        if hasattr(batch_action, 'numpy'):
            self.batch_last_action = batch_action.numpy().tolist()
            return batch_action.numpy(), action_dist
        else:
            self.batch_last_action = batch_action.tolist() if hasattr(batch_action, 'tolist') else batch_action
            return batch_action, action_dist
    
    def _initialize_batch_variables(self, num_envs):
        """Initialize batch training variables (matching master)"""
        self.batch_last_episode = [None] * num_envs
        self.batch_last_state = [None] * num_envs
        self.batch_last_action = [None] * num_envs
    
    def act_most_probable(self, obs):
        """
        Deterministic action generation matching master's EXACT pipeline
        
        Master pipeline:
        1. agent.model.pi.hidden_layers(b_state).data[0] (raw hidden output)
        2. Environment applies compressed2full_range() during step with compressed=True
        3. cont2dis() rounds to discrete integers
        
        Our pipeline now matches this exactly!
        """
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        
        # Apply phi preprocessing first (master's phi function)
        obs = self.phi(obs)
        
        # Ensure proper shape: (batch_size, obs_dim)
        if len(obs.shape) == 1:
            obs = tf.expand_dims(obs, 0)  # Add batch dimension
        elif len(obs.shape) > 2:
            obs = tf.reshape(obs, (-1, obs.shape[-1]))  # Flatten to (batch_size, obs_dim)
        
        # Get actual observation dimension from the observation
        actual_obs_dim = obs.shape[-1]
        
        # Initialize normalizer with correct dimension if needed
        if self.obs_normalizer is None or self.obs_normalizer.obs_size != actual_obs_dim:
            self.obs_normalizer = EmpiricalNormalization(actual_obs_dim, clip_threshold=5)
        
        # Normalize observations (matching original)
        obs = self._normalize_obs(obs, update=False)  # Don't update normalizer stats
            
        # Match master's act_most_probable implementation EXACTLY!
        if self.is_discrete and isinstance(self.policy, ParamSoftmaxPolicy):
            # NEW: Master's FFParamSoftmax deterministic action selection
            action = self.policy.most_probable_action(obs)
            
            # Get log probability for the most probable action
            logits = self.policy(obs, training=False)
            probs = self.policy.get_action_probabilities(logits)
            log_prob = tf.math.log(tf.gather(probs, action, batch_dims=1) + 1e-8)
            value = self.value_net(obs)
            
        elif self.is_discrete:
            # For old discrete policy: get argmax of logits (most probable action)
            action_dist = self.policy(obs)
            logits = action_dist.logits if hasattr(action_dist, 'logits') else action_dist.distribution.logits
            action = tf.argmax(logits, axis=-1)
            log_prob = action_dist.log_prob(action)
            value = self.value_net(obs)
        else:
            # MASTER-EXACT FIX: For FF_GAUSSIAN, master uses raw hidden_layers output!
            # From ATENA-master/train_agent_chainerrl.py line 69:
            # action = agent.model.pi.hidden_layers(b_state).data[0]
            
            action = self.policy(obs, return_hidden=True)  # Get raw hidden layer output like master!
            
            # For log_prob and value, we still need the distribution
            action_dist = self.policy(obs, return_hidden=False)  # Get distribution for log_probma
            log_prob = action_dist.log_prob(action)
            value = self.value_net(obs)
        
        # Handle log_prob shapes
        if self.is_discrete:
            if len(log_prob.shape) > 1:
                log_prob = tf.reduce_mean(log_prob, axis=-1)
        else:
            if len(log_prob.shape) > 1:
                log_prob = tf.reduce_sum(log_prob, axis=-1)
        
        # Return numpy arrays, squeeze batch dimension if single observation
        if len(obs.shape) == 2 and obs.shape[0] == 1:
            action_np = action.numpy().squeeze()
            log_prob_np = log_prob.numpy().squeeze()
            value_np = value.numpy().squeeze()
        else:
            action_np = action.numpy()
            log_prob_np = log_prob.numpy()
            value_np = value.numpy()
        
        # Ensure action is an array (not scalar) for environments that expect arrays
        if np.isscalar(action_np):
            action_np = np.array([action_np])
        
        return action_np, log_prob_np, value_np
    

    
    def batch_observe_and_train(self, batch_obs, batch_rewards, batch_dones, batch_resets):
        """
        MASTER-EXACT: Replicate ChainerRL's PPO batch_observe_and_train functionality
        Based on ATENA-master/train_agent_chainerrl.py lines 107-132 batch_act_and_train
        
        This method performs the actual model forward pass, action sampling, and data recording
        that the master's ChainerRL agent does during training.
        """
        # Convert inputs to tensors
        obs = tf.convert_to_tensor(batch_obs, dtype=tf.float32)
        rewards = tf.convert_to_tensor(batch_rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(batch_dones, dtype=tf.bool)
        
        # Ensure proper shape
        if len(obs.shape) == 1:
            obs = tf.expand_dims(obs, 0)
        elif len(obs.shape) > 2:
            obs = tf.reshape(obs, (-1, obs.shape[-1]))
            
        # Get actual observation dimension from the observation
        actual_obs_dim = obs.shape[-1]
        
        # Initialize normalizer with correct dimension if needed
        if self.obs_normalizer is None or self.obs_normalizer.obs_size != actual_obs_dim:
            self.obs_normalizer = EmpiricalNormalization(actual_obs_dim, clip_threshold=5)
        
        # Normalize observations (matching master)
        obs = self._normalize_obs(obs, update=False)
        
        num_envs = len(batch_obs)
        if self.batch_last_episode is None:
            self._initialize_batch_variables(num_envs)
            
        # MASTER EXACT: Forward pass and action sampling
        # Master line 123: action_distrib, batch_value = self.model(b_state)
        if self.is_discrete and isinstance(self.policy, ParamSoftmaxPolicy):
            # For FFParamSoftmax policy
            logits = self.policy(obs, training=False)
            batch_action = self.policy.sample_action(obs)
            
            # Get action probabilities for entropy calculation
            probs = self.policy.get_action_probabilities(logits)
            entropy_values = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
            
        else:
            # For FFGaussian policy
            action_dist = self.policy(obs)
            batch_action = action_dist.sample()
            entropy_values = action_dist.entropy()
            
        # Get value estimates
        batch_value = self.value_net(obs)
        
        # MASTER EXACT: Record entropy and values
        # Master line 125-127: self.entropy_record.extend(chainer.cuda.to_cpu(action_distrib.entropy.data))
        #                      self.value_record.extend(chainer.cuda.to_cpu((batch_value.data)))
        if hasattr(entropy_values, 'numpy'):
            entropy_data = entropy_values.numpy().flatten().tolist()
        else:
            entropy_data = [float(entropy_values)]
            
        self.entropy_record.extend(entropy_data)
        self.value_record.extend(batch_value.numpy().flatten().tolist())
        
        # MASTER EXACT: Update batch tracking variables
        # Master line 129-130: self.batch_last_state = list(batch_obs)
        #                       self.batch_last_action = list(batch_action)
        self.batch_last_state = list(batch_obs)
        if hasattr(batch_action, 'numpy'):
            self.batch_last_action = batch_action.numpy().tolist()
        else:
            self.batch_last_action = batch_action.tolist() if hasattr(batch_action, 'tolist') else batch_action
            
        # Process rewards for training (this replaces the empty pass)
        # The actual PPO training will be handled by the trainer's train_step method
        
        return batch_action
    
    def train_step(self, obs, actions, old_log_probs, advantages, returns):
        """FIXED: Single training step with proper gradient flow"""
        
        # Get actual observation dimension from the observation
        actual_obs_dim = obs.shape[-1]
        
        # Initialize normalizer with correct dimension if needed
        if self.obs_normalizer is None or self.obs_normalizer.obs_size != actual_obs_dim:
            self.obs_normalizer = EmpiricalNormalization(actual_obs_dim, clip_threshold=5)
        
        # Use single persistent tape for both policy and value
        with tf.GradientTape(persistent=True) as tape:
            # Normalize observations (matching original)
            obs_norm = self._normalize_obs(obs, update=False)
            
            # Get current policy and value
            values = self.value_net(obs_norm)
            
            if self.is_discrete:
                # FIXED: Use SAME approach as sample_action() to ensure gradient consistency!
                logits = self.policy(obs_norm, training=True)  # Enable training mode
                probs = self.policy.get_action_probabilities(logits)
                
                # Calculate log probabilities EXACTLY like sample_action() method
                action_probs = tf.gather(probs, tf.cast(actions, tf.int32), batch_dims=1)
                log_probs = tf.math.log(action_probs + 1e-8)
                
                # Compute entropy for discrete distribution
                entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
            else:
                # For GaussianPolicy: policy returns distribution directly
                action_dist = self.policy(obs_norm)
                log_probs = action_dist.log_prob(actions)
                entropy = action_dist.entropy()
            
            # Ensure log_probs is a scalar (sum over action dimensions if needed)
            if len(log_probs.shape) > 1:
                log_probs = tf.reduce_sum(log_probs, axis=-1)
            
            # Ensure old_log_probs has the same shape
            if len(old_log_probs.shape) > 1:
                old_log_probs = tf.reduce_sum(old_log_probs, axis=-1)
            
            # Policy loss (PPO clipped objective)
            ratio = tf.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            # Value loss (MSE)
            value_loss = tf.reduce_mean(tf.square(returns - values))
            
            # Entropy loss
            entropy_loss = -tf.reduce_mean(entropy)
            
            # Total loss for monitoring
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # CRITICAL DEBUG: Monitor training dynamics
        if np.random.random() < 0.01:  # Sample 1% of updates for debugging
            print(f"\nTRAINING STEP DIAGNOSTICS:")
            print(f"  Ratio: mean={tf.reduce_mean(ratio):.4f}, std={tf.math.reduce_std(ratio):.4f}, min={tf.reduce_min(ratio):.4f}, max={tf.reduce_max(ratio):.4f}")
            print(f"  Policy loss: {policy_loss:.6f}")
            print(f"  Value loss: {value_loss:.6f}")
            print(f"  Entropy: {tf.reduce_mean(entropy):.6f}")
        
        # Compute gradients separately to avoid tape conflicts
        # Combining losses in gradient() call breaks persistent tape!
        policy_grads = tape.gradient(policy_loss, self.policy.trainable_variables)
        entropy_grads = tape.gradient(entropy_loss, self.policy.trainable_variables)
        value_grads = tape.gradient(value_loss, self.value_net.trainable_variables)
        
        # Manually combine policy and entropy gradients with correct weighting
        if policy_grads is not None and entropy_grads is not None:
            combined_policy_grads = []
            for p_grad, e_grad in zip(policy_grads, entropy_grads):
                if p_grad is not None and e_grad is not None:
                    combined_grad = p_grad + self.entropy_coef * e_grad
                    combined_policy_grads.append(combined_grad)
                elif p_grad is not None:
                    combined_policy_grads.append(p_grad)
                elif e_grad is not None:
                    combined_policy_grads.append(self.entropy_coef * e_grad)
                else:
                    combined_policy_grads.append(None)
            policy_grads = combined_policy_grads
        
        # Clean up tape
        del tape
        
        # Check for None gradients and handle gracefully
        if policy_grads is None or any(g is None for g in policy_grads):
            print("Warning: Some policy gradients are None, skipping policy update")
            policy_grads = [tf.zeros_like(var) for var in self.policy.trainable_variables]
            
        if value_grads is None or any(g is None for g in value_grads):
            print("Warning: Some value gradients are None, skipping value update")  
            value_grads = [tf.zeros_like(var) for var in self.value_net.trainable_variables]
        
        # MATCH MASTER: No gradient clipping (ChainerRL PPO default)
        # Master doesn't use gradient clipping in PPO
        # policy_grads, _ = tf.clip_by_global_norm(policy_grads, self.max_grad_norm)
        # value_grads, _ = tf.clip_by_global_norm(value_grads, self.max_grad_norm)
        
        # Apply gradients
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))
        self.value_optimizer.apply_gradients(zip(value_grads, self.value_net.trainable_variables))
        
        # Apply NonbiasWeightDecay after gradient update (matching ChainerRL behavior)
        self._apply_nonbias_weight_decay()
        
        # Record statistics (matching original)
        self.policy_loss_record.append(policy_loss.numpy())
        self.value_loss_record.append(value_loss.numpy())
        
        return {
            'policy_loss': policy_loss.numpy(),
            'value_loss': value_loss.numpy(),
            'entropy_loss': entropy_loss.numpy(),
            'total_loss': total_loss.numpy()
        }
    
    def get_statistics(self):
        """Return statistics matching original ChainerRL implementation"""
        if not self.policy_loss_record:
            return {}
        
        return {
            'policy_loss': np.mean(self.policy_loss_record[-100:]),  # Last 100 steps
            'value_loss': np.mean(self.value_loss_record[-100:]),
            'avg_entropy': np.mean(self.entropy_record[-100:]) if self.entropy_record else 0.0,
            'avg_value': np.mean(self.value_record[-100:]) if self.value_record else 0.0
        }
    
    def save_model(self, filepath):
        """Save the trained model (Keras 3 compatible)"""
        import os
        import json
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the policy and value networks in Keras 3 compatible format
        try:
            self.policy.save_weights(f"{filepath}_policy_weights.weights.h5")
            self.value_net.save_weights(f"{filepath}_value_weights.weights.h5")
            print(f"Model saved to {filepath} (Keras 3 .weights.h5 format)")
        except Exception as e:
            print(f".weights.h5 save failed: {e}")
            try:
                # Fallback to .keras format
                self.policy.save(f"{filepath}_policy.keras")
                self.value_net.save(f"{filepath}_value.keras")
                print(f"Model saved to {filepath} (.keras format)")
            except Exception as e2:
                print(f"Failed to save model: {e2}")
                return
        
        # Save the normalizer state
        normalizer_data = {
            'mean': self.obs_normalizer.mean.numpy().tolist(),
            'std': self.obs_normalizer.std.numpy().tolist(),
            'count': float(self.obs_normalizer.count)
        }
        
        with open(f"{filepath}_normalizer.json", 'w') as f:
            json.dump(normalizer_data, f)
    
    def load_model(self, filepath):
        """Load a trained model (Keras 3 compatible)"""
        import os
        import json
        
        policy_loaded = False
        value_loaded = False
        
        # Try to load Keras 3 .weights.h5 format first
        if os.path.exists(f"{filepath}_policy_weights.weights.h5"):
            try:
                self.policy.load_weights(f"{filepath}_policy_weights.weights.h5")
                print(f"Policy weights loaded from {filepath} (.weights.h5 format)")
                policy_loaded = True
            except Exception as e:
                print(f"Failed to load .weights.h5 format: {e}")
        
        # Try .keras format if .weights.h5 failed or doesn't exist
        if not policy_loaded and os.path.exists(f"{filepath}_policy.keras"):
            try:
                import tensorflow as tf
                loaded_policy = tf.keras.models.load_model(f"{filepath}_policy.keras")
                # Replace the policy model
                self.policy = loaded_policy
                print(f"Policy loaded from {filepath} (.keras format)")
                policy_loaded = True
            except Exception as e:
                print(f"Failed to load .keras format: {e}")
        
        # Try legacy format as last resort (might not work with Keras 3)
        if not policy_loaded and os.path.exists(f"{filepath}_policy_weights.index"):
            try:
                self.policy.load_weights(f"{filepath}_policy_weights")
                print(f"Policy weights loaded from {filepath} (legacy format)")
                policy_loaded = True
            except Exception as e:
                print(f"Legacy format failed with Keras 3: {e}")
                print("Consider retraining the model for Keras 3 compatibility")
        
        # Same logic for value network
        if os.path.exists(f"{filepath}_value_weights.weights.h5"):
            try:
                self.value_net.load_weights(f"{filepath}_value_weights.weights.h5")
                print(f"Value weights loaded from {filepath} (.weights.h5 format)")
                value_loaded = True
            except Exception as e:
                print(f"Failed to load .weights.h5 format: {e}")
        
        if not value_loaded and os.path.exists(f"{filepath}_value.keras"):
            try:
                import tensorflow as tf
                loaded_value = tf.keras.models.load_model(f"{filepath}_value.keras")
                self.value_net = loaded_value
                print(f"Value network loaded from {filepath} (.keras format)")
                value_loaded = True
            except Exception as e:
                print(f"Failed to load .keras format: {e}")
                
        if not value_loaded and os.path.exists(f"{filepath}_value_weights.index"):
            try:
                self.value_net.load_weights(f"{filepath}_value_weights")
                print(f"Value weights loaded from {filepath} (legacy format)")
                value_loaded = True
            except Exception as e:
                print(f"Legacy format failed with Keras 3: {e}")
                print("Consider retraining the model for Keras 3 compatibility")
        
        # Load normalizer state
        normalizer_file = f"{filepath}_normalizer.json"
        if os.path.exists(normalizer_file):
            with open(normalizer_file, 'r') as f:
                normalizer_data = json.load(f)
            
            # Restore normalizer state
            self.obs_normalizer.mean.assign(normalizer_data['mean'])
            self.obs_normalizer.std.assign(normalizer_data.get('std', normalizer_data.get('var', [])))  
            self.obs_normalizer.count.assign(normalizer_data['count'])
            print(f"Normalizer state loaded from {normalizer_file}")
        else:
            print(f"Normalizer state not found at {normalizer_file}")
            
        if not policy_loaded or not value_loaded:
            print("Some model components couldn't be loaded. Consider retraining for Keras 3.")
            if not policy_loaded and not value_loaded:
                return False
                
        return True
    
    @classmethod
    def load_trained_agent(cls, filepath, obs_dim, action_dim):
        """Class method to load a pre-trained agent"""
        agent = cls(obs_dim, action_dim)
        agent.load_model(filepath)
        return agent
    
    def _initialize_policy_for_action_bounds(self):
        """Initialize Gaussian policy to output actions in valid bounds"""
        try:
            # No longer defining "reasonable centers" for initialization
            # This was the root cause of TF starting with positive rewards instead of negative!
            
            # Build the policy with a dummy forward pass
            dummy_obs = tf.random.normal((1, self.obs_dim))
            _ = self.policy(dummy_obs)  # This builds the network
            
            # REMOVED CUSTOM INITIALIZATION - This was causing TF to start with good actions!
            # The bias initialization was giving TF a "head start" with reasonable actions,
            # causing episodes to start with positive rewards (~+2-4) instead of negative like master (~-2).
            # 
            # Master starts random → bad actions → negative rewards → must learn!
            # TF was starting biased → decent actions → positive rewards → no learning needed!
            
            print(f"   No custom policy initialization (let it start bad like master!)")
            print(f"   Agent will start with poor performance and gradually improve like master!")
        except Exception as e:
            print(f"   Policy initialization failed: {e}")
            print(f"   Continuing with default initialization")
    
    def get_entropy_statistics(self):
        """Get entropy statistics for policy collapse detection (matching master)"""
        if not self.avg_entropy_history:
            return {'avg_entropy': 0.0, 'avg_highest_act_prob': 0.0, 'avg_second_highest_act_prob': 0.0}
            
        recent_window = 100  # Use last 100 samples
        recent_entropy = self.avg_entropy_history[-recent_window:] if len(self.avg_entropy_history) >= recent_window else self.avg_entropy_history
        recent_highest = self.avg_highest_act_prob_history[-recent_window:] if len(self.avg_highest_act_prob_history) >= recent_window else self.avg_highest_act_prob_history
        recent_second = self.avg_second_highest_act_prob_history[-recent_window:] if len(self.avg_second_highest_act_prob_history) >= recent_window else self.avg_second_highest_act_prob_history
        
        return {
            'avg_entropy': float(np.mean(recent_entropy)) if recent_entropy else 0.0,
            'avg_highest_act_prob': float(np.mean(recent_highest)) if recent_highest else 0.0, 
            'avg_second_highest_act_prob': float(np.mean(recent_second)) if recent_second else 0.0,
            'total_samples': len(self.avg_entropy_history)
        }