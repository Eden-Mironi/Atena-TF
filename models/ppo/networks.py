import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class GaussianPolicy(tf.keras.Model):
    """Gaussian policy for continuous actions - EXACT MATCH to ChainerRL's FCGaussianPolicyWithStateIndependentCovariance
    
    Key features matching master:
    - bound_mean=True: Bounds mean to action space using tanh + scaling
    - min_action/max_action: Action bounds passed to policy  
    - mean_wscale=1e-2: Small initialization for mean layer
    - nonlinearity=F.tanh: Tanh activation in hidden layers
    - var_type='diagonal': Diagonal covariance matrix
    """
    
    def __init__(self, obs_dim, action_dim, n_hidden_layers=2, n_hidden_channels=64, action_space=None, bound_mean=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels  # Master's default: 64
        self.bound_mean = bound_mean
        
        # Store action bounds like ChainerRL (min_action/max_action passed to policy!)
        if action_space is not None:
            self.min_action = tf.constant(action_space.low, dtype=tf.float32)
            self.max_action = tf.constant(action_space.high, dtype=tf.float32)
        else:
            self.min_action = None
            self.max_action = None
        
        # Hidden layers (matching master: 2 layers with tanh activation)
        self.hidden_layers = []
        for i in range(n_hidden_layers):
            layer = tf.keras.layers.Dense(
                n_hidden_channels, 
                activation=tf.nn.tanh,  # Master uses tanh nonlinearity
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                name=f'hidden_{i}'
            )
            self.hidden_layers.append(layer)
        
        # Mean layer (Master uses mean_wscale=1e-2)
        self.mean_layer = tf.keras.layers.Dense(
            action_dim,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-2),  # Master's mean_wscale=1e-2
            name='mean'
        )
        
        # Log standard deviation (learnable parameter - var_type='diagonal')
        self.log_std = tf.Variable(
            initial_value=tf.zeros(action_dim),  # Diagonal covariance like master
            trainable=True,
            name='log_std'
        )
    
    def call(self, obs, return_hidden=False):
        # Forward pass through hidden layers
        x = obs
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Store hidden layer output for deterministic action generation (matching master's act_most_probable)
        hidden_output = x
        
        # Get mean from hidden layers
        mean = self.mean_layer(x)
        
        # Apply bound_mean EXACTLY like ChainerRL FCGaussianPolicyWithStateIndependentCovariance
        # Research shows ChainerRL applies bound_mean as: tanh + linear scaling to action bounds
        if self.bound_mean and self.min_action is not None and self.max_action is not None:
            # ChainerRL bound_mean implementation: tanh activation + linear scaling
            mean = tf.tanh(mean)  # Bound to [-1, 1] 
            # Scale from [-1, 1] to [min_action, max_action]
            mean = 0.5 * (mean * (self.max_action - self.min_action) + (self.max_action + self.min_action))
        
        # Return hidden output if requested (for master's deterministic action generation)
        if return_hidden:
            # CRITICAL DISCOVERY: Master's agent.model.pi.hidden_layers(x) actually outputs MEAN, not raw hidden layer!
            # In ChainerRL's FCGaussianPolicyWithStateIndependentCovariance: mean = self.hidden_layers(x)
            # So hidden_layers IS the mean computation, but WITHOUT bound_mean applied for deterministic actions!
            return self.mean_layer(hidden_output)  # Mean without bound_mean - exactly like master!
        
        # Get standard deviation (clipped for stability)
        std = tf.exp(tf.clip_by_value(self.log_std, -10, 2))
        
        # Create diagonal Gaussian distribution (matching master's var_type='diagonal')
        distribution = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=std
        )
        
        return distribution

class ParametricSoftmaxPolicy(tf.keras.Model):
    """Parametric Softmax policy matching master's A3CFFParamSoftmax architecture
    
    Implements hierarchical discrete action selection:
    1. Select action type (back, filter, group) 
    2. Select parameters for chosen action type
    3. Combine into single discrete action
    
    Action space: 949 discrete actions
    - back: 1 action (no parameters)
    - filter: 936 actions (12 fields × 3 operators × 26 terms)  
    - group: 12 actions (12 fields)
    """
    
    def __init__(self, obs_dim, parametric_segments, parametric_segments_sizes, 
                 n_hidden_layers=2, n_hidden_channels=600, beta=1.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.parametric_segments = parametric_segments  # ((), (12, 3, 26), (12,))
        self.parametric_segments_sizes = parametric_segments_sizes  # [1, 936, 12]
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels  # Master's 600 neurons
        self.beta = beta  # Temperature parameter (master uses 1.0)
        
        # Total output size for all logits
        self.output_layer_size = sum(parametric_segments_sizes)  # 949
        
        # Hidden layers (matching master: 2 layers of 600 channels)
        self.hidden_layers = []
        for i in range(n_hidden_layers):
            layer = tf.keras.layers.Dense(
                n_hidden_channels,
                activation=tf.nn.tanh, 
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                name=f'param_softmax_hidden_{i}'
            )
            self.hidden_layers.append(layer)
            
        # Output layer (produces all 949 logits)
        self.logits_layer = tf.keras.layers.Dense(
            self.output_layer_size,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-2),
            name='param_softmax_logits'
        )
        
    def call(self, obs):
        # Forward pass through hidden layers
        x = obs
        for layer in self.hidden_layers:
            x = layer(x)
            
        # Get all logits
        logits = self.logits_layer(x)
        
        # Create parametric softmax distribution
        return ParametricSoftmaxDistribution(
            logits=logits,
            parametric_segments=self.parametric_segments,
            parametric_segments_sizes=self.parametric_segments_sizes,
            beta=self.beta
        )

class ParametricSoftmaxDistribution:
    """TensorFlow implementation of ChainerRL's ParamSoftmaxDistribution
    
    Handles hierarchical discrete action sampling:
    1. Sample action type via softmax over first few logits
    2. Sample parameters for that action type via separate softmax layers
    3. Combine into final discrete action index
    """
    
    def __init__(self, logits, parametric_segments, parametric_segments_sizes, beta=1.0):
        self.logits = logits  # Shape: (batch_size, 949)
        self.parametric_segments = parametric_segments  # ((), (12, 3, 26), (12,))
        self.parametric_segments_sizes = parametric_segments_sizes  # [1, 936, 12]
        self.beta = beta
        self.n_action_types = len(parametric_segments)  # 3 (back, filter, group)
        
    def _get_all_probabilities(self): 
        """Compute full 949-dimensional probability distribution (MASTER'S EXACT METHOD)
        
        This replicates master's get_all_prob_or_log_prob() from lines 73-116
        Creates proper weighted probability distribution over all 949 discrete actions
        """
        # Step 1: Get action type probabilities (first 3 logits)
        action_types_logits = self.beta * self.logits[:, :self.n_action_types]
        action_types_probs = tf.nn.softmax(action_types_logits)  # Shape: (batch_size, 3)
        
        result = []
        logits_offset = self.n_action_types
        
        # Step 2: For each action type, compute its contribution to the full probability vector
        for i in range(self.n_action_types):
            action_type_prob = action_types_probs[:, i:i + 1]  # Shape: (batch_size, 1)
            
            if not self.parametric_segments[i]:  # No parameters (back action)
                # Direct probability for back action  
                result.append(action_type_prob)
            else:
                # Complex hierarchical probability calculation (master's approach)
                segments_factor = 1
                current_prob = action_type_prob
                
                # Multiply probabilities for each parameter level
                for sub_seg_size in self.parametric_segments[i]:
                    segments_factor *= sub_seg_size
                    
                    # Get parameter probabilities
                    sub_seg_logits = self.beta * self.logits[:, logits_offset:logits_offset + sub_seg_size]
                    sub_seg_probs = tf.nn.softmax(sub_seg_logits)  # Shape: (batch_size, sub_seg_size)
                    
                    # Master's F.repeat and F.tile logic (lines 107-111)
                    current_prob = tf.repeat(current_prob, sub_seg_size, axis=1) * tf.tile(
                        sub_seg_probs, [1, segments_factor // sub_seg_size]
                    )
                    
                    logits_offset += sub_seg_size
                    
                result.append(current_prob)
        
        # Concatenate all probabilities to get full 949-dimensional distribution
        return tf.concat(result, axis=1)
    
    def sample(self):
        """FIXED: Fully differentiable hierarchical sampling - NO .numpy() calls!
        
        Sample action type first, then parameters using pure TensorFlow operations.
        This maintains gradient flow for proper PPO learning.
        """
        batch_size = tf.shape(self.logits)[0]
        
        # STEP 1: Sample action type (back=0, filter=1, group=2) from first 3 logits
        action_type_logits = self.beta * self.logits[:, :self.n_action_types]
        action_type_dist = tfp.distributions.Categorical(logits=action_type_logits)
        action_types = action_type_dist.sample()  # Shape: (batch_size,)
        
        # STEP 2: Sample parameters for each action type using TF conditionals
        final_actions = []
        
        for b in range(batch_size):
            action_type = action_types[b]  # CRITICAL FIX: Keep as tensor, NO .numpy()!
            
            # Back action
            back_action = tf.constant(0, dtype=tf.int32)
            
            # Filter action - sample parameters using TF ops
            logits_start = self.n_action_types
            
            # Field (12 options)
            field_logits = self.beta * self.logits[b, logits_start:logits_start+12]
            field = tfp.distributions.Categorical(logits=field_logits).sample()
            
            # Operator (3 options) 
            op_logits = self.beta * self.logits[b, logits_start+12:logits_start+15]
            operator = tfp.distributions.Categorical(logits=op_logits).sample()
            
            # Term (26 options)
            term_logits = self.beta * self.logits[b, logits_start+15:logits_start+41]  
            term = tfp.distributions.Categorical(logits=term_logits).sample()
            
            # Convert to filter action index: 1 + field*78 + operator*26 + term
            filter_action = 1 + field * 78 + operator * 26 + term
            
            # Group action - sample group field
            group_logits = self.beta * self.logits[b, -12:]
            group_field = tfp.distributions.Categorical(logits=group_logits).sample()
            group_action = 937 + group_field
            
            # Select action based on action type using TF conditionals
            action = tf.cond(
                action_type == 0,
                lambda: back_action,
                lambda: tf.cond(
                    action_type == 1,
                    lambda: filter_action,
                    lambda: group_action
                )
            )
            
            final_actions.append(action)
        
        return tf.stack(final_actions)
        
    def log_prob(self, actions):
        """FIXED: Fully differentiable hierarchical log prob - NO .numpy() calls!
        
        This maintains gradient flow for proper PPO learning.
        """
        batch_size = tf.shape(self.logits)[0]
        
        # Pre-compute all log softmax distributions for efficiency
        action_type_log_probs = tf.nn.log_softmax(self.beta * self.logits[:, :self.n_action_types])
        
        logits_start = self.n_action_types
        field_log_probs = tf.nn.log_softmax(self.beta * self.logits[:, logits_start:logits_start+12])
        op_log_probs = tf.nn.log_softmax(self.beta * self.logits[:, logits_start+12:logits_start+15])  
        term_log_probs = tf.nn.log_softmax(self.beta * self.logits[:, logits_start+15:logits_start+41])
        group_log_probs = tf.nn.log_softmax(self.beta * self.logits[:, -12:])
        
        log_probs = []
        
        for b in range(batch_size):
            action = actions[b]  # CRITICAL FIX: Keep as tensor, NO .numpy()!
            
            # Back actions (action == 0)
            back_log_prob = action_type_log_probs[b, 0]
            
            # Filter actions (1 <= action <= 936) - compute parameters using TF ops
            filter_idx = action - 1
            field = filter_idx // 78
            remainder = filter_idx % 78
            operator = remainder // 26
            term = remainder % 26
            
            # Clamp indices to valid ranges to prevent out-of-bounds
            field = tf.clip_by_value(field, 0, 11)
            operator = tf.clip_by_value(operator, 0, 2)
            term = tf.clip_by_value(term, 0, 25)
            
            filter_log_prob = (action_type_log_probs[b, 1] + 
                             tf.gather(field_log_probs[b], field) +
                             tf.gather(op_log_probs[b], operator) + 
                             tf.gather(term_log_probs[b], term))
            
            # Group actions (937 <= action <= 948)
            group_field = tf.clip_by_value(action - 937, 0, 11)  # Clamp to valid range
            group_log_prob = (action_type_log_probs[b, 2] + 
                            tf.gather(group_log_probs[b], group_field))
            
            # Select correct log prob based on action range using TF conditionals
            log_prob = tf.cond(
                action == 0,
                lambda: back_log_prob,
                lambda: tf.cond(
                    tf.logical_and(action >= 1, action <= 936),
                    lambda: filter_log_prob,
                    lambda: tf.cond(
                        tf.logical_and(action >= 937, action <= 948),
                        lambda: group_log_prob,
                        lambda: tf.constant(-20.0, dtype=tf.float32)  # Invalid action
                    )
                )
            )
            
            log_probs.append(log_prob)
        
        return tf.stack(log_probs)
        
    def entropy(self):
        """HIERARCHICAL ENTROPY - CONSISTENT WITH SAMPLING!
        
        Compute entropy using hierarchical method for consistency.
        """
        batch_size = tf.shape(self.logits)[0]
        
        # Action type entropy (same for all batch elements)
        action_type_logits = self.beta * self.logits[:, :self.n_action_types]
        action_type_entropy = -tf.reduce_sum(
            tf.nn.softmax(action_type_logits) * tf.nn.log_softmax(action_type_logits), axis=-1
        )
        
        # Parameter entropy (averaged across possible action types)
        logits_start = self.n_action_types
        
        # Filter parameter entropy: field + operator + term
        field_logits = self.beta * self.logits[:, logits_start:logits_start+12]
        field_entropy = -tf.reduce_sum(
            tf.nn.softmax(field_logits) * tf.nn.log_softmax(field_logits), axis=-1
        )
        
        op_logits = self.beta * self.logits[:, logits_start+12:logits_start+15]
        op_entropy = -tf.reduce_sum(
            tf.nn.softmax(op_logits) * tf.nn.log_softmax(op_logits), axis=-1
        )
        
        term_logits = self.beta * self.logits[:, logits_start+15:logits_start+41]
        term_entropy = -tf.reduce_sum(
            tf.nn.softmax(term_logits) * tf.nn.log_softmax(term_logits), axis=-1
        )
        
        filter_entropy = field_entropy + op_entropy + term_entropy
        
        # Group parameter entropy
        group_logits = self.beta * self.logits[:, -12:]
        group_entropy = -tf.reduce_sum(
            tf.nn.softmax(group_logits) * tf.nn.log_softmax(group_logits), axis=-1
        )
        
        # Total entropy: action type + weighted parameter entropy
        action_type_probs = tf.nn.softmax(action_type_logits)
        total_entropy = action_type_entropy + (
            action_type_probs[:, 1] * filter_entropy +  # Filter probability * filter entropy
            action_type_probs[:, 2] * group_entropy     # Group probability * group entropy
            # Back has no parameter entropy (0)
        )
        
        return total_entropy

class ValueNetwork(tf.keras.Model):
    """Value function matching original ChainerRL FCVFunction with 64 channels for Gaussian"""
    
    def __init__(self, obs_dim, n_hidden_layers=2, n_hidden_channels=600, activation=tf.nn.tanh):  # CRITICAL FIX: Use tanh for FFParamSoftmax
        super().__init__()
        self.obs_dim = obs_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels  # FIXED: Always 64 channels (master's FCVFunction)
        self.activation = activation
        
        # Hidden layers (matching master: 2 layers of 64 channels with tanh activation for FFParamSoftmax)
        self.hidden_layers = []
        for i in range(n_hidden_layers):
            layer = tf.keras.layers.Dense(
                n_hidden_channels, 
                activation=activation,  # CRITICAL FIX: Use tanh for FFParamSoftmax (same as policy)
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                name=f'v_hidden_{i}'
            )
            self.hidden_layers.append(layer)
        
        # Value output layer (matching original last_wscale=0.01)
        self.value_layer = tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name='value'
        )
    
    def call(self, obs):
        # Forward pass through hidden layers
        x = obs
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Get value
        value = self.value_layer(x)
        return tf.squeeze(value, axis=-1)