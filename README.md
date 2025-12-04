# ATENA: Automated Tabular Exploration and Navigation Assistant

**AI-Powered Data Exploration System using Deep Reinforcement Learning**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange)
![Keras](https://img.shields.io/badge/Keras-3.0-red)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green)

---

## 📖 Table of Contents

- [Overview](#overview)
- [What is ATENA?](#what-is-atena)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
- [Reward System](#reward-system)
- [Training](#training)
- [Evaluation](#evaluation)
- [Notebooks Guide](#notebooks-guide)
- [Performance Results](#performance-results)
- [Component Documentation](#component-documentation)
- [Advanced Topics](#advanced-topics)

---

## Overview

ATENA (Automated Tabular Exploration and Navigation Assistant) is an intelligent **reinforcement learning agent** that learns to explore tabular datasets like a human data analyst. The system uses **Proximal Policy Optimization (PPO)** to learn effective data exploration strategies through three core actions: **filter**, **group**, and **back**.

### Key Features

✅ **Human-like Exploration**: Learns from 76 real analyst sessions  
✅ **Intelligent Actions**: Balanced filter/group/back action distribution  
✅ **Multi-dimensional Rewards**: Diversity + Interestingness + Humanity  
✅ **Weak Supervision**: Snorkel-based learning from labeling functions  
✅ **High Performance**: 84% BLEU-4 similarity with expert sessions  
✅ **Production Ready**: Modern TensorFlow 2.x / Keras 3 implementation  

### What Makes ATENA Special?

Unlike traditional data exploration tools, ATENA:
- **Learns** exploration strategies rather than following fixed rules
- **Adapts** to different datasets and domains
- **Balances** exploration (discovering new patterns) with exploitation (following promising leads)
- **Mimics** human analyst behavior through weak supervision
- **Provides** next-step recommendations for interactive data analysis

---

## What is ATENA?

### The Problem: Modern Interactive Data Analysis

Data analysts face challenges exploring complex datasets:
- **Too many possible paths**: Which column to filter? What values to group by?
- **No guidance**: Traditional tools don't suggest next steps
- **Expertise required**: Non-experts struggle with effective exploration strategies
- **Time-consuming**: Manual trial-and-error is inefficient

### The Solution: AI-Powered Exploration Assistant

ATENA is a **reinforcement learning agent** trained to:
1. **Observe** the current data view
2. **Suggest** the next exploration action
3. **Execute** filters, groupings, or navigation
4. **Learn** from rewards based on diversity, interestingness, and human-likeness

### Core Actions

#### 1. **Filter** 
Create data subsets based on column conditions.

**Example**: Filter `ip_src = '192.168.1.1'` to focus on traffic from a specific source.

#### 2. **Group**
Aggregate data by columns with statistical summaries.

**Example**: Group by `tcp_dstport` to see traffic distribution across destination ports.

#### 3. **Back**
Navigate to previous exploration states.

**Example**: Return to the parent view after exploring a filtered subset.

---

## Quick Start

### Prerequisites

- Python 3.10+ (3.12 recommended)
- 8GB+ RAM
- (Optional) CUDA-compatible GPU for faster training

### Installation

```bash
# Navigate to project
cd "atena-tf"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install gym_atena package
pip install -e .
```

### Quick Demo

Try ATENA interactively with a pre-trained model:

```bash
jupyter notebook ./Notebooks/ATENA-TF-Master-Welcome.ipynb
```

The welcome notebooks demonstrate:
- Loading the environment
- Taking manual actions
- Understanding the action space
- Analyzing rewards and reward components
- Getting AI recommendations (ATENA-TF-Welcome)
- Analyst view with HTML navigation (ATENA-TF-Master-Welcome)

### Training Your Own Agent

```bash
# Standard training (500K steps, ~2-3 hours on GPU)
python train_with_decay.py --steps 500000 --seed 42 --enable-decay

# Quick test run (100K steps, ~30 minutes)
python train_with_decay.py --steps 100000 --seed 42

# Full training (1M steps, ~6-8 hours)
python train_with_decay.py --steps 1000000 --seed 42 --enable-decay
```

**Training outputs**:
- Trained models saved to `results/{timestamp}/`
- Episode logs in `episode_summary.jsonl`
- Checkpoints every 50K steps
- Action distribution tracking

### Generating Exploration Sessions

```bash
# Generate a readable exploration session
python generate_session_output.py \
    --model_path results/latest/trained_model \
    --dataset 0 \
    --steps 12 \
    --output my_session.txt
```

---

## System Architecture

### High-Level Overview

```
┌─────────────────┐
│   Tabular Data  │
│   (CSV/TSV)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   ATENA Environment (OpenAI Gym)    │
│   - State representation            │
│   - Action execution                │
│   - Reward calculation              │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   PPO Agent (TensorFlow/Keras)      │
│   - Policy network (600→600→949)    │
│   - Value network (600→600→1)       │
│   - Action selection                │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Training Loop                     │
│   - Experience collection           │
│   - Advantage estimation (GAE)      │
│   - Policy/value updates            │
└─────────────────────────────────────┘
```

### Core Components

```
atena-tf/
├── Configuration/
│   └── config.py                    # System parameters & reward coefficients
│
├── gym_atena/                       # OpenAI Gym Environment
│   ├── envs/
│   │   ├── atena_env_cont.py        # Base continuous action environment
│   │   ├── enhanced_atena_env.py    # Enhanced with complete rewards
│   │   └── env_properties.py        # Dataset operations (filter, group)
│   ├── lib/
│   │   ├── tokenization.py          # Column value tokenization
│   │   └── tree_measures.py         # BLEU/GLEU evaluation metrics
│   ├── data_schemas/                # Dataset schemas
│   │   ├── networking_schema.py     # Network packet analysis
│   │   ├── flights_schema.py        # Flight data analysis
│   │   └── ...                      # Other dataset schemas
│   └── reactida/                    # REACT-IDA benchmark datasets
│       └── raw_datasets/            # 4 real analyst session datasets
│
├── models/
│   └── ppo/
│       ├── agent.py                 # PPO Agent (TensorFlow 2 / Keras 3)
│       └── networks.py              # Policy & Value networks
│
├── training/
│   ├── trainer.py                   # Training loop manager
│   └── vectorized_envs.py           # Environment vectorization
│
├── Evaluation/
│   ├── evaluator_tf.py              # Comprehensive evaluation system
│   ├── evaluation_measures_tf.py    # Tree BLEU, TED, Precision/Recall
│   └── notebook_utils.py            # Utilities for notebooks
│
├── snorkel_checkpoints/             # Pre-trained Snorkel models
│   ├── GenerativeModel.weights.pkl  # Weak supervision model
│   └── priors.txt                   # Prior probabilities
│
├── train_with_decay.py              # Primary training script
├── evaluate_model.py                # Model evaluation CLI
├── generate_session_output.py       # Session generation
└── create_real_proof_comparison.py  # Performance validation
```

---

## How It Works

### Training Loop (Episode-based)

```
┌─────────────────────────────────────────────────────────────┐
│                    ATENA Training Loop                       │
└─────────────────────────────────────────────────────────────┘

1. ENVIRONMENT INITIALIZATION
   Dataset → Initial State (full data view)
   
2. AGENT INTERACTION (12 steps per episode)
   For each step:
   ├── Observe: State vector (51D: column stats, grouping state)
   ├── Think: PPO Policy Network → Action probabilities over 949 actions
   ├── Act: Select action [type, column, operator, value, ...]
   │   ├── Filter: Column + condition + value
   │   ├── Group: Column + aggregation
   │   └── Back: Return to previous state
   ├── Execute: Environment applies action → New data view
   └── Reward: Composite reward from multiple components

3. REWARD CALCULATION
   Total Reward = Diversity + Interestingness + Humanity
   
   ├── Diversity (coeff: 2.0)
   │   └── Minimum similarity to previous states
   │
   ├── Interestingness (coeffs: 1.5 KL, 2.0 compaction)
   │   ├── KL Divergence (for filters)
   │   └── Compaction Gain (for groups)
   │
   └── Humanity (coeff: 1.0)
       ├── Rule-based scoring (35+ rules)
       └── Snorkel classifier (learned from human sessions)

4. PPO UPDATE (every 2048 steps)
   ├── Compute advantages using GAE (λ=0.97, γ=0.995)
   ├── Update policy network (clip ratio=0.2)
   └── Update value network (MSE loss)
```

### State Representation (51 dimensions)

The agent observes each data view as a 51-dimensional vector:

**Column Statistics** (12 fields × 4 stats = 48 dimensions):
- Unique value count (normalized)
- Column entropy
- Data type indicator
- Visibility flag (whether column is displayed)

**Navigation State** (3 dimensions):
- Current depth in exploration tree
- Number of available back actions
- Grouping state indicator

### Action Space (949 discrete actions)

**Parametric Softmax Structure**:
- **1 back action**: No parameters needed
- **936 filter actions**: 12 fields × 3 operators × 26 value bins
  - Operators: equals (==), greater than (>), less than (<)
  - Value bins: Discretized column value space
- **12 group actions**: One per field (with automatic aggregation)

**Total**: 1 + 936 + 12 = **949 discrete actions**

---

## Reward System

The reward system is the **core intelligence** of ATENA, teaching the agent what makes good data exploration.

### 1. Diversity Reward (coefficient: 2.0)

**Purpose**: Encourage exploring new views, avoid repetition.

**Calculation**:
```python
# Compute similarity to ALL previous states
similarities = [cosine_similarity(new_state, prev_state) 
                for prev_state in history]

# Reward based on minimum similarity (most different view)
diversity_reward = 2.0 * (1 - min(similarities))
```

**Example**:
- New unique view → +2.0 reward
- Already seen view → -1.0 reward

### 2. Interestingness Rewards

#### 2a. KL Divergence (coefficient: 1.5, filters only)

**Purpose**: Reward filters that create interesting distribution changes.

**Calculation**:
```python
# Compare column distributions before/after filter
before_dist = value_counts(original) / len(original)
after_dist = value_counts(filtered) / len(filtered)

kl_divergence = KL(after_dist || before_dist)
kl_reward = 1.5 * kl_divergence
```

**Example**: Filtering to a rare subset → High KL → High reward

#### 2b. Compaction Gain (coefficient: 2.0, groups only)

**Purpose**: Reward groupings that meaningfully summarize data.

**Calculation**:
```python
# Information preserved vs size reduction
compaction_gain = (original_entropy - grouped_entropy) / log(compression_ratio)
compaction_reward = 2.0 * compaction_gain
```

**Example**: 1000 rows → 5 meaningful groups → High compaction reward

### 3. Humanity Reward (coefficient: 1.0)

**Purpose**: Ensure coherent exploration following human patterns.

#### 3a. Rule-Based Scoring (35+ handcrafted rules)

**Example Rules**:
- `filter_from_displayed_column`: +0.7 (good: filter on visible columns)
- `humane_columns_group`: +0.4 (good: group on semantic columns)
- `filter_as_first_action`: -1.0 (bad: explore first, then filter)
- `group_on_filtered_column`: -1.0 (bad: redundant grouping)

#### 3b. Snorkel Weak Supervision (learned from 76 human sessions)

**Training**:
1. Collect 76 real analyst exploration sessions from REACT-IDA benchmark
2. Define 245 labeling functions (LFs) that vote on action quality
3. Train Snorkel generative model to combine LF votes
4. Use model to score agent actions during training

**Scoring**:
```python
prob_good = snorkel_model.predict(action_features)
snorkel_score = 2 * (prob_good - 0.5)

# Examples:
# prob_good = 0.9 → score = +0.8 (great!)
# prob_good = 0.1 → score = -0.8 (bad!)
# prob_good = 0.5 → score = 0.0 (neutral)
```

### 4. Penalties

- **Empty display**: -1.0 (action produces no results)
- **Repeated state**: -1.0 (already visited this exact view)
- **Back with no history**: -1.0 (can't go back from start)
- **Invalid filter term** (e.g., `<UNK>`): -20.0 (severe penalty)

### Total Reward Formula

```python
# For Filter/Group actions:
total_reward = (diversity_reward + 
                interestingness_reward + 
                humanity_reward + 
                snorkel_reward) * 0.01  # Scale factor

# For Back actions:
total_reward = snorkel_reward * 0.01  # Humanity/Snorkel only

# Override with penalty if applicable
if is_empty_display or is_repeated_state or is_invalid_action:
    total_reward = penalty * 0.01
```

---

## Training

### Datasets

ATENA supports multiple datasets/schemas. Configure in `Configuration/config.py`:

```python
# Available schemas:
schema = 'NETWORKING'      # Default: network packet analysis (12 fields)
# schema = 'FLIGHTS'       # Flight data analysis
# schema = 'BIG_FLIGHTS'   # Larger flight dataset
# schema = 'WIDE_FLIGHTS'  # Flight data with more columns
# schema = 'WIDE12_FLIGHTS' # Flight data optimized for 12 columns
```

**NETWORKING Dataset** (Default):
- **Source**: REACT-IDA benchmark (Honeynet Project)
- **Size**: 4 datasets, 350-13K rows each
- **Domain**: Cyber-security network traffic analysis
- **Fields**: packet_number, eth_src, eth_dst, ip_src, ip_dst, tcp_srcport, tcp_dstport, highest_layer, length, sniff_timestamp, etc.
- **Use Case**: Detecting security events (malware, hacking, scans)

### Training Script

**Primary Script**: `train_with_decay.py`

```bash
# Recommended: Standard training with decay
python train_with_decay.py --steps 500000 --seed 42 --enable-decay

# Parameters:
#   --steps: Total training steps (500K recommended, 1M for full training)
#   --seed: Random seed for reproducibility
#   --enable-decay: Enable learning rate + clip ratio decay
#   --outdir: Output directory (default: auto-generated timestamp)
#   --resume-from: Resume from checkpoint directory
#   --resume-step: Step number to resume from
```

**What Happens During Training**:

1. **Environment Setup**: Load dataset, initialize vectorized environment
2. **Agent Creation**: Build PPO agent with policy/value networks
3. **Experience Collection**: Agent interacts with environment for 2048 steps
4. **Batch Training**: Update policy/value networks using PPO algorithm
5. **Checkpointing**: Save models every 50K steps
6. **Logging**: Track rewards, action distributions, episode statistics

**Training Output**:
```
results/{timestamp}/
├── trained_model_policy_weights.weights.h5    # Final policy network
├── trained_model_value_weights.weights.h5     # Final value network
├── best_agent_policy_weights.weights.h5       # Best checkpoint
├── checkpoint_step_{N}/                       # Periodic checkpoints
│   ├── policy_weights.weights.h5
│   └── value_weights.weights.h5
├── episode_summary.jsonl                      # Per-episode metrics
└── training_log.txt                           # Detailed logs
```

### Training Hyperparameters

**PPO Algorithm**:
- Update interval: 2048 steps
- Minibatch size: 64
- Epochs per update: 10
- Clip ratio: 0.2 (with decay to 0)
- Learning rate: 3e-4 (with decay to 0)
- Gamma (discount): 0.995
- Lambda (GAE): 0.97

**Network Architecture**:
- Policy network: 51 → 600 → 600 → 949 (tanh activation)
- Value network: 51 → 600 → 600 → 1 (tanh activation)
- Optimizer: Adam
- Initialization: Xavier/Glorot uniform

**Reward Coefficients**:
- Diversity: 2.0
- KL divergence: 1.5
- Compaction: 2.0
- Humanity: 1.0
- Reward scale: 0.01 (applied to all rewards)

---

## Evaluation

### Metrics

#### 1. Tree BLEU-4 / GLEU
**Hierarchical action sequence similarity** with expert reference sessions.

- Measures how similar agent's exploration paths are to human experts
- Accounts for tree structure of back actions
- **Target**: >60% indicates strong performance

#### 2. Action Distribution
**Balance across action types** (back/filter/group).

- Well-trained agents show 20-40% for each action type
- Imbalanced distributions indicate poor learning
- **Target**: No action type >50%

#### 3. Reward Progression
**Learning curve** showing episode rewards over time.

- Should show upward trend
- Convergence after ~100-150 episodes
- **Target**: Mean episode reward >3.0

#### 4. Tree Edit Distance (TED)
**Display tree similarity** between agent and expert sessions.

- Lower is better (fewer edits needed)
- Complements BLEU scores
- **Target**: <50 edits for 12-step sessions

### Evaluation Tools

#### Jupyter Notebooks

**1. Master_Compatible_Evaluation.ipynb**
- Complete evaluation suite
- BLEU/GLEU scoring against expert sessions
- Statistical significance testing
- Action distribution analysis

**2. ATENA_TF_Evaluation.ipynb**
- Quick model assessment
- Per-dataset performance metrics
- Reward component breakdowns
- Learning curve visualizations

**3. evaluate_agent_notebook.ipynb**
- Interactive step-by-step evaluation
- Real-time reward calculation
- Action selection debugging
- Custom dataset testing

#### Command-Line Tools

**generate_session_output.py**
```bash
python generate_session_output.py \
    --model_path results/latest/trained_model \
    --dataset 0 \
    --steps 12 \
    --output session.txt
```

Creates human-readable exploration sessions with:
- Action descriptions
- Reward breakdowns
- Humanity rule activations
- Resulting data views

**evaluate_model.py**
```bash
python evaluate_model.py \
    --model_path results/latest/trained_model \
    --datasets 0 1 2 3 \
    --num_episodes 10
```

Batch evaluation across multiple datasets with statistics.

---

## Notebooks Guide

### Welcome Notebooks - Choose Your Path

ATENA-TF 2 provides **three different welcome notebooks** for different use cases:

| Notebook | Structure | Best For | Key Features |
|----------|-----------|----------|--------------|
| **ATENA-TF-Welcome** | Modern quick start | New users, demos | ✨ Trained agent focus<br>🤖 Interactive recommender<br>📊 Modern formatting |
| **ATENA-TF-Master-Welcome** | **Classic deep dive** | **ATENA-master users**<br>Migration | 🔄 Exact port of original<br>📋 Same action sequences<br>🎯 Detailed explanations |
| **ATENA_TF_Welcome_Colab** | Colab-ready | Cloud users | ☁️ Google Colab optimized<br>📦 Minimal setup |

#### Which Welcome Notebook Should I Use?

**Use ATENA-TF-Welcome.ipynb if:**
- You're new to ATENA and want a quick introduction
- You want to see trained agents in action
- You prefer modern, concise examples

**Use ATENA-TF-Master-Welcome.ipynb if:**
- You're migrating from ATENA-master (ChainerRL version)
- You want the exact same structure as the original notebook
- You need detailed reward component analysis
- You want to understand action vectors in depth
- You prefer analyst view with HTML navigation

**Use ATENA_TF_Welcome_Colab.ipynb if:**
- You want to run ATENA in Google Colab
- You don't want to set up a local environment

### Essential Notebooks

| Notebook | Purpose | Time | Best For |
|----------|---------|------|----------|
| **Master_Compatible_Evaluation** | Full evaluation suite | 20min | Academic evaluation, papers |
| **Live_Recommendations_System** | Interactive recommender | 15min | Demos, UX testing |
| **ATENA_TF_Evaluation** | Quick model assessment | 10min | Development, debugging |

### Visualization Notebooks

| Notebook | Purpose | Time | Best For |
|----------|---------|------|----------|
| **3d_graphs_notebook** | 3D reward visualizations | 5min | Understanding rewards |
| **vldb_demo_graphs** | Publication-quality plots | 5min | Papers, presentations |
| **compare_rewards_notebook** | Multi-run comparison | 15min | Hyperparameter tuning |

### Analysis Notebooks

| Notebook | Purpose | Time | Best For |
|----------|---------|------|----------|
| **evaluate_agent_notebook** | Comprehensive analysis | 20min | Deep debugging |
| **agent_sessions_for_analyst** | Session generation | 15min | User studies |
| **expert_analysis_notebook** | Human session analysis | 15min | Understanding humans |
| **cluster_human_sessions** | Human session clustering | 10min | Initial setup |

### Advanced Notebooks

| Notebook | Purpose | Time | Best For |
|----------|---------|------|----------|
| **atena_snorkel_notebook** | Snorkel LF development | 30min | Weak supervision work |
| **Snorkel_Development** | Advanced Snorkel tuning | 30min | Research |
| **user_study_notebook** | User study replay | 30min | Behavioral research |

---

## Performance Results

### Action Distribution ✅

**Trained ATENA Agent**:
- Back: 27.8% (navigation)
- Filter: 32.5% (exploration)
- Group: 39.7% (aggregation)

**Result**: Well-balanced across all action types → No action monopoly ✅

### Reward Performance ✅

**Training Run Statistics**:
- Mean Episode Reward: 4.5-5.0
- Peak Episode Reward: 8.0+
- Typical Range: -2.0 to +6.0
- Convergence: ~100-150 episodes

**Example High-Quality Session**:
```
Step 1: Filter on eth_src          → Reward: +3.40
Step 2: Group on tcp_dstport       → Reward: +4.26
Step 3: Group on eth_dst           → Reward: +4.49
Step 4: Group on eth_src           → Reward: +2.83
Step 5: Group on highest_layer     → Reward: +3.54
...
Result: Coherent exploration with logical progression ✅
```

### BLEU Evaluation ✅

**Similarity with Expert Sessions**:

| Dataset | Avg BLEU-4 | Peak BLEU-4 | Interpretation |
|---------|-----------|-------------|----------------|
| Dataset 1 | 68.9% | 84.1% | Approaching expert-level |
| Dataset 2 | 65.7% | 75.2% | Strong performance |
| Dataset 3 | 49.0% | 62.4% | Acceptable |
| Dataset 4 | 51.8% | 68.1% | Acceptable |

**Peak Performance**: 84.1% BLEU-4 similarity with human experts ✅

### Training Characteristics ✅

**Stability**:
- ✅ No training crashes
- ✅ Smooth learning curves
- ✅ Convergent policies
- ✅ Balanced action distributions by episode 100

**Efficiency**:
- Episodes to convergence: ~100-150
- Training time: ~30 minutes for 100 episodes (GPU)
- Memory usage: ~8GB peak
- GPU utilization: ~75% average

---

## Component Documentation

### Key Classes

#### `EnhancedATENAEnv` (gym_atena/envs/enhanced_atena_env.py)
**Purpose**: OpenAI Gym environment with complete reward system.

**Key Methods**:
- `reset()`: Initialize new episode with full dataset
- `step(action)`: Execute action, compute reward, return next state
- `compute_enhanced_interestingness_reward()`: KL + compaction
- `compute_diversity_reward_master_exact()`: Min similarity
- `compute_snorkel_humanity_score()`: Weak supervision

#### `PPOAgent` (models/ppo/agent.py)
**Purpose**: Proximal Policy Optimization agent.

**Key Methods**:
- `select_action(state)`: Choose action using policy network
- `train_step()`: Update policy/value networks using PPO
- `compute_gae()`: Generalized Advantage Estimation
- `save_weights() / load_weights()`: Model persistence

#### `ParametricSoftmaxPolicy` (models/ppo/networks.py)
**Purpose**: Policy network with parametric softmax output.

**Architecture**:
```
Input (51D) → Dense(600, tanh) → Dense(600, tanh) → 
├→ Action head (949 logits)
└→ Temperature scaling (β=1.0)
```

#### `HumanityRuleEngine` (gym_atena/envs/enhanced_atena_env.py)
**Purpose**: Apply 35+ handcrafted humanity rules.

**Rule Categories**:
- Filter quality (displayed columns, value validity)
- Group quality (semantic columns, compaction)
- Navigation quality (back timing, subsession coherence)
- Action sequences (filter after group, recursive actions)

---

## Advanced Topics

### Weak Supervision with Snorkel

ATENA uses **Snorkel** to learn from human analyst sessions without manual labels.

**How It Works**:
1. **Collect Human Sessions**: 76 real exploration sessions from REACT-IDA benchmark
2. **Define Labeling Functions (LFs)**: 245 programmatic rules that vote on action quality
3. **Train Generative Model**: Snorkel learns to combine LF votes (accounting for accuracy/correlation)
4. **Score Agent Actions**: Use trained model to provide humanity rewards during RL training

**Pre-trained Models**:
- Located in `snorkel_checkpoints/` (NETWORKING), `snorkel_flights_checkpoints/` (FLIGHTS)
- Files: `GenerativeModel.weights.pkl`, `GenerativeModel.hps.pkl`, `priors.txt`
- Trained on 76 human sessions with 245 LFs

**Customization**:
- Modify LFs in `gym_atena/envs/atena_snorkel_*_lfs.py`
- Retrain using `atena_snorkel_notebook.ipynb`
- Evaluate using `Evaluate_Snorkel_Model.ipynb`

### Custom Datasets

To add a new dataset:

1. **Create Schema** (`gym_atena/data_schemas/your_schema.py`):
```python
class YourSchema:
    schema_name = 'YOUR_DATASET'
    fields = ['field1', 'field2', ...]
    field_types = {'field1': 'str', 'field2': 'int', ...}
    primary_key = 'field1'
    # ... other schema properties
```

2. **Add Data Files** (`gym_atena/your_dataset/`):
```
gym_atena/your_dataset/
├── dataset0.tsv
├── dataset1.tsv
└── ...
```

3. **Configure** (`Configuration/config.py`):
```python
schema = 'YOUR_DATASET'
```

4. **Train Snorkel Model** (optional, for humanity rewards):
- Collect human sessions
- Define domain-specific LFs
- Train using Snorkel notebooks

### Hyperparameter Tuning

Key hyperparameters to tune:

**Reward Coefficients** (`Configuration/config.py`):
```python
kl_coeff = 1.5           # Filter interestingness
compaction_coeff = 2.0   # Group interestingness
diversity_coeff = 2.0    # Exploration diversity
humanity_coeff = 1.0     # Human-likeness
```

**PPO Parameters** (`models/ppo/agent.py`):
```python
learning_rate = 3e-4     # Policy/value learning rate
clip_ratio = 0.2         # PPO clipping parameter
gamma = 0.995            # Discount factor
lambda_ = 0.97           # GAE lambda
```

**Network Architecture** (`models/ppo/networks.py`):
```python
hidden_dims = [600, 600]  # Hidden layer sizes (CRITICAL: don't reduce!)
activation = 'tanh'       # Activation function
```

**Recommendation**: Start with default values. They were systematically optimized and work well.

---

## References

### Original Research

**Paper**: "Next-Step Suggestions for Modern Interactive Data Analysis Platforms"  
Tova Milo and Amit Somech. KDD 2018.  
DOI: https://doi.org/10.1145/3219819.3219848

**REACT-IDA Benchmark**: Real analyst sessions in cybersecurity domain  
Located in: `gym_atena/reactida/`

### Key Technologies

- **TensorFlow 2.16+**: Deep learning framework
- **Keras 3**: Neural network API
- **OpenAI Gym**: RL environment interface
- **Proximal Policy Optimization (PPO)**: RL algorithm (Schulman et al., 2017)
- **Snorkel 0.9+**: Weak supervision framework (Ratner et al., 2017)

### Additional Documentation

- **Migration Documentation**: `ATENA_MIGRATION_DOCUMENTATION.md` (ChainerRL→TensorFlow details)

---

## Contributing

This is a research project. For questions or collaboration:

1. **Understand the system**: Start with `ATENA-TF-Welcome.ipynb`
2. **Read the documentation**: This README + migration doc
3. **Explore the code**: Well-documented Python files
4. **Run experiments**: Try training with different parameters

---

## Acknowledgments

- Original ATENA authors for the groundbreaking research
- REACT-IDA benchmark creators for real analyst session data
- Snorkel team for weak supervision framework
- OpenAI Gym for standardized RL interface

---

**Quick Links**:
- 📓 [Modern Welcome](./Notebooks/ATENA-TF-Welcome.ipynb) | [Classic Welcome](./Notebooks/ATENA-TF-Master-Welcome.ipynb)
- 📊 [Evaluation Notebook](./Notebooks/Master_Compatible_Evaluation.ipynb)
- 🎯 [Training Script](train_with_decay.py)
- 📚 [Migration Docs](ATENA_MIGRATION_DOCUMENTATION.md)
