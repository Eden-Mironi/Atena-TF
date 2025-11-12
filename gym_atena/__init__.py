import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Remove the old registration
# register(
#     id='ATENAld-v0',
#     entry_point='gym_ianna.envs:ATENAEnv',
# )

# Add the new registration
register(
    id='ATENAcont-v0',
    entry_point='gym_atena.envs.enhanced_atena_env:EnhancedATENAEnv',
    max_episode_steps=1000,
    reward_threshold=1.0,
    nondeterministic=True,
)

# Add this to verify registration
print("Registering ATENAcont-v0 environment")