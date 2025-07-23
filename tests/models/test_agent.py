import pytest
import jax
import jax.numpy as jnp
import distrax
import craftax

# --- Imports for code being tested ---
# Note: Adjust these paths to match your project structure.
from agents.ppo import ActorCritic, make_train, Transition
from models.impala_cnn import ImpalaCNN
from models.rnn import RNNEncoder

# --- Constants for testing ---
IMAGE_SHAPE = (63, 63, 3)
ACTION_DIM = 17  # As used in Craftax
HIDDEN_SIZE = 256

# ================================================================
# == 1. Unit Tests for the ActorCritic Module
# ================================================================

@pytest.mark.parametrize("batch_size", [1, 4])
def test_actor_critic_shape_and_finite(batch_size):
    """
    Tests the forward pass of the ActorCritic module for correct output shapes
    and ensures no NaN/Inf values are produced.
    """
    model = ActorCritic(action_dim=ACTION_DIM, hidden_size=HIDDEN_SIZE)
    rng = jax.random.PRNGKey(0)

    # Dummy inputs matching the expected dimensions
    obs = jnp.zeros((batch_size,) + IMAGE_SHAPE)
    carry = jnp.zeros((batch_size, HIDDEN_SIZE))
    done = jnp.zeros((batch_size,), dtype=bool)
    
    # Initialize and apply the model
    variables = model.init(rng, carry, obs, done)
    new_carry, pi, value = model.apply(variables, carry, obs, done)

    # --- Assertions ---
    # Check shapes
    assert new_carry.shape == (batch_size, HIDDEN_SIZE)
    assert isinstance(pi, distrax.Categorical)
    assert pi.logits.shape == (batch_size, ACTION_DIM)
    assert value.shape == (batch_size,)

    # Check for numerical stability
    assert jnp.all(jnp.isfinite(new_carry))
    assert jnp.all(jnp.isfinite(pi.logits))
    assert jnp.all(jnp.isfinite(value))

@pytest.mark.parametrize("batch_size", [1, 4])
def test_actor_critic_gradients(batch_size):
    """
    Tests that gradients are computable and non-zero for the ActorCritic parameters.
    """
    model = ActorCritic(action_dim=ACTION_DIM, hidden_size=HIDDEN_SIZE)
    rng = jax.random.PRNGKey(1)

    # Dummy inputs and initialization
    obs = jax.random.normal(rng, (batch_size,) + IMAGE_SHAPE)
    carry = jax.random.normal(rng, (batch_size, HIDDEN_SIZE))
    done = jnp.zeros((batch_size,), dtype=bool)
    variables = model.init(rng, carry, obs, done)

    # Define a loss function that takes the full variables dictionary
    # but still allows differentiation w.r.t. params.
    def loss_fn(params):
        # Re-construct the variables dictionary inside the loss function
        # This keeps the function signature compatible with jax.grad
        full_vars = {'params': params, 'batch_stats': variables['batch_stats']}
        new_carry, pi, value = model.apply(full_vars, carry, obs, done)
        return jnp.sum(new_carry) + jnp.sum(pi.logits) + jnp.sum(value)

    # Compute gradients only on the 'params' part
    grads = jax.grad(loss_fn)(variables['params'])

    # --- Assertions ---
    param_leaves, param_treedef = jax.tree_util.tree_flatten(variables['params'])
    grad_leaves, grad_treedef = jax.tree_util.tree_flatten(grads)
    
    assert grad_treedef == param_treedef
    
    for g in grad_leaves:
        assert jnp.all(jnp.isfinite(g))
        assert jnp.any(g != 0)

# ================================================================
# == 2. Integration Test (Smoke Test) for make_train
# ================================================================

# def test_make_train_smoke_test():
#     config = {
#         "ENV_NAME": "Craftax-Classic-Pixels-v1",
#         "NUM_ENVS": 2,
#         "NUM_STEPS": 8,
#         "TOTAL_TIMESTEPS": 32,
#         "UPDATE_EPOCHS": 1,
#         "NUM_MINIBATCHES": 1,
#         "LR": 3e-4,
#         "GAMMA": 0.99,
#         "GAE_LAMBDA": 0.95,
#         "CLIP_EPS": 0.2,
#         "VF_COEF": 0.5,
#         "ENT_COEF": 0.01,
#         "MAX_GRAD_NORM": 0.5,
#         "PPO_TARGET_ALPHA": 0.95,
#         "LAYER_SIZE": 256,    # <— this was missing
#     }

#     rng = jax.random.PRNGKey(42)
#     train_fn = make_train(config)
#     train_jit = jax.jit(train_fn)
#     final = train_jit(rng)

#     # Because `train_jit` returns the runner-state tuple:
#     ts = final[0]          # TrainState is the first element
#     assert ts.step > 0



@pytest.mark.parametrize("batch_size", [1])  # only one repeat needed
def test_make_train_smoke_test(batch_size):
    """
    Smoke test for make_train:
    - Compiles and runs without crashing
    - Returns a 9‐tuple runner‐state
    - TrainState.step has advanced
    - Moving‐average stats are finite
    """
    config = {
        "ENV_NAME": "Craftax-Classic-Pixels-v1",
        "NUM_ENVS": 2,
        "NUM_STEPS": 8,
        "TOTAL_TIMESTEPS": 32,
        "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES": 1,
        "LR": 3e-4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.01,
        "MAX_GRAD_NORM": 0.5,
        "PPO_TARGET_ALPHA": 0.95,
        "LAYER_SIZE": 256,
        # disable any external logging
        "USE_WANDB": False,
    }

    rng = jax.random.PRNGKey(42)
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)
    final = train_jit(rng)

    # runner‐state should be a 9‐tuple:
    # (TrainState, env_state, obs, done, hidden, key, step, mu_targ, sig_targ)
    assert isinstance(final, tuple)
    assert len(final) == 9

    ts, env_state, obs, done, hidden, key, step, mu_targ, sig_targ = final

    # 1) optimizer step advanced
    assert ts.step > 0

    # 2) moving‐average stats are finite scalars
    assert jnp.isscalar(mu_targ) and jnp.isfinite(mu_targ)
    assert jnp.isscalar(sig_targ) and jnp.isfinite(sig_targ)
