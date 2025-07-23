import pytest
import jax
import jax.numpy as jnp
from models.rnn import RNNEncoder

# Constants matching ImpalaCNN output size
EMBED_DIM = 8 * 8 * 128

@pytest.mark.parametrize("batch_size,hidden_size", [(1, 256), (2, 256), (3, 128)])
def test_rnn_shape_and_no_nan(batch_size, hidden_size):
    model = RNNEncoder(hidden_size=hidden_size)
    rng = jax.random.PRNGKey(0)
    # Dummy inputs
    z = jnp.zeros((batch_size, EMBED_DIM), dtype=jnp.float32)
    carry = jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)
    variables = model.init(rng, carry, z)
    new_carry, y = model.apply(variables, carry, z)
    # Shapes
    assert new_carry.shape == (batch_size, hidden_size)
    assert y.shape == (batch_size, hidden_size)
    # No NaNs or Infs
    assert jnp.all(jnp.isfinite(new_carry)), "new_carry contains NaN/Inf"
    assert jnp.all(jnp.isfinite(y)), "y contains NaN/Inf"

@pytest.mark.parametrize("batch_size,hidden_size", [(1,256), (2,256)])
def test_rnn_gradients(batch_size, hidden_size):
    model = RNNEncoder(hidden_size=hidden_size)
    rng = jax.random.PRNGKey(1)
    # Random inputs
    z = jax.random.normal(rng, (batch_size, EMBED_DIM))
    carry = jax.random.normal(rng, (batch_size, hidden_size))
    variables = model.init(rng, carry, z)

    # Gradient w.r.t. z
    def forward_z(z_in):
        _, y_out = model.apply(variables, carry, z_in)
        return jnp.sum(y_out)
    grad_fn_z = jax.grad(forward_z)
    grads_z = grad_fn_z(z)
    assert grads_z.shape == z.shape
    assert jnp.all(jnp.isfinite(grads_z))
    assert jnp.any(grads_z != 0)

    # Gradient w.r.t. carry
    def forward_c(carry_in):
        new_carry, _ = model.apply(variables, carry_in, z)
        return jnp.sum(new_carry)
    grad_fn_c = jax.grad(forward_c)
    grads_c = grad_fn_c(carry)
    assert grads_c.shape == carry.shape
    assert jnp.all(jnp.isfinite(grads_c))
    assert jnp.any(grads_c != 0)