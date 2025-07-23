import pytest
import jax
import jax.numpy as jnp
from models.impala_cnn import ImpalaCNN

@pytest.mark.parametrize("batch_size", [1, 2])
def test_impala_cnn_output_shape_and_no_nan(batch_size):
    model = ImpalaCNN()
    rng = jax.random.PRNGKey(0)
    x = jnp.zeros((batch_size, 63, 63, 3), dtype=jnp.float32)
    variables = model.init(rng, x)
    out = model.apply(variables, x)
    # Expect output shape [B, 8*8*128]
    assert out.shape == (batch_size, 8*8*128)
    # No NaNs or Infs in output
    assert jnp.all(jnp.isfinite(out)), "Output contains NaNs or Infs"

@pytest.mark.parametrize("batch_size", [1, 2])
def test_impala_cnn_different_inputs(batch_size):
    model = ImpalaCNN()
    rng = jax.random.PRNGKey(42)
    x1 = jnp.zeros((batch_size, 63, 63, 3), dtype=jnp.float32)
    x2 = jnp.ones((batch_size, 63, 63, 3), dtype=jnp.float32)
    variables = model.init(rng, x1)
    out1 = model.apply(variables, x1)
    out2 = model.apply(variables, x2)
    # Outputs for different inputs should differ
    assert not jnp.allclose(out1, out2), "Model outputs identical for different inputs"

@pytest.mark.parametrize("batch_size", [1, 2])
def test_impala_cnn_gradients(batch_size):
    model = ImpalaCNN()
    rng = jax.random.PRNGKey(123)
    # Use a random input to test gradients
    x = jax.random.normal(rng, (batch_size, 63, 63, 3), dtype=jnp.float32)
    variables = model.init(rng, x)

    def forward(inp):
        # sum outputs to get a scalar
        return jnp.sum(model.apply(variables, inp))

    # Compute gradient w.r.t. input
    grad_fn = jax.grad(forward)
    grads = grad_fn(x)
    # Gradient should have same shape as input
    assert grads.shape == x.shape
    # Gradients should be finite
    assert jnp.all(jnp.isfinite(grads)), "Input gradients contain NaNs or Infs"
    # There should be at least one non-zero gradient
    assert jnp.any(grads != 0), "Gradients are all zero, check computation"
