import jax.numpy as jnp
import flax.linen as nn

class RNNEncoder(nn.Module):
    """
    RNN encoder for MFRL baseline:
      - (a) LayerNorm on CNN embedding z
      - (b) Linear map to hidden_size
      - (c) ReLU activation
      - Update hidden state via GRUCell
      - ReLU on GRU output y
    Input:
      carry: [B, hidden_size]
      z:     [B, embed_dim] (e.g. 8192)
    Output:
      new_carry: [B, hidden_size]
      y:         [B, hidden_size]
    """
    hidden_size: int

    @nn.compact
    def __call__(self, carry: jnp.ndarray, z: jnp.ndarray):
        # (a) Layer normalization on embedding
        x = nn.LayerNorm()(z)
        # (b) Linear mapping to hidden_size
        x = nn.Dense(self.hidden_size)(x)
        # (c) ReLU activation
        x = nn.relu(x)
        # RNN update: GRU cell
        new_carry, y = nn.GRUCell(self.hidden_size)(carry, x)
        # Final ReLU on GRU output
        y = nn.relu(y)
        return new_carry, y
