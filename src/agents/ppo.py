import argparse
import os
import sys
import numpy as np



import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
import distrax
import gymnax
from typing import Sequence, NamedTuple, Any, Dict
from craftax.craftax_env import make_craftax_env_from_name
import wandb

from jax.tree_util import tree_map



import argparse, os, time
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManagerOptions, CheckpointManager
from flax.training import orbax_utils


# Part 2: import new model components
from models.impala_cnn import ImpalaCNN
from models.rnn import RNNEncoder

#from wrappers import LogWrapper, AutoResetEnvWrapper, BatchEnvWrapper
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from craftax.craftax_env import make_craftax_env_from_name



class ActorCritic(nn.Module):
    action_dim: int
    hidden_size: int = 256

    @nn.compact
    def __call__(self, carry, obs, done):
        # Input: Image O_t, last hidden state h_{t-1}, parameters Φ
        # Output: action a_t, value v_t, new hidden state h_t
        # z_t = ImpalaCNN_Φ(O_t)
        # h_t, y_t = RNN_Φ([h_{t-1}, z_t])
        # a_t ~ π_Φ([y_t, z_t])
        # v_t = V_Φ([y_t, z_t])

        # reset hidden state where done
        carry = jnp.where(done[:, None], jnp.zeros_like(carry), carry)
        # CNN encoding
        z = ImpalaCNN()(obs)                   # [B,8192]
        # RNN update
        carry, y = RNNEncoder(self.hidden_size)(carry, z)  # y: [B,256]
        # concatenate features: [y_t, z_t]
        features = jnp.concatenate([y, z], axis=-1)  # [B,8448]

        # Actor and critic architecture:
        # Finally, the CNN output z_t and the RNN output y_t are concatenated,
        # resulting in the 8448-dimensional embedding input shared by the actor and critic.
        # Actor network: (a) LayerNorm; (b) FC→2048→ReLU; (c) two residual blocks→ReLU;
        # (d) LayerNorm; (e) FC→action_dim logits.
        # Critic network: (a) LayerNorm; (b) FC->2048→ReLU; (c) two residual blocks→ReLU;
        # (d) LayerNorm; (e) FC→1 value.

        # Actor head
        x = nn.LayerNorm()(features)            # (a) layer normalization
        x = nn.Dense(2048)(x); x = nn.relu(x)   # (b) FC + ReLU
        for _ in range(2):                      # (c) two residual blocks
            r = nn.Dense(2048)(x); r = nn.relu(r)
            r = nn.Dense(2048)(r)
            x = nn.relu(x + r)
        x = nn.LayerNorm()(x)                   # (d) layer normalization
        logits = nn.Dense(self.action_dim)(x)   # (e) final FC to action logits
        pi = distrax.Categorical(logits=logits)

        # Critic head
        v = nn.LayerNorm()(features)            # (a) layer normalization
        v = nn.Dense(2048)(v); v = nn.relu(v)   # (b) FC + ReLU
        for _ in range(2):                      # (c) two residual blocks
            r2 = nn.Dense(2048)(v); r2 = nn.relu(r2)
            r2 = nn.Dense(2048)(r2)
            v = nn.relu(v + r2)
        v = nn.LayerNorm()(v)                   # (d) layer normalization
        value = nn.Dense(1)(v).squeeze(-1)      # (e) final FC to scalar value

        return carry, pi, value

####################


