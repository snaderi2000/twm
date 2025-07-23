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

class Transition(NamedTuple):
    """A single step transition for PPO."""
    done: jnp.ndarray       # shape [B]
    action: jnp.ndarray     # shape [B]
    value: jnp.ndarray      # shape [B]
    reward: jnp.ndarray     # shape [B]
    log_prob: jnp.ndarray   # shape [B]
    obs: jnp.ndarray        # shape [B, *obs_shape]
    info: Any               # env-specific info

def make_train(config: Dict):
    # Compute derived config values
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    # default PPO target moving-average discount
    alpha = config.get("PPO_TARGET_ALPHA", 0.95)

    # Build and wrap the environment
    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config.get("USE_OPTIMISTIC_RESETS", False)
    )
    params = env.default_params
    env = LogWrapper(env)
    if config.get("USE_OPTIMISTIC_RESETS", False):
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config.get("OPTIMISTIC_RESET_RATIO", config["NUM_ENVS"]), config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    # Linear learning-rate schedule
    def linear_schedule(step):
        frac = (
            1.0
            - (step // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # Initialize actor-critic network
        action_dim = env.action_space(params).n
        hidden_size = config["LAYER_SIZE"]
        network = ActorCritic(action_dim, hidden_size=hidden_size)

        # Dummy inputs for initialization
        obs_shape = env.observation_space(params).shape
        dummy_obs = jnp.zeros((config["NUM_ENVS"], *obs_shape), jnp.float32)
        dummy_done = jnp.zeros((config["NUM_ENVS"],), jnp.bool_)
        init_carry = jnp.zeros((config["NUM_ENVS"], hidden_size), jnp.float32)

        rng, init_rng = jax.random.split(rng)
        net_params = network.init(init_rng, init_carry, dummy_obs, dummy_done)

        # Optimizer
        if config.get("ANNEAL_LR", False):
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=net_params,
            tx=tx,
        )

        # Reset environment and RNN state
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng, params)
        hstate = init_carry

        # Outer moving-average stats
        mu_targ = jnp.zeros((), dtype=jnp.float32)
        sig_targ = jnp.ones((), dtype=jnp.float32)

        def _update_step(carry, _):
            ts, es, last_obs, last_done, hidden, key, step, mu_targ, sig_targ = carry

            # Rollout trajectory
            def _env_step(state, _):
                ts, es, lo, ld, h, k, st = state
                k, subk = jax.random.split(k)
                new_h, pi, val = network.apply(ts.params, h, lo, ld)
                a = pi.sample(seed=subk)
                lp = pi.log_prob(a)
                k, subk = jax.random.split(k)
                nxt_obs, nxt_es, rew, nxt_done, info = env.step(subk, es, a, params)
                trans = Transition(done=ld, action=a, value=val, reward=rew, log_prob=lp, obs=lo, info=info)
                return (ts, nxt_es, nxt_obs, nxt_done, new_h, k, st), trans

            (ts, es, obs, done, hidden, key, step), traj = jax.lax.scan(
                _env_step,
                (ts, es, last_obs, last_done, hidden, key, step), None, config["NUM_STEPS"]
            )

            # Bootstrap final value
            _, _, boot_val = network.apply(ts.params, hidden, obs, done)

            # Compute GAE and raw returns
            def _gae_fn(c, t):
                gae, nv, nd = c
                delta = t.reward + config["GAMMA"] * nv * (1-nd) - t.value
                new_gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1-nd) * gae
                return (new_gae, t.value, t.done), new_gae

            (_, advs) = jax.lax.scan(_gae_fn, (jnp.zeros_like(boot_val), boot_val, done), traj, reverse=True)
            returns = advs + traj.value  # [T, B]

            # Advantage normalization (batch)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            # Update moving-average stats & normalize targets
            batch_mu  = returns.mean()
            batch_sig = returns.std()
            mu_targ   = alpha * mu_targ   + (1-alpha) * batch_mu
            sig_targ  = alpha * sig_targ  + (1-alpha) * batch_sig
            ret_norm  = (returns - mu_targ) / (sig_targ + 1e-8)

            # PPO update epochs & minibatches (scan over epochs)
            def _epoch_fn(state, _):
                ts, h0, traj_b, adv_b, ret_b, k = state
                k, pk = jax.random.split(k)
                perm = jax.random.permutation(pk, config["NUM_ENVS"])
                traj_sh = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=1), traj_b)
                adv_sh = jnp.take(adv_b, perm, axis=1)
                ret_sh = jnp.take(ret_b, perm, axis=1)
                def reshape(x):
                    T,B = x.shape[:2]; mb=config["NUM_MINIBATCHES"]; per=B//mb
                    return jnp.swapaxes(x.reshape(T,mb,per,*x.shape[2:]),0,1)
                traj_mb = jax.tree_util.tree_map(reshape, traj_sh)
                adv_mb  = reshape(adv_sh)
                ret_mb  = reshape(ret_sh)
                def _mb(st, batch):
                    trajx, advx, retx = batch
                    def loss_fn(params):
                        def scan_step(c, inp):
                            o,d = inp; h_new,pi_t,v_t = network.apply(params,c,o,d)
                            return h_new,(pi_t,v_t)
                        _, (pi_seq,v_seq) = jax.lax.scan(scan_step, h0, (trajx.obs, trajx.done))
                        logp = pi_seq.log_prob(trajx.action)
                        ratio = jnp.exp(logp - trajx.log_prob)
                        eps = config["CLIP_EPS"]
                        a1 = ratio*advx; a2=jnp.clip(ratio,1-eps,1+eps)*advx
                        aloss = -jnp.mean(jnp.minimum(a1,a2))
                        v_old = trajx.value
                        v_clip = v_old + (v_seq - v_old).clip(-eps,eps)
                        vloss = 0.5*jnp.mean(jnp.maximum((v_seq-retx)**2,(v_clip-retx)**2))
                        ent = pi_seq.entropy().mean()
                        return aloss + config["VF_COEF"]*vloss - config["ENT_COEF"]*ent
                    grads = jax.grad(loss_fn)(st.params)
                    return st.apply_gradients(grads=grads), None
                ts, _ = jax.lax.scan(_mb, ts, (traj_mb, adv_mb, ret_mb))
                return (ts,h0,traj_b,adv_b,ret_b,k), None

            init_epoch = (ts, hidden, traj, advs, ret_norm, key)
            (ts, hidden, *_), _ = jax.lax.scan(_epoch_fn, init_epoch, None, config["UPDATE_EPOCHS"])

            return (ts, es, obs, done, hidden, key, step, mu_targ, sig_targ), None

        # run outer updates
        rng, lk = jax.random.split(rng)
        init = (train_state, env_state, obs, jnp.zeros((config["NUM_ENVS"]),bool), hstate, lk, 0, mu_targ, sig_targ)
        final_carry, _ = jax.lax.scan(_update_step, init, None, config["NUM_UPDATES"])
        return final_carry

    return train

# def make_train(config: Dict):
#     # Compute derived config values
#     config["NUM_UPDATES"] = (
#         config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
#     )
#     config["MINIBATCH_SIZE"] = (
#         config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
#     )

#     # Build and wrap the environment
#     env = make_craftax_env_from_name(
#         config["ENV_NAME"], not config.get("USE_OPTIMISTIC_RESETS", False)
#     )
#     params = env.default_params
#     env = LogWrapper(env)
#     if config.get("USE_OPTIMISTIC_RESETS", False):
#         env = OptimisticResetVecEnvWrapper(
#             env,
#             num_envs=config["NUM_ENVS"],
#             reset_ratio=min(config.get("OPTIMISTIC_RESET_RATIO", config["NUM_ENVS"]), config["NUM_ENVS"]),
#         )
#     else:
#         env = AutoResetEnvWrapper(env)
#         env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

#     # Linear learning-rate schedule
#     def linear_schedule(step):
#         frac = (
#             1.0
#             - (step // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
#             / config["NUM_UPDATES"]
#         )
#         return config["LR"] * frac

#     def train(rng):
#         # Initialize actor-critic network
#         action_dim = env.action_space(params).n
#         hidden_size = config["LAYER_SIZE"]
#         network = ActorCritic(action_dim, hidden_size=hidden_size)

#         # Dummy inputs for initialization
#         obs_shape = env.observation_space(params).shape
#         dummy_obs = jnp.zeros((config["NUM_ENVS"], *obs_shape), jnp.float32)
#         dummy_done = jnp.zeros((config["NUM_ENVS"],), jnp.bool_)
#         init_carry = jnp.zeros((config["NUM_ENVS"], hidden_size), jnp.float32)

#         rng, init_rng = jax.random.split(rng)
#         net_params = network.init(init_rng, init_carry, dummy_obs, dummy_done)

#         # Optimizer
#         if config.get("ANNEAL_LR", False):
#             tx = optax.chain(
#                 optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
#                 optax.adam(learning_rate=linear_schedule, eps=1e-5),
#             )
#         else:
#             tx = optax.chain(
#                 optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
#                 optax.adam(config["LR"], eps=1e-5),
#             )
#         train_state = TrainState.create(
#             apply_fn=network.apply,
#             params=net_params,
#             tx=tx,
#         )

#         # Reset environment and RNN state
#         rng, reset_rng = jax.random.split(rng)
#         obs, env_state = env.reset(reset_rng, params)
#         hstate = init_carry

#         # Single update: rollout + PPO update
#         def _update_step(carry, _):
#             ts, es, last_obs, last_done, hidden, key, step = carry

#             # Rollout trajectory
#             def _env_step(state, _):
#                 ts, es, lo, ld, h, k, st = state
#                 k, subk = jax.random.split(k)
#                 new_h, pi, val = network.apply(ts.params, h, lo, ld)
#                 a = pi.sample(seed=subk)
#                 lp = pi.log_prob(a)

#                 k, subk = jax.random.split(k)
#                 nxt_obs, nxt_es, rew, nxt_done, info = env.step(subk, es, a, params)

#                 trans = Transition(
#                     done=ld,
#                     action=a,
#                     value=val,
#                     reward=rew,
#                     log_prob=lp,
#                     obs=lo,
#                     info=info,
#                 )
#                 return (ts, nxt_es, nxt_obs, nxt_done, new_h, k, st), trans

#             (ts, es, obs, done, hidden, key, step), traj = jax.lax.scan(
#                 _env_step,
#                 (ts, es, last_obs, last_done, hidden, key, step),
#                 None,
#                 config["NUM_STEPS"],
#             )

#             # Bootstrap final value
#             _, _, boot_val = network.apply(ts.params, hidden, obs, done)

#             # Compute GAE and returns
#             def _gae_fn(c, t):
#                 gae, nv, nd = c
#                 delta = t.reward + config["GAMMA"] * nv * (1-nd) - t.value
#                 new_gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1-nd) * gae
#                 return (new_gae, t.value, t.done), new_gae

#             (_, advs) = jax.lax.scan(
#                 _gae_fn,
#                 (jnp.zeros_like(boot_val), boot_val, done),
#                 traj,
#                 reverse=True,
#                 unroll=16,
#             )
#             # Compute raw returns
#             returns = advs + traj.value
#             # Advantage normalization
#             advs = (advs - jnp.mean(advs)) / (jnp.std(advs) + 1e-8)
#             # Target standardization (batch)
#             ret_norm = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-8)

#             # PPO update epochs & minibatches
#             def _epoch_fn(state, _):
#                 ts, h0, traj_b, adv_b, ret_b, k = state
#                 k, pk = jax.random.split(k)
#                 perm = jax.random.permutation(pk, config["NUM_ENVS"])

#                 # Shuffle but keep h0 the same
#                 traj_sh = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=1), traj_b)
#                 adv_sh = jnp.take(adv_b, perm, axis=1)
#                 ret_sh = jnp.take(ret_b, perm, axis=1)

#                 # Reshape helper
#                 def reshape_arr(x):
#                     T, B = x.shape[:2]
#                     mb = config["NUM_MINIBATCHES"]
#                     per = B // mb
#                     return jnp.swapaxes(x.reshape(T, mb, per, *x.shape[2:]), 0, 1)

#                 # Reshape each field of the traj NamedTuple and arrays
#                 traj_mb = jax.tree_util.tree_map(reshape_arr, traj_sh)
#                 adv_mb = reshape_arr(adv_sh)
#                 ret_mb = reshape_arr(ret_sh)

#                 # Minibatch update
#                 def _mb_update(st, batch):
#                     trajx, advx, retx = batch

#                     def loss_fn(params):
#                         # Run the network over the trajectory via scan
#                         def scan_step(carry, inp):
#                             obs_t, done_t = inp
#                             h_new, pi_t, v_t = network.apply(params, carry, obs_t, done_t)
#                             return h_new, (pi_t, v_t)

#                         # Scan over time dimension to get sequences of pi and v
#                         _, (pi_seq, v_seq) = jax.lax.scan(
#                             scan_step,
#                             h0,
#                             (trajx.obs, trajx.done),
#                             length=trajx.obs.shape[0],
#                         )
#                         # Compute log-probs, policy loss
#                         logp_seq = pi_seq.log_prob(trajx.action)
#                         ratio = jnp.exp(logp_seq - trajx.log_prob)
#                         eps = config["CLIP_EPS"]
#                         s1 = ratio * advx
#                         s2 = jnp.clip(ratio, 1 - eps, 1 + eps) * advx
#                         actor_loss = -jnp.mean(jnp.minimum(s1, s2))

#                         # Value loss with clipping
#                         v_old = trajx.value
#                         v_clip = v_old + (v_seq - v_old).clip(-eps, eps)
#                         loss1 = (v_seq - retx) ** 2
#                         loss2 = (v_clip - retx) ** 2
#                         value_loss = 0.5 * jnp.mean(jnp.maximum(loss1, loss2))

#                         # Entropy bonus
#                         entropy = pi_seq.entropy().mean()
#                         return actor_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy

#                     grads = jax.grad(loss_fn)(st.params)
#                     st = st.apply_gradients(grads=grads)
#                     return st, None

#                 # Run minibatch updates, carrying TrainState directly
#                 ts, _ = jax.lax.scan(
#                     _mb_update,
#                     ts,
#                     (traj_mb, adv_mb, ret_mb),
#                 )
#                 # Return updated TrainState and RNN carry
#                 return (ts, h0, traj_b, adv_b, ret_b, k), None

#             init_epoch = (ts, hidden, traj, advs, returns, key)
#             (ts, hidden, *_), _ = jax.lax.scan(
#                 _epoch_fn, init_epoch, None, config["UPDATE_EPOCHS"]
#             )

#             return (ts, es, obs, done, hidden, key, step), None

#         # Run all updates
#         rng, loop_key = jax.random.split(rng)
#         init = (
#             train_state,
#             env_state,
#             obs,
#             jnp.zeros((config["NUM_ENVS"]), dtype=bool),
#             hstate,
#             loop_key,
#             0,
#         )
#         final_state, _ = jax.lax.scan(
#             _update_step, init, None, config["NUM_UPDATES"]
#         )

#         # Return only the final runner_state
#         return final_state

#     return train


# def run_train(args):
#     # Upper‐case config dict
#     config = {k.upper(): v for k, v in vars(args).items()}

#     # Initialize Weights & Biases if requested
#     if config.get("USE_WANDB", False):
#         wandb.init(
#             project=config["WANDB_PROJECT"],
#             entity=config["WANDB_ENTITY"],
#             config=config,
#             name=f"{config['ENV_NAME']}-PPO_RNN-{int(config['TOTAL_TIMESTEPS'] // 1e6)}M",
#         )

#     # Split RNG for repeats
#     rng = jax.random.PRNGKey(config["SEED"])
#     rngs = jax.random.split(rng, config["NUM_REPEATS"])

#     # JIT + VMAP the training function
#     train_fn = make_train(config)
#     train_jit = jax.jit(train_fn)
#     train_vmap = jax.vmap(train_jit)

#     # Run and time
#     t0 = time.time()
#     out = train_vmap(rngs)  # out.shape = (NUM_REPEATS, ... runner‐state tuple ...)
#     t1 = time.time()
#     print(f"Time to run experiment: {t1 - t0:.1f}s")
#     print(f"SPS: {config['TOTAL_TIMESTEPS'] / (t1 - t0):.0f}")

#     # Save the first repeat's policy if requested
#     if config.get("USE_WANDB", False) and config.get("SAVE_POLICY", False):
#         def _save_network(rs_index, dir_name):
#             # out[rs_index] is the final runner‐state tuple for that repeat
#             runner_state = out[rs_index]
#             # The first element of runner_state is the TrainState
#             train_state = runner_state[0]
#             orbax_checkpointer = PyTreeCheckpointer()
#             options = CheckpointManagerOptions(max_to_keep=1, create=True)
#             path = os.path.join(wandb.run.dir, dir_name)
#             manager = CheckpointManager(path, orbax_checkpointer, options)
#             save_args = orbax_utils.save_args_from_target(train_state)
#             manager.save(config["TOTAL_TIMESTEPS"], train_state,
#                          save_kwargs={"save_args": save_args})
#             print(f"Saved policy to {path}")

#         _save_network(0, "policies")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env_name", type=str, default="Craftax-Classic-Pixels-v1")
#     parser.add_argument("--num_envs", type=int, default=48)
#     parser.add_argument("--num_steps", type=int, default=96)
#     parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e9)
#     parser.add_argument("--lr", type=float, default=4.5e-4)
#     parser.add_argument("--update_epochs", type=int, default=4)
#     parser.add_argument("--num_minibatches", type=int, default=8)
#     parser.add_argument("--gamma", type=float, default=0.925)
#     parser.add_argument("--gae_lambda", type=float, default=0.625)
#     parser.add_argument("--clip_eps", type=float, default=0.2)
#     parser.add_argument("--ent_coef", type=float, default=0.01)
#     parser.add_argument("--vf_coef", type=float, default=1.0)
#     parser.add_argument("--max_grad_norm", type=float, default=0.5)
#     parser.add_argument("--anneal_lr", action=argparse.BooleanOptionalAction, default=True)
#     parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=False)
#     parser.add_argument("--wandb_project", type=str, default=None)
#     parser.add_argument("--wandb_entity", type=str, default=None)
#     parser.add_argument("--save_policy", action=argparse.BooleanOptionalAction, default=False)
#     parser.add_argument("--num_repeats", type=int, default=1)
#     parser.add_argument("--layer_size", type=int, default=256)
#     parser.add_argument("--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=False)
#     parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

#     args, rest = parser.parse_known_args()
#     if rest:
#         raise ValueError(f"Unknown args: {rest}")

#     # If the user disabled JIT, wrap in disable_jit context
#     if args.jit:
#         run_train(args)
#     else:
#         with jax.disable_jit():
#             run_train(args)
