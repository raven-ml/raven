"""Reference test: 2D correlated Gaussian with BlackJAX NUTS and HMC."""
import jax
import jax.numpy as jnp
import blackjax

true_mean = jnp.array([3.0, -1.0])
true_cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])
cov_inv = jnp.linalg.inv(true_cov)

def log_prob(x):
    d = x - true_mean
    return -0.5 * d @ cov_inv @ d

# --- HMC ---
key = jax.random.key(42)
inv_mass = jnp.ones(2)
hmc_alg = blackjax.hmc(log_prob, step_size=0.1, inverse_mass_matrix=inv_mass, num_integration_steps=20)
state = hmc_alg.init(jnp.zeros(2))
step_fn = jax.jit(hmc_alg.step)

samples_hmc = []
for i in range(4000):  # 1000 warmup + 3000 samples
    key = jax.random.fold_in(key, i)
    state, info = step_fn(key, state)
    if i >= 1000:
        samples_hmc.append(state.position)

samples_hmc = jnp.stack(samples_hmc)
print(f"HMC (no adaptation, fixed step_size=0.1):")
print(f"  mean = {jnp.mean(samples_hmc, axis=0)}")
print(f"  var  = {jnp.var(samples_hmc, axis=0)}")
print()

# --- NUTS ---
key = jax.random.key(42)
nuts_alg = blackjax.nuts(log_prob, step_size=0.1, inverse_mass_matrix=inv_mass)
state = nuts_alg.init(jnp.zeros(2))
step_fn = jax.jit(nuts_alg.step)

samples_nuts = []
for i in range(4000):
    key = jax.random.fold_in(key, i)
    state, info = step_fn(key, state)
    if i >= 1000:
        samples_nuts.append(state.position)

samples_nuts = jnp.stack(samples_nuts)
print(f"NUTS (no adaptation, fixed step_size=0.1):")
print(f"  mean = {jnp.mean(samples_nuts, axis=0)}")
print(f"  var  = {jnp.var(samples_nuts, axis=0)}")
print()

# --- NUTS with window adaptation ---
key = jax.random.key(42)
warmup = blackjax.window_adaptation(blackjax.nuts, log_prob, num_steps=1000)
key, warmup_key = jax.random.split(key)
(adapted_state, adapted_params), _ = warmup.run(warmup_key, jnp.zeros(2))
print(f"Window adaptation results:")
print(f"  step_size = {adapted_params['step_size']}")
print(f"  inv_mass  = {adapted_params['inverse_mass_matrix']}")

nuts_adapted = blackjax.nuts(log_prob, **adapted_params)
step_fn = jax.jit(nuts_adapted.step)
state = adapted_state

samples_adapted = []
for i in range(3000):
    key = jax.random.fold_in(key, i)
    state, info = step_fn(key, state)
    samples_adapted.append(state.position)

samples_adapted = jnp.stack(samples_adapted)
print(f"NUTS (with window adaptation):")
print(f"  mean = {jnp.mean(samples_adapted, axis=0)}")
print(f"  var  = {jnp.var(samples_adapted, axis=0)}")
