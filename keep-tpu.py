import time
import jax
import jax.numpy as jnp

print("TPU:", jax.devices("tpu"))

if len(jax.devices("tpu")) < 8:
    raise RuntimeError("JAX does not see all 8 TPU devices!")

N = 4096

x = jnp.ones((8, N, N), dtype=jnp.bfloat16)
y = jnp.ones((8, N, N), dtype=jnp.bfloat16)

@jax.pmap
def work(a, b):
    x = a @ b
    x = x @ b
    x = x @ b
    return x

# Compile
work(x, y).block_until_ready()

print("TPU v5e-8 active for 60 seconds...")

start = time.time()

while time.time() - start < 60:
    work(x, y).block_until_ready()

print("Done.")
