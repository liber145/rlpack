import numpy as np
from rlpack.environment.atari_wrappers import make_atari, make_ram_atari

import tensorflow as tf

# image_pl = tf.placeholder(tf.uint8, [84, 84, 4], "image")
# image_pl = tf.to_float(image_pl)

# sess = tf.Session()


# env = make_atari("AlienNoFrameskip-v4")
# s = env.reset()

# for i in range(1000):
#     a = env.sample_action()
#     next_s, r, d, _ = env.step(a)

#     print(f"a: {a} \t s: max: {np.max(next_s)} min: {np.min(next_s)} {type(next_s)} \t r: {r}, d: {d}")

#     if d is True:
#         img = sess.run(image_pl, feed_dict={image_pl: next_s})
#         print(f"max: {np.max(img)} min: {np.min(img)}  {next_s.dtype}")
#         input()


env = make_ram_atari("Breakout-ramNoFrameskip-v4")
env.seed(1)
s = env.reset()

print(env.dim_observation)
print(env.dim_action)
input()

all_r = 0
for i in range(10000):
    a = env.sample_action()
    next_s, r, d, info = env.step(a)
    all_r += r

    print(f"iter {i} a: {a} \t s: {s.shape} max: {np.max(next_s)} min: {np.min(next_s)} {type(next_s)} \t r: {r}, all_r: {all_r} d: {d} info:{info}  {env.observation_space.shape}")
    print(next_s)
    if d is True:
        input()
