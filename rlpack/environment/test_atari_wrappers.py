import numpy as np
from rlpack.environment.atari_wrappers import make_atari

import tensorflow as tf

image_pl = tf.placeholder(tf.uint8, [84, 84, 4], "image")
image_pl = tf.to_float(image_pl)

sess = tf.Session()


env = make_atari("AlienNoFrameskip-v4")
s = env.reset()

for i in range(1000):
    a = env.sample_action()
    next_s, r, d, _ = env.step(a)
    # next_s = np.asarray(next_s, dtype=np.uint8)

    print(f"a: {a} \t s: max: {np.max(next_s)} min: {np.min(next_s)} {type(next_s)} \t r: {r}, d: {d}")

    if d is True:
        img = sess.run(image_pl, feed_dict={image_pl: next_s})
        print(f"max: {np.max(img)} min: {np.min(img)}  {next_s.dtype}")
        input()
