from rlpack.environment import ClassicControlWrapper


env = ClassicControlWrapper("CartPole-v1", 1)
s = env.reset()


for i in range(10000):
    a = env.sample_action()
    next_s, r, d, info = env.step(a)

    print(f"iter: {i} s: {s} a: {a} r: {r} d: {d[0]} info: {info}")
    if d[0]:
        print("info:", info)
        input()
