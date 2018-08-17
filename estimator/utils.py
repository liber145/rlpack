import math
import numpy as np


def unfold(traj):
    return map(np.array, zip(*traj))


def gen_batch(trajectories, batch_size):
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(
        np.concatenate, zip(*map(unfold, trajectories)))

    n_sample = state_batch.shape[0]
    index = np.arange(n_sample)
    np.random.shuffle(index)

    for i in range(math.ceil(n_sample / batch_size)):
        span_index = slice(i*batch_size, min((i+1)*batch_size, n_sample))
        span_index = index[span_index]
        yield state_batch[span_index, :], action_batch[span_index], reward_batch[span_index], next_state_batch[span_index, :], done_batch[span_index]


def process_traj(traj, batch_size, discount=0.99):
    """traj构成：sarsd"""
    n_sample = len(traj)

    # span_reward 表示从traj中最后一个state到当前state之间的discount reward总和。
    span_reward = [0]
    for transition in reversed(traj):
        span_reward.append(transition[2] + span_reward[-1] * discount)
    span_reward.pop(0)
    span_reward_batch = np.array(span_reward[::-1])[:, np.newaxis]

    # Let last state as base state.
    last_state = traj[-1][3]
    last_done = traj[-1][4]

    if last_state.ndim == 1:
        last_state_batch = np.tile(last_state, (n_sample, 1))
    elif last_state.ndim == 3:
        last_state_batch = np.tile(last_state, (n_sample, 1, 1, 1))
    else:
        assert False

    last_done_batch = np.tile(last_done, (n_sample, 1))

    # Get state, action ,reward, next state, done batch.
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(
        np.array, zip(*traj))
    reward_batch = reward_batch[:, np.newaxis]
    done_batch = done_batch[:, np.newaxis]
    action_batch = action_batch[:, np.newaxis] \
        if action_batch.ndim == 1 else action_batch

    return state_batch, action_batch, reward_batch, next_state_batch, done_batch, span_reward_batch, last_state_batch, last_done_batch


def trajectories_to_batch(trajectories, batch_size, discount):
    state_batch, action_batch, reward_batch, next_state_batch, done_batch, span_reward_batch, last_state_batch, last_done_batch = map(
        np.concatenate, zip(*map(lambda x: process_traj(x, batch_size, discount), trajectories)))

    return {"state": state_batch,
            "action": action_batch,
            "reward": reward_batch,
            "nextstate": next_state_batch,
            "done": done_batch,
            "spanreward": span_reward_batch,
            "laststate": last_state_batch,
            "lastdone": last_done_batch
            }


def generator(data_batch, batch_size=64):
    n_sample = data_batch["state"].shape[0]
    index = np.arange(n_sample)
    np.random.shuffle(index)

    for i in range(math.ceil(n_sample / batch_size)):
        span_index = slice(i*batch_size, min((i+1)*batch_size, n_sample))
        span_index = index[span_index]
        # yield {"state": data_batch["state"][span_index, :],
        #        "action": data_batch["action"][span_index, :],
        #        "reward": data_batch["reward"][span_index, :],
        #        "nextstate": data_batch["nextstate"][span_index, :],
        #        "done": data_batch["done"][span_index, :],
        #        "spanreward": data_batch["spanreward"][span_index, :],
        #        "laststate": data_batch["laststate"][span_index, :],
        #        "lastdone": data_batch["lastdone"][span_index, :],
        #        "oldmu": data_batch["oldmu"][span_index, :]
        #        }
        yield {key: item[span_index, :] for key, item in data_batch.items()}
