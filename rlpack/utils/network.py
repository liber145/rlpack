import tensorflow as tf
import numpy as np


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def gaussian_likelihood(x, mu, log_std, EPS=1e-8):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def diagonal_gaussian_kl(mu0, log_std0, mu1, log_std1, EPS=1e-8):
    """
    tf symbol for mean KL divergence between two batches of diagonal gaussian distributions,
    where distributions are specified by means and log stds.
    (https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Multivariate_normal_distributions)
    """
    var0, var1 = tf.exp(2 * log_std0), tf.exp(2 * log_std1)
    pre_sum = 0.5*(((mu1 - mu0)**2 + var0)/(var1 + EPS) - 1) + log_std1 - log_std0
    all_kls = tf.reduce_sum(pre_sum, axis=1)
    return tf.reduce_mean(all_kls)


def mlp_gaussian_policy(x, a, hidden_sizes, activation=tf.nn.relu, output_activation=None):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi, mu, log_std


def mlp_gaussian_policy2(x, a, hidden_sizes, activation=tf.nn.relu, output_activation=None, LOG_STD_MIN=-20, LOG_STD_MAX=2):
    """高斯分布中的action在负无穷到正无穷之间，而大部分环境的action是有界的。因此，
    1. 使用tanh将action压缩到有界空间，
    2. 使用雅克比修改求导规则。
    """
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)
    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)

    mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
    return mu, pi, logp_pi


def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


def discrete_sparse_policy(x, a, n_act, hidden_sizes, activation=tf.nn.relu, output_activation=None, top=3):
    """输入状态x，输出top动作构成的sparse分布。
    """
    # 计算logits.
    net = mlp(x, hidden_sizes, activation, output_activation)
    logits = tf.layers.dense(net, n_act, activation=None)

    # 计算top动作概率。
    top_v, top_idx = tf.nn.top_k(logits, k=top, sorted=False)

    kth = tf.reduce_min(top_v, axis=1, keepdims=True)
    top_logits = tf.where(logits >= kth, logits, -1e5*tf.ones_like(logits))
    probs = tf.nn.softmax(top_logits)

    # top_sm = tf.nn.softmax(top_v)

    # # 构建新坐标。
    # p_shape = tf.shape(logits)
    # p_row_idx = tf.tile(tf.range(p_shape[0])[:, tf.newaxis], (1, top))
    # scatter_idx = tf.stack([p_row_idx, top_idx], axis=-1)
    # probs = tf.scatter_nd(scatter_idx, top_sm, p_shape)


    


    # 计算概率。
    batchsize = tf.shape(x)[0]
    p_idx = tf.stack([tf.range(batchsize), a], axis=1)
    selected_p = tf.gather_nd(probs, p_idx)
    print(">>>>>>>> selected_p:", selected_p)


    # 采样。
    dist = tf.distributions.Categorical(probs=probs)
    newa = dist.sample()
    print(">>>>>>>>>> newa:", newa)
    print(">>>>>>>>>> p:", probs)
    print(">>>>>>>>>> selected_p:", selected_p)
    return newa, probs, selected_p, logits


def discrete_alpha(p1, p2, alpha=0.5, EPS=1e-8):
    r = (p1+EPS) / (p2 + EPS)
    pre_sum = tf.clip_by_value( (r**alpha - r*alpha + alpha - 1) / (alpha*(alpha-1)) * p2, -40, 40 )
    all_alphas = tf.reduce_sum(pre_sum, axis=1)
    return tf.reduce_mean(all_alphas)

def discrete_policy(x, a, n_act, hidden_sizes, activation=tf.nn.relu, output_activation=None):
    net = mlp(x, hidden_sizes, activation, output_activation)
    logits = tf.layers.dense(net, n_act, activation=None)

    # 动作抽样。
    sampled_a = tf.random.categorical(logits, num_samples=1)

    # 概率。
    probs = tf.nn.softmax(logits)
    p_idx = tf.stack([tf.range(tf.shape(x)[0]), a], axis=-1)
    selected_p = tf.gather_nd(probs, p_idx)

    return tf.squeeze(sampled_a, axis=1), probs, selected_p

def discrete_kl(p1, p2, EPS=1e-8):
    pre_sum = tf.log(p1 / (p2+EPS)) * p1 
    all_kls = tf.reduce_sum(pre_sum, axis=-1)
    return tf.reduce_mean(all_kls)