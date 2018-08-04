
### 实现的算法
- DQN
- DoubleDQN
- AveDQN
- SoftDQN
- DistDQN
- PolicyGradient
- DDPG
- TRPO
- PPO
- AdvantageActorCritic

```bash
python main.py --env_name "Reacher-v2" --model ppo --result_path ./results/ppo --n_env 4
```


### 依赖
- tensorflow1.9
- gym

### 组成
三个部分，Estimator，Middleware，Environment
- Estimator构建算法，包括搭建网络，更新策略。
- Middleware：存取中间SARS数据，负责通信。
- Environment：包装游戏环境，主要是gym，即step-reset这种离散模式。

### 其他
掐断程序的正确姿势：Ctrl-C
