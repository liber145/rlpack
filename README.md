目前代码，离散动作空间的算法在CartPole-v0和CartPole-v1上进行检测，连续动作空间的算法暂未确定在哪里检测。

DQN:
1. CartPole-v0通过。对discount敏感，0.9结果比较稳定；0.95结果不稳定。
2. CartPole-v1未通过。可达到475以上，但是不稳定，迅速衰减，未再上升。需要调参。learning rate改成线性衰减，还是不行。加大discount从0.9到0.99之后，问题解决。但是不稳定，迅速衰减没了。


TODO：
- [ ]  state归一化
- [ ]  多环境benchmark
- [ ]  learning rate scheduler
- [ ]  模型断点续跑

