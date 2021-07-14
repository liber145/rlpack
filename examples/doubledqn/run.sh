CUDA_VISIBLE_DEVICES=0 python test.py --env_name CartPole-v1 --discount 0.99 &
CUDA_VISIBLE_DEVICES=1 python test.py --env_name CartPole-v1 --discount 0.995 &
CUDA_VISIBLE_DEVICES=2 python test.py --env_name CartPole-v0 --discount 0.99 &
CUDA_VISIBLE_DEVICES=3 python test.py --env_name CartPole-v0 --discount 0.995 &