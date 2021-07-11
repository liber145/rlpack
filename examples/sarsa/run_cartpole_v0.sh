LogDir=cartpole_v0_log
mkdir -p $LogDir

# Train.
python -u classic_control.py \
--env CartPole-v0 \
--alg SARSA \
--discount 0.99 \
--update_every_n_episode 4 \
--num_train_steps 1000000 \
--num_warmup_steps 10000 \
--lr 1e-3 \
--momentum 0.9 \
--weight_decay 1e-4 \
--max_grad_norm 1.0 \
--batch_size 64 \
--log_dir $LogDir \
|& tee -a $LogDir/cartpole_v0.log 

# Plot.
python plot_result.py \
--env cartpole_v0 \
--log_dir $LogDir \
--result_path $LogDir/result.pk

