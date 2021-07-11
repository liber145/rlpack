LogDir=cartpole_v1_log
mkdir $LogDir

# Train.
python -u classic_control.py \
--env CartPole-v1 \
--alg DQN \
--discount 0.99 \
--update_freq 1 \
--num_train_steps 1000000 \
--num_warmup_steps 10000 \
--lr 1e-3 \
--momentum 0.9 \
--weight_decay 1e-4 \
--max_grad_norm 1.0 \
--batch_size 64 \
--log_dir $LogDir \
--delay_update_freq 4 \
|& tee -a $LogDir/cartpole_v1.log 

# Plot.
python plot_result.py \
--env cartplot_v1 \
--log_dir $LogDir \
--result_path $LogDir/result.pk
