import libtmux
import time

server = libtmux.Server()
session = server.new_session(session_name="atari")

window = session.new_window()
pane = window.panes[0]
pane.send_keys("cd ~/rlpack")
time.sleep(0.3)
pane.send_keys("conda activate gpu")
time.sleep(0.3)
pane.send_keys(f"python examples/distributed_ramatari/run_dqn.py --gpu 0")

pane = pane.split_window(vertical=True)
time.sleep(0.3)
pane.send_keys("ssh cpu1")
time.sleep(1.3)
pane.send_keys("cd ~/rlpack")
time.sleep(0.3)
pane.send_keys("conda activate gpu")
time.sleep(0.3)
pane.send_keys(f"python examples/distributed_ramatari/run_env.py --ip gpu1")

pane = pane.split_window(vertical=True)
time.sleep(0.3)
pane.send_keys("ssh cpu1")
time.sleep(1.3)
pane.send_keys("cd ~/rlpack")
time.sleep(0.3)
pane.send_keys("conda activate gpu")
time.sleep(0.3)
pane.send_keys(f"python examples/distributed_ramatari/run_env.py --ip gpu1")

