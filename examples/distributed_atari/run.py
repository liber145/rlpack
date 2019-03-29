import libtmux
import time

server = libtmux.Server()
session = server.new_session(session_name="atari")
window = session.new_window()


pane = window.panes[0]
pane.send_keys("cd /data/lyj/rlpack")
time.sleep(0.3)
pane.send_keys("conda activate gpu")
time.sleep(0.3)
pane.send_keys(f"python examples/distributed_atari/run_dqn.py --gpu 0")


pane_list = list()

for _ in range(3):
    window = session.new_window()
    pane1 = window.panes[0]
    pane1.send_keys("1")

    pane2 = pane1.split_window(vertical=True)
    pane2.send_keys("2")

    pane3 = pane2.split_window(vertical=False)
    pane3.send_keys("3")

    pane4 = pane2.split_window(vertical=True)
    pane4.send_keys("4")

    pane5 = pane3.split_window(vertical=True)
    pane5.send_keys("5")

    pane6 = pane1.split_window(vertical=False)
    pane6.send_keys("6")

    pane7 = pane6.split_window(vertical=True)
    pane7.send_keys("7")

    pane8 = pane1.split_window(vertical=True)
    pane8.send_keys("8")

    pane_list.extend([pane1, pane2, pane3, pane4, pane5, pane6, pane7, pane8])

for pane in pane_list:
    time.sleep(0.3)
    pane.send_keys("ssh cpu1")
    time.sleep(0.3)
    pane.send_keys("cd /data/lyj/rlpack")
    time.sleep(0.3)
    pane.send_keys("conda activate gpu")
    time.sleep(0.3)
    pane.send_keys(f"python examples/distributed_atari/run_env.py --ip gpu1")
