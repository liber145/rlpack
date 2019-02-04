import libtmux
import time

server = libtmux.Server()
session = server.new_session(session_name="atari")
cdwp = "cd ~/Programs/rlpack"
activate_env = "conda activate py36"
export_path = "export PYTHONPATH=."


env_names = ["Alien", "Amidar", "BankHeist", "BattleZone", "Centipede", "ChopperCommand", "DemonAttack", "DoubleDunk"]

for i in range(len(env_names)):
    gpu_id = i // 2
    envname = env_names[i]

    window = session.new_window()
    pane = window.panes[0]
    pane.send_keys(cdwp)
    time.sleep(0.3)
    pane.send_keys(activate_env)
    time.sleep(0.3)
    pane.send_keys(export_path)
    time.sleep(0.3)
    pane.send_keys(f"python examples/atari/run_dqn_atari.py --env_name {envname}-ramNoFrameskip-v4 --gpu {gpu_id}")
