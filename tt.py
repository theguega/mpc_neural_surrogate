import h5py
import numpy as np

episodes = []
with h5py.File("data/robot_mpc_dataset.h5", "r") as f:
    keys = np.array(sorted(f["episodes"].keys()))
    for ep in keys:
        grp = f["episodes"][ep]
        states = grp["states"][:]
        targets = grp["targets"][:]
        actions = grp["actions"][:]
        episodes.append((states, targets, actions))

print(len(episodes))
