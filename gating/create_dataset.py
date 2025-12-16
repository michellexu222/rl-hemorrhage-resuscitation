import sys
sys.path.insert(0, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\env")

import os
import numpy as np
import dill as pickle
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from hemorrhage_env import HemorrhageEnv

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

dataset_path = os.path.join(script_dir, "train_data_bv.csv")
if not os.path.exists(dataset_path):
    with open(dataset_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bv1", "bv2", "bv3", "severity", "label"])

patient_paths = os.listdir(os.path.join(parent_dir, "configs", "patient_configs"))
low_sev = [("liver", 0.05), ("liver", 0.1), ("liver", 0.15), ("spleen", 0.5), ("spleen", 0.6), ("spleen", 0.7),
           ("spleen", 0.8)]
high_sev = [("liver", 0.2), ("liver", 0.25), ("liver", 0.3), ("spleen", 0.35), ("liver", 0.4), ("spleen", 0.9),
           ("spleen", 1.0)]

for config in patient_paths:
    env = HemorrhageEnv(state_file=os.path.join(parent_dir, "configs", "patient_configs", config))

    # 0
    for hem in low_sev:
        obs, info = env.reset(organ=hem[0], severity=hem[1])
        severity = info["hem"][1]
        organ = info["hem"][0]

        with open(dataset_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                info["bv1"],
                info["bv2"],
                info["bv3"],
                severity,
                0
            ])
    # 1
    for hem in high_sev:
        obs, info = env.reset(organ=hem[0], severity=hem[1])
        severity = info["hem"][1]
        organ = info["hem"][0]

        with open(dataset_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                info["bv1"],
                info["bv2"],
                info["bv3"],
                severity,
                1
            ])
