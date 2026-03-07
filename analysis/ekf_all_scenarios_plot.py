import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- Path setup (works on any machine) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ekf.py lives in the same analysis/ folder
from ekf import EKFLocalizer

# --- Scenario definitions ---
scenarios = {
    "scenario_1": {
        "label": "Scenario 1 (Low Noise)",
        "M": np.diag([0.005**2, 0.002**2]),
        "R": np.diag([0.05**2,  0.01**2]),
    },
    "scenario_2": {
        "label": "Scenario 2 (High Noise)",
        "M": np.diag([0.2**2, 0.1**2]),
        "R": np.diag([0.5**2, 0.15**2]),
    },
    "scenario_3": {
        "label": "Scenario 3 (Moderate Noise)",
        "M": np.diag([0.05**2, 0.02**2]),
        "R": np.diag([0.2**2,  0.05**2]),
    },
}

DT = 0.1

for folder, cfg in scenarios.items():

    print(f"Processing: {folder} — {cfg['label']}")

    # --- Load data ---
    gt        = np.load(os.path.join(DATA_DIR, folder, "ground_truth.npy"))
    odom      = np.load(os.path.join(DATA_DIR, folder, "odometry.npy"))
    landmarks = np.load(os.path.join(DATA_DIR, folder, "landmarks.npy"))
    obs       = np.load(os.path.join(DATA_DIR, folder, "observations.npy"))

    # --- Run EKF ---
    ekf = EKFLocalizer(
        initial_state         = np.array([0.0, 0.0, 0.0]),
        initial_covariance    = np.eye(3) * 0.1,
        control_noise_cov     = cfg["M"],
        measurement_noise_cov = cfg["R"],
    )

    ekf_path = [[ekf.mu[0, 0], ekf.mu[1, 0]]]

    for i in range(1, len(odom)):

        # Predict
        dx    = odom[i, 0] - odom[i-1, 0]
        dy    = odom[i, 1] - odom[i-1, 1]
        v     = np.sqrt(dx**2 + dy**2) / DT
        omega = (odom[i, 2] - odom[i-1, 2]) / DT
        ekf.predict([v, omega], DT)

        # Update with all landmark observations at this step
        for row in obs[obs[:, 0] == i]:
            _, lm_id, r, b = row
            ekf.update([r, b], landmarks[int(lm_id)])

        ekf_path.append([ekf.mu[0, 0], ekf.mu[1, 0]])

    ekf_path = np.array(ekf_path)

    # --- Plot ---
    plt.figure(figsize=(8, 8))

    plt.plot(odom[:, 0], odom[:, 1],
             linestyle="--", color="red", linewidth=2, label="Noisy Odometry")
    plt.plot(gt[:, 0], gt[:, 1],
             color="green", linewidth=3, label="Ground Truth")
    plt.plot(ekf_path[:, 0], ekf_path[:, 1],
             color="blue", linewidth=2, label="EKF Estimate")

    plt.scatter(landmarks[:, 0], landmarks[:, 1],
                marker="*", s=250, color="blue", edgecolors="black", zorder=5, label="Landmarks")
    for j, (lx, ly) in enumerate(landmarks):
        plt.text(lx + 0.15, ly + 0.15, f"L{j}", fontsize=10, color="blue")

    plt.scatter(gt[0, 0],  gt[0, 1],  c="black",  s=120, zorder=6, label="Start")
    plt.scatter(gt[-1, 0], gt[-1, 1], c="purple", s=120, zorder=6, label="End")

    plt.xlabel("X position (m)", fontsize=12)
    plt.ylabel("Y position (m)", fontsize=12)
    plt.title(f"Trajectory Comparison — {cfg['label']}", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    out = os.path.join(PLOTS_DIR, f"trajectory_{folder}.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved: {out}")