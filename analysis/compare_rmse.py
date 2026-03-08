"""
compare_rmse.py — RMSE bar chart + per-step error curves for all three scenarios.

"""

import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

from ekf import EKFLocalizer

DT = 0.1

SCENARIOS = {
    "scenario_1": {
        "label": "Scenario 1\n(Low Noise)",
        "M": np.diag([0.005**2, 0.002**2]),
        "R": np.diag([0.05**2,  0.01**2]),
    },
    "scenario_2": {
        "label": "Scenario 2\n(High Noise)",
        "M": np.diag([0.2**2, 0.1**2]),
        "R": np.diag([0.5**2, 0.15**2]),
    },
    "scenario_3": {
        "label": "Scenario 3\n(Moderate Noise)",
        "M": np.diag([0.05**2, 0.02**2]),
        "R": np.diag([0.2**2,  0.05**2]),
    },
}


def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


odom_rmse_vals, ekf_rmse_vals, labels = [], [], []
odom_errors, ekf_errors = {}, {}

for folder, cfg in SCENARIOS.items():
    gt        = np.load(os.path.join(DATA_DIR, folder, "ground_truth.npy"))
    odom      = np.load(os.path.join(DATA_DIR, folder, "odometry.npy"))
    landmarks = np.load(os.path.join(DATA_DIR, folder, "landmarks.npy"))
    obs       = np.load(os.path.join(DATA_DIR, folder, "observations.npy"))

    odom_error = np.sqrt((gt[:, 0] - odom[:, 0])**2 + (gt[:, 1] - odom[:, 1])**2)
    odom_rmse  = np.sqrt(np.mean(odom_error**2))

    ekf = EKFLocalizer(
        initial_state         = np.array([0.0, 0.0, 0.0]),
        initial_covariance    = np.eye(3) * 0.1,
        control_noise_cov     = cfg["M"],
        measurement_noise_cov = cfg["R"],
    )

    ekf_path = [[ekf.mu[0, 0], ekf.mu[1, 0]]]
    for i in range(1, len(odom)):
        dx    = odom[i, 0] - odom[i-1, 0]
        dy    = odom[i, 1] - odom[i-1, 1]
        v     = np.sqrt(dx**2 + dy**2) / DT
        # wrap before dividing
        omega = normalize_angle(odom[i, 2] - odom[i-1, 2]) / DT
        ekf.predict([v, omega], DT)

        for row in obs[obs[:, 0] == i]:
            _, lm_id, r, b = row
            ekf.update([r, b], landmarks[int(lm_id)])

        ekf_path.append([ekf.mu[0, 0], ekf.mu[1, 0]])

    ekf_path  = np.array(ekf_path)
    ekf_error = np.sqrt((gt[:, 0] - ekf_path[:, 0])**2 + (gt[:, 1] - ekf_path[:, 1])**2)
    ekf_rmse  = np.sqrt(np.mean(ekf_error**2))

    reduction = (odom_rmse - ekf_rmse) / odom_rmse * 100
    print(f"{folder}  Odom RMSE: {odom_rmse:.4f} m  |  EKF RMSE: {ekf_rmse:.4f} m  |  Reduction: {reduction:.1f}%")

    odom_rmse_vals.append(odom_rmse)
    ekf_rmse_vals.append(ekf_rmse)
    labels.append(cfg["label"])
    odom_errors[folder] = odom_error
    ekf_errors[folder]  = ekf_error

#  Plot 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ["steelblue", "darkorange", "seagreen"]

for idx, folder in enumerate(SCENARIOS.keys()):
    short = f"S{idx+1}"
    axes[0].plot(odom_errors[folder], linestyle="--", color=colors[idx],
                 alpha=0.6, label=f"Odometry — {short}")
    axes[0].plot(ekf_errors[folder],  linestyle="-",  color=colors[idx],
                 label=f"EKF — {short}")

axes[0].set_title("Position Error Over Time", fontsize=13)
axes[0].set_xlabel("Time step")
axes[0].set_ylabel("Position Error (m)")
axes[0].legend(fontsize=8)
axes[0].grid(True)

x     = np.arange(len(labels))
width = 0.35
bars1 = axes[1].bar(x - width/2, odom_rmse_vals, width, label="Odometry RMSE", color="red",  alpha=0.8)
bars2 = axes[1].bar(x + width/2, ekf_rmse_vals,  width, label="EKF RMSE",      color="blue", alpha=0.8)

for bar in list(bars1) + list(bars2):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}",
                 ha="center", va="bottom", fontsize=9)

axes[1].set_title("RMSE: Odometry vs EKF", fontsize=13)
axes[1].set_xlabel("Scenario")
axes[1].set_ylabel("RMSE (m)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontsize=9)
axes[1].legend()
axes[1].grid(axis="y")

plt.suptitle("EKF Localization Performance Across Scenarios", fontsize=14, fontweight="bold")
plt.tight_layout()
out = os.path.join(PLOTS_DIR, "compare_rmse.png")
plt.savefig(out, dpi=300)
plt.close()
print(f"\nSaved: {out}")