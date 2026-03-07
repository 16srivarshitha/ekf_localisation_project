import numpy as np
import matplotlib.pyplot as plt
import sys

scenario_path = sys.argv[1]

gt = np.load(f"{scenario_path}/ground_truth.npy")
odom = np.load(f"{scenario_path}/odometry.npy")
landmarks = np.load(f"{scenario_path}/landmarks.npy")

plt.figure(figsize=(8,8))

plt.plot(odom[:,0], odom[:,1],
         linestyle='--',
         color='red',
         linewidth=3,
         label="Odometry")

plt.plot(gt[:,0], gt[:,1],
         color='green',
         linewidth=3,
         label="Ground Truth")

plt.scatter(landmarks[:,0], landmarks[:,1],
            marker='*',
            s=250,
            color='blue',
            label="Landmarks")

plt.scatter(gt[0,0], gt[0,1], c='black', s=120, label="Start")
plt.scatter(gt[-1,0], gt[-1,1], c='purple', s=120, label="End")

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title(f"Trajectory Plot: {scenario_path}")

plt.legend()
plt.grid(True)
plt.axis("equal")


plt.savefig(f"{scenario_path}_trajectory.png", dpi=300)

plt.show()
