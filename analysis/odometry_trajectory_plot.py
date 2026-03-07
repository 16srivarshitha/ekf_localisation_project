import numpy as np
import matplotlib.pyplot as plt

# Load data
gt = np.load("ground_truth.npy")
odom = np.load("odometry.npy")
landmarks = np.load("landmarks.npy")

# Extract coordinates
gt_x = gt[:,0]
gt_y = gt[:,1]

odom_x = odom[:,0]
odom_y = odom[:,1]

plt.figure(figsize=(8,8))

# Ground truth path
plt.plot(gt_x, gt_y,
         color='green',
         linewidth=3,
         label="Ground Truth")

# Odometry path
plt.plot(odom_x, odom_y,
         linestyle='--',
         color='red',
         linewidth=2,
         label="Noisy Odometry")

# Landmarks
plt.scatter(landmarks[:,0],
            landmarks[:,1],
            marker='*',
            s=250,
            color='blue',
            edgecolors='black',
            label="Landmarks")

# Label landmarks
for i,(lx,ly) in enumerate(landmarks):
    plt.text(lx+0.15, ly+0.15, f"L{i}",
             fontsize=11,
             color='blue')

# Start point
plt.scatter(gt_x[0], gt_y[0],
            color='black',
            s=120,
            label="Start")

# End point
plt.scatter(gt_x[-1], gt_y[-1],
            color='purple',
            s=120,
            label="End")

plt.xlabel("X(m)", fontsize=12)
plt.ylabel("Y(m)", fontsize=12)

plt.title("Robot Trajectory: Ground Truth vs Odometry",
          fontsize=14)

plt.legend(fontsize=11)
plt.grid(True)
plt.axis("equal")

plt.tight_layout()

plt.savefig("trajectory_plot.png", dpi=300)

plt.show()
