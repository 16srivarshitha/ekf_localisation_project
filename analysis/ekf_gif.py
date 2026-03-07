import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy.linalg as la
from PIL import Image
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")
FRAMES_DIR = os.path.join(PLOTS_DIR, "gif_frames")
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

from ekf import EKFLocalizer

#  Load base scenario data 
gt        = np.load(os.path.join(DATA_DIR, "scenario_1", "ground_truth.npy"))
odom      = np.load(os.path.join(DATA_DIR, "scenario_1", "odometry.npy"))
landmarks = np.load(os.path.join(DATA_DIR, "scenario_1", "landmarks.npy"))
obs       = np.load(os.path.join(DATA_DIR, "scenario_1", "observations.npy"))

DT = 0.1

#  Run EKF, store full history 
ekf = EKFLocalizer(
    initial_state         = np.array([0.0, 0.0, 0.0]),
    initial_covariance    = np.eye(3) * 0.1,
    control_noise_cov     = np.diag([0.2**2, 0.1**2]),
    measurement_noise_cov = np.diag([0.5**2, 0.15**2]),
)

ekf_xs = [ekf.mu[0, 0]]
ekf_ys = [ekf.mu[1, 0]]
sigmas = [ekf.Sigma.copy()]

for i in range(1, len(odom)):
    dx    = odom[i, 0] - odom[i-1, 0]
    dy    = odom[i, 1] - odom[i-1, 1]
    v     = np.sqrt(dx**2 + dy**2) / DT
    omega = (odom[i, 2] - odom[i-1, 2]) / DT
    ekf.predict([v, omega], DT)

    for row in obs[obs[:, 0] == i]:
        _, lm_id, r, b = row
        ekf.update([r, b], landmarks[int(lm_id)])

    ekf_xs.append(ekf.mu[0, 0])
    ekf_ys.append(ekf.mu[1, 0])
    sigmas.append(ekf.Sigma.copy())

ekf_xs = np.array(ekf_xs)
ekf_ys = np.array(ekf_ys)

#  Helper: draw covariance ellipse 
def draw_cov_ellipse(ax, x, y, cov2d, n_std=2.0, **kwargs):
    try:
        vals, vecs = la.eigh(cov2d)
        vals  = np.abs(vals)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w = 2 * n_std * np.sqrt(vals[0])
        h = 2 * n_std * np.sqrt(vals[1])
        ell = Ellipse(xy=(x, y), width=w, height=h, angle=angle, **kwargs)
        ax.add_patch(ell)
    except Exception:
        pass

#  Generate frames 
frame_steps = list(range(0, len(gt), 4)) + [len(gt) - 1]
print(f"Generating {len(frame_steps)} frames...")

for fi, step in enumerate(frame_steps):

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-6, 10)
    ax.set_ylim(-3, 13)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, color="#444")
    ax.set_facecolor("#0d0d1a")
    fig.patch.set_facecolor("#0d0d1a")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    ax.tick_params(colors="#aaa")

    # Paths drawn up to current step
    ax.plot(gt[:step+1, 0],   gt[:step+1, 1],   color="#00ff44", linewidth=2.5, label="Ground Truth")
    ax.plot(odom[:step+1, 0], odom[:step+1, 1], color="#ff4444", linewidth=1.5, linestyle="--", label="Odometry")
    ax.plot(ekf_xs[:step+1],  ekf_ys[:step+1],  color="#55aaff", linewidth=2.0, label="EKF Estimate")

    # Landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 1],
               marker="*", s=300, color="#ffcc00", edgecolors="white", zorder=5, linewidth=0.5)
    for j, (lx, ly) in enumerate(landmarks):
        ax.text(lx + 0.2, ly + 0.2, f"L{j}", color="#ffcc00", fontsize=9)

    # Current robot positions
    if step > 0:
        ax.scatter(gt[step, 0],   gt[step, 1],   color="#00ff44", s=90, zorder=7)
        ax.scatter(odom[step, 0], odom[step, 1], color="#ff4444", s=90, zorder=7)
    ax.scatter(ekf_xs[step], ekf_ys[step],
               color="#55aaff", s=120, zorder=8, edgecolors="white", linewidth=1)

    # 2-sigma covariance ellipse around EKF position

    if step > 0:
        draw_cov_ellipse(ax, ekf_xs[step], ekf_ys[step],
                     sigmas[step][:2, :2], n_std=2.0,
                     facecolor="#55aaff", alpha=0.20,
                     edgecolor="#88ccff", linewidth=1.5)

    # Start marker
    ax.scatter(0, 0, color="white", s=100, marker="D", zorder=9)

    ax.legend(loc="upper left", fontsize=9,
              facecolor="#1a1a2e", labelcolor="white", edgecolor="#555")
    ax.set_xlabel("X (m)", color="#ccc", fontsize=11)
    ax.set_ylabel("Y (m)", color="#ccc", fontsize=11)
    ax.set_title(
        f"EKF Localization  |  t = {step * DT:.1f}s  (step {step}/{len(gt)-1})",
        color="white", fontsize=12, pad=8,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(FRAMES_DIR, f"frame_{fi:04d}.png"),
        dpi=90, facecolor=fig.get_facecolor(),
    )
    plt.close()

print(f"  Frames saved to: {FRAMES_DIR}")

#  Assemble GIF 
print("Assembling GIF...")
frame_files   = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.png")))
frames        = [Image.open(f).convert("RGBA") for f in frame_files]
durations     = [80] * len(frames)
durations[-1] = 1500  # hold on last frame

out = os.path.join(PLOTS_DIR, "ekf_localization.gif")
frames[0].save(
    out,
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=0,
    optimize=False,
)
print(f"  GIF saved: {out}  ({os.path.getsize(out)/1024/1024:.1f} MB)")