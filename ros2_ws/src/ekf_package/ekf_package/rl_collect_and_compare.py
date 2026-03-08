"""
rl_collect_and_compare.py
--------------------------
Step 1: Subscribes to /odometry/filtered (robot_localization output)
        and /odom (raw odometry) simultaneously while simulation runs.
        Saves collected data to rl_collected.csv

Step 2: Loads rl_collected.csv + your ekf_log.csv,
        regenerates ground truth, computes RMSE table and 4 plots.

-----------
Terminal 1:  ros2 run ekf_package ekf_node
Terminal 2:  ros2 run robot_localization ekf_node --ros-args --params-file /ros2_ws/src/ekf_package/rl_config.yaml
Terminal 3:  ros2 run ekf_package simulation_publisher
Terminal 4:  python3 /ros2_ws/src/ekf_package/ekf_package/rl_collect_and_compare.py --collect
Terminal 4:  python3 /ros2_ws/src/ekf_package/ekf_package/rl_collect_and_compare.py --compare
"""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

#  Shared parameters 
DT        = 0.1
STEPS     = 300
V         = 1.0
OMEGA     = 0.2
MAX_RANGE = 5.0
SV, SO    = 0.05, 0.02   # odometry noise (matches your simulation.py)

LANDMARKS = np.array([
    [3.0, 2.0], [6.0, 4.0], [2.0, 7.0],
    [8.0, 1.0], [5.0, 8.0],
])

EKF_LOG_PATH  = '/ros2_ws/src/ekf_log.csv'
RL_COLLECT_PATH = '/ros2_ws/src/rl_collected.csv'
OUT_DIR       = '/ros2_ws/src/'


def normalize_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def motion_model(state, v, omega, dt):
    x, y, theta = state
    return np.array([
        x + v * np.cos(theta) * dt,
        y + v * np.sin(theta) * dt,
        normalize_angle(theta + omega * dt)
    ])


# 
# MODE 1: --collect   (run while ROS simulation is live)
# 
def collect():
    """
    Subscribes to /odometry/filtered and /odom.
    Records x, y at each message stamp.
    Runs until STEPS messages received from /odometry/filtered.
    """
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry

    class Collector(Node):
        def __init__(self):
            super().__init__('rl_collector')
            self.rl_data   = []
            self.odom_data = []
            self.create_subscription(Odometry, '/odometry/filtered', self._rl_cb,   10)
            self.create_subscription(Odometry, '/odom',              self._odom_cb, 10)
            self.get_logger().info('Collecting /odometry/filtered and /odom ...')

        def _rl_cb(self, msg):
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            self.rl_data.append((len(self.rl_data), x, y))
            if len(self.rl_data) % 50 == 0:
                self.get_logger().info(f'  rl collected: {len(self.rl_data)} msgs')
            if len(self.rl_data) >= STEPS + 1:
                self.get_logger().info('Collection complete.')
                raise SystemExit

        def _odom_cb(self, msg):
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            self.odom_data.append((len(self.odom_data), x, y))

    rclpy.init()
    node = Collector()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass

    # Save
    with open(RL_COLLECT_PATH, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'rl_x', 'rl_y'])
        for row in node.rl_data[:STEPS + 1]:
            w.writerow([row[0], round(row[1], 5), round(row[2], 5)])

    print(f'Saved {len(node.rl_data)} rl rows → {RL_COLLECT_PATH}')
    node.destroy_node()
    rclpy.shutdown()


# 
# MODE 2: --compare   (offline, uses saved CSVs)
# 
def compare():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    #  Load your real EKF log 
    print("Loading ekf_log.csv ...")
    ekf_rows = list(csv.DictReader(open(EKF_LOG_PATH)))
    ekf_by_step = {}
    for r in ekf_rows:
        step = int(r['step'])
        ekf_by_step[step] = (float(r['ekf_x']), float(r['ekf_y']))
    ekf_path = np.array([ekf_by_step.get(s, (0.0, 0.0)) for s in range(STEPS + 1)])

    #  Load robot_localization collected data 
    if not os.path.exists(RL_COLLECT_PATH):
        print(f"\nERROR: {RL_COLLECT_PATH} not found.")
        print("Run with --collect first while simulation is live.")
        sys.exit(1)

    print("Loading rl_collected.csv ...")
    rl_rows = list(csv.DictReader(open(RL_COLLECT_PATH)))
    rl_path = np.array([[float(r['rl_x']), float(r['rl_y'])] for r in rl_rows])
    if len(rl_path) < STEPS + 1:
        # pad with last value if short
        while len(rl_path) < STEPS + 1:
            rl_path = np.vstack([rl_path, rl_path[-1]])

    #  Regenerate ground truth + raw odometry (seed=42) 
    print("Regenerating ground truth (seed=42) ...")
    np.random.seed(42)
    gt = [np.array([0., 0., 0.])]
    for _ in range(STEPS):
        gt.append(motion_model(gt[-1], V, OMEGA, DT))
    gt = np.array(gt)

    odom = [np.array([0., 0., 0.])]
    for _ in range(STEPS):
        vn = V     + np.random.normal(0, SV)
        wn = OMEGA + np.random.normal(0, SO)
        odom.append(motion_model(odom[-1], vn, wn, DT))
    odom = np.array(odom)

    #  Errors 
    steps_arr = np.arange(STEPS + 1)
    odom_err = np.sqrt((odom[:,0]-gt[:,0])**2    + (odom[:,1]-gt[:,1])**2)
    ekf_err  = np.sqrt((ekf_path[:,0]-gt[:,0])**2 + (ekf_path[:,1]-gt[:,1])**2)
    rl_err   = np.sqrt((rl_path[:,0]-gt[:,0])**2  + (rl_path[:,1]-gt[:,1])**2)

    odom_rmse = float(np.sqrt(np.mean(odom_err**2)))
    ekf_rmse  = float(np.sqrt(np.mean(ekf_err**2)))
    rl_rmse   = float(np.sqrt(np.mean(rl_err**2)))

    ekf_vs_odom = (odom_rmse - ekf_rmse) / odom_rmse * 100
    rl_vs_odom  = (odom_rmse - rl_rmse)  / odom_rmse * 100
    ekf_vs_rl   = (rl_rmse   - ekf_rmse) / rl_rmse   * 100

    print(f"\n{'='*55}")
    print(f"  RESULTS (real ROS data)")
    print(f"{'='*55}")
    print(f"  Odometry RMSE:           {odom_rmse:.4f} m")
    print(f"  Your EKF RMSE:           {ekf_rmse:.4f} m  ({ekf_vs_odom:+.1f}% vs odom)")
    print(f"  robot_localization RMSE: {rl_rmse:.4f} m  ({rl_vs_odom:+.1f}% vs odom)")
    print(f"  Your EKF vs rl:          {ekf_vs_rl:+.1f}% better")
    print(f"{'='*55}\n")

    #  Save summary CSV (table) ─
    summary_path = os.path.join(OUT_DIR, 'comparison_summary_ros.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Method', 'RMSE (m)', 'Improvement vs Odometry (%)', 'Improvement vs rl (%)'])
        w.writerow(['Odometry',           round(odom_rmse,4), '—',                   f'{-ekf_vs_odom:+.1f}'])
        w.writerow(['Your Landmark EKF',  round(ekf_rmse,4),  f'{ekf_vs_odom:+.1f}', f'{ekf_vs_rl:+.1f}'])
        w.writerow(['robot_localization', round(rl_rmse,4),   f'{rl_vs_odom:+.1f}',  '—'])
    print(f"Summary table → {summary_path}")

    #  Save per-step CSV 
    perstep_path = os.path.join(OUT_DIR, 'comparison_perstep_ros.csv')
    with open(perstep_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step','gt_x','gt_y','odom_x','odom_y',
                    'ekf_x','ekf_y','rl_x','rl_y',
                    'odom_error_m','ekf_error_m','rl_error_m'])
        for i in range(STEPS + 1):
            w.writerow([i,
                round(float(gt[i,0]),4),       round(float(gt[i,1]),4),
                round(float(odom[i,0]),4),     round(float(odom[i,1]),4),
                round(float(ekf_path[i,0]),4), round(float(ekf_path[i,1]),4),
                round(float(rl_path[i,0]),4),  round(float(rl_path[i,1]),4),
                round(float(odom_err[i]),5),
                round(float(ekf_err[i]),5),
                round(float(rl_err[i]),5)])
    print(f"Per-step data → {perstep_path}")

    #  Colours ─
    DARK = '#0d1117'
    C_GT, C_ODOM, C_EKF, C_RL = '#ffffff', '#e74c3c', '#2ecc71', '#3498db'

    def dark_ax(fig, ax):
        fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
        for spine in ax.spines.values(): spine.set_edgecolor('#444')
        ax.tick_params(colors='#aaa')
        ax.xaxis.label.set_color('#aaa'); ax.yaxis.label.set_color('#aaa')

    #  Plot 1: Trajectories ─
    fig, ax = plt.subplots(figsize=(10, 10))
    dark_ax(fig, ax)
    ax.plot(gt[:,0],       gt[:,1],       color=C_GT,   lw=2.5, label='Ground Truth', zorder=5)
    ax.plot(odom[:,0],     odom[:,1],     color=C_ODOM, lw=1.5, alpha=0.8, label=f'Odometry  RMSE={odom_rmse:.4f}m')
    ax.plot(rl_path[:,0],  rl_path[:,1],  color=C_RL,   lw=1.8, alpha=0.85,
            label=f'robot_localization  RMSE={rl_rmse:.4f}m')
    ax.plot(ekf_path[:,0], ekf_path[:,1], color=C_EKF,  lw=2.0, alpha=0.95,
            label=f'Your EKF  RMSE={ekf_rmse:.4f}m')
    ax.scatter(LANDMARKS[:,0], LANDMARKS[:,1], c='cyan', s=150, zorder=7,
               marker='^', edgecolors='white', linewidth=0.8)
    for i,(lx,ly) in enumerate(LANDMARKS):
        ax.annotate(f'L{i}', (lx,ly), textcoords='offset points',
                    xytext=(6,4), fontsize=10, color='cyan')
    ax.scatter([0],[0], c='white', s=120, zorder=8, marker='o')
    ax.annotate('Start', (0,0), textcoords='offset points', xytext=(6,4), color='white', fontsize=10)
    ax.set_xlabel('X (m)', fontsize=12); ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Trajectory: Your EKF vs robot_localization vs Odometry\n(Real ROS data)',
                 color='white', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, facecolor='#1e1e2e', labelcolor='white', edgecolor='#444')
    ax.grid(alpha=0.15, color='white'); ax.set_aspect('equal')
    p = os.path.join(OUT_DIR, 'plot1_trajectories_ros.png')
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close()
    print(f"Plot 1 → {p}")

    #  Plot 2: Error over time 
    fig, ax = plt.subplots(figsize=(13, 5))
    dark_ax(fig, ax)
    ax.fill_between(steps_arr, 0, odom_err, color=C_ODOM, alpha=0.10)
    ax.plot(steps_arr, odom_err, color=C_ODOM, lw=1.2, alpha=0.7,
            label=f'Odometry  RMSE={odom_rmse:.4f}m')
    ax.plot(steps_arr, rl_err,   color=C_RL,   lw=1.5, alpha=0.85,
            label=f'robot_localization  RMSE={rl_rmse:.4f}m')
    ax.plot(steps_arr, ekf_err,  color=C_EKF,  lw=2.0, alpha=0.95,
            label=f'Your EKF  RMSE={ekf_rmse:.4f}m')
    ax.axvspan(184, 296, alpha=0.07, color='yellow')
    ymax = max(odom_err.max(), rl_err.max(), ekf_err.max())
    ax.text(240, ymax*0.8, 'No\nlandmarks', color='yellow', alpha=0.7, fontsize=9, ha='center')
    ax.axvline(25,  color='cyan', lw=0.8, alpha=0.4, linestyle='--')
    ax.axvline(297, color='cyan', lw=0.8, alpha=0.4, linestyle='--')
    ax.text(25,  ymax*0.05, 'L1\nenters', color='cyan', alpha=0.6, fontsize=8, ha='center')
    ax.text(297, ymax*0.05, 'L0\nre-enters', color='cyan', alpha=0.6, fontsize=8, ha='center')
    ax.set_xlabel('Step', fontsize=12); ax.set_ylabel('Position Error (m)', fontsize=12)
    ax.set_title('Per-Step Error: Your EKF vs robot_localization (Real ROS data)',
                 color='white', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, facecolor='#1e1e2e', labelcolor='white', edgecolor='#444')
    ax.grid(alpha=0.15, color='white')
    p = os.path.join(OUT_DIR, 'plot2_error_over_time_ros.png')
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close()
    print(f"Plot 2 → {p}")

    #  Plot 3: RMSE bar chart ─
    fig, ax = plt.subplots(figsize=(8, 6))
    dark_ax(fig, ax)
    methods = ['Odometry', 'robot_localization\n(odom-only)', 'Your\nLandmark EKF']
    values  = [odom_rmse, rl_rmse, ekf_rmse]
    colors  = [C_ODOM, C_RL, C_EKF]
    bars = ax.bar(methods, values, color=colors, alpha=0.85, edgecolor='white', width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., val + max(values)*0.01,
                f'{val:.4f} m', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='white')
    ax.set_ylabel('RMSE (m)', fontsize=12)
    ax.set_title('RMSE Comparison (Real ROS data)',
                 color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#aaa', labelsize=11)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis='y', alpha=0.2, color='white')
    p = os.path.join(OUT_DIR, 'plot3_rmse_bar_ros.png')
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close()
    print(f"Plot 3 → {p}")

    #  Plot 4: σ_xx from your actual log ─
    sigma_by_step = {}
    for r in ekf_rows:
        step = int(r['step'])
        if r['sigma_xx']:
            sigma_by_step[step] = float(r['sigma_xx'])
    sigma_steps  = sorted(sigma_by_step.keys())
    sigma_values = [sigma_by_step[s] for s in sigma_steps]

    fig, ax = plt.subplots(figsize=(13, 5))
    dark_ax(fig, ax)
    ax.plot(sigma_steps, sigma_values, color=C_EKF, lw=1.5, alpha=0.9,
            label='σ_xx — x-position variance (from your EKF log)')
    ax.fill_between(sigma_steps, 0, sigma_values, color=C_EKF, alpha=0.1)
    ymax = max(sigma_values)
    ax.axvspan(184, 296, alpha=0.07, color='yellow')
    ax.text(240, ymax*0.7,  'No landmarks\n(variance grows)',   color='yellow', alpha=0.7, fontsize=9, ha='center')
    ax.text(30,  ymax*0.5,  'Landmarks visible\n(variance drops)', color='cyan', alpha=0.7, fontsize=9, ha='center')
    ax.set_xlabel('Step', fontsize=12); ax.set_ylabel('σ_xx (m²)', fontsize=12)
    ax.set_title('EKF Covariance σ_xx Over Time — Predict/Update Cycle Visible\n(Real logged data)',
                 color='white', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, facecolor='#1e1e2e', labelcolor='white', edgecolor='#444')
    ax.grid(alpha=0.15, color='white')
    p = os.path.join(OUT_DIR, 'plot4_covariance_ros.png')
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close()
    print(f"Plot 4 → {p}")

    print("\nAll done.")


#  Entry point ─
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true',
                        help='Collect live ROS data from /odometry/filtered')
    parser.add_argument('--compare', action='store_true',
                        help='Generate tables and plots from saved CSVs')
    args = parser.parse_args()

    if args.collect:
        collect()
    elif args.compare:
        compare()
    else:
        print("Usage:")
        print("  python3 rl_collect_and_compare.py --collect   (run while ROS is live)")
        print("  python3 rl_collect_and_compare.py --compare   (generate tables + plots)")