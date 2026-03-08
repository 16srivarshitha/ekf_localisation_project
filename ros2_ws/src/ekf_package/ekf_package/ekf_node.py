"""
ekf_node.py  —  ROS 2 EKF localization node with readable terminal logging + CSV export.
"""

import csv
import math
import os
import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg        import Odometry, Path
from geometry_msgs.msg   import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg        import Float64MultiArray
from visualization_msgs.msg import MarkerArray

from ekf_package.ekf import EKFLocalizer

DT       = 0.1
LOG_PATH = '/ros2_ws/src/ekf_log.csv'

_G   = '\033[92m'; _Y  = '\033[93m'; _C  = '\033[96m'
_B   = '\033[94m'; _W  = '\033[97m'; _DIM = '\033[2m'
_RST = '\033[0m';  _BOLD = '\033[1m'


def normalize_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class EKFNode(Node):

    def __init__(self):
        super().__init__('ekf_node')

        self.ekf = EKFLocalizer(
            initial_state         = np.array([0.0, 0.0, 0.0]),
            initial_covariance    = np.eye(3) * 0.1,
            control_noise_cov     = np.diag([0.05**2, 0.02**2]),
            measurement_noise_cov = np.diag([0.20**2, 0.05**2]),
        )

        self.landmark_map: dict[int, np.ndarray] = {}
        self.prev_odom   = None
        self.step        = 0
        self.prev_sigma  = None

        # ── CSV log ───────────────────────────────────────────────────────
        self._csv_file = open(LOG_PATH, 'w', newline='')
        self._csv      = csv.writer(self._csv_file)
        self._csv.writerow([
            'step', 'ekf_x', 'ekf_y', 'theta_deg',
            'sigma_xx', 'sigma_yy', 'sigma_yaw',
            'event', 'obs_landmark', 'obs_range_m', 'obs_bearing_deg',
            'sigma_xx_before', 'sigma_xx_after'
        ])
        self.get_logger().info(f'CSV log -> {LOG_PATH}')

        self.ekf_path_msg = Path()
        self.ekf_path_msg.header.frame_id = 'map'

        self.pub_ekf_path = self.create_publisher(Path,                      '/ekf_path',           10)
        self.pub_ekf_pose = self.create_publisher(PoseWithCovarianceStamped, '/ekf/estimated_pose', 10)

        self.create_subscription(Odometry,          '/odom',                  self._odom_cb, 10)
        self.create_subscription(Float64MultiArray, '/landmark_observations', self._obs_cb,  10)
        self.create_subscription(MarkerArray,       '/landmarks',             self._lm_cb,   10)

        print(f'\n{_BOLD}{_W}{"="*64}{_RST}')
        print(f'{_BOLD}{_C}   EKF Localization Node  —  waiting for data...{_RST}')
        print(f'{_BOLD}{_W}{"="*64}{_RST}\n')
        print(f'{_DIM}  {"Step":>5}  {"EKF x":>8}  {"EKF y":>8}  {"θ":>8}  {"σ_xx":>9}  Event{_RST}')
        print(f'{_DIM}  {"─"*5}  {"─"*8}  {"─"*8}  {"─"*8}  {"─"*9}  {"─"*30}{_RST}')

    def __del__(self):
        try:
            self._csv_file.close()
        except Exception:
            pass

    # ── Landmark map ─────────────────────────────────────────────────────────
    def _lm_cb(self, msg: MarkerArray):
        if self.landmark_map:
            return
        for m in msg.markers:
            self.landmark_map[m.id] = np.array([m.pose.position.x, m.pose.position.y])
        print(f'\n{_G}{_BOLD}  ✓ Landmark map loaded — {len(self.landmark_map)} landmarks{_RST}')
        for lid, pos in sorted(self.landmark_map.items()):
            print(f'{_DIM}    L{lid}:  x={pos[0]:.1f}  y={pos[1]:.1f}{_RST}')
        print()

    # ── Odometry -> predict ───────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        ts    = msg.header.stamp
        x     = msg.pose.pose.position.x
        y     = msg.pose.pose.position.y
        qz    = msg.pose.pose.orientation.z
        qw    = msg.pose.pose.orientation.w
        theta = 2.0 * math.atan2(qz, qw)

        if self.prev_odom is None:
            self.prev_odom = (x, y, theta)
            return

        px, py, ptheta = self.prev_odom
        v     = math.sqrt((x - px)**2 + (y - py)**2) / DT
        omega = normalize_angle(theta - ptheta) / DT

        self.ekf.predict([v, omega], DT)
        self.step     += 1
        self.prev_odom = (x, y, theta)

        ex    = float(self.ekf.mu[0, 0])
        ey    = float(self.ekf.mu[1, 0])
        eth   = math.degrees(float(self.ekf.mu[2, 0]))
        s_xx  = float(self.ekf.Sigma[0, 0])
        s_yy  = float(self.ekf.Sigma[1, 1])
        s_yaw = float(self.ekf.Sigma[2, 2])

        event = 'predict'
        if self.prev_sigma is not None and s_xx < self.prev_sigma * 0.95:
            drop  = (self.prev_sigma - s_xx) / self.prev_sigma * 100
            event = f'UPDATE σ↓{drop:.1f}%'
        self.prev_sigma = s_xx

        # terminal
        colour = _Y if 'UPDATE' in event else _DIM
        ev_str = f'{_G}▼ {event}{_RST}' if 'UPDATE' in event else f'{_DIM}{event}{_RST}'
        print(f'{colour}  {self.step:>5}  {ex:>8.3f}  {ey:>8.3f}  {eth:>7.1f}°  {s_xx:>9.5f}  {ev_str}{_RST}')

        # CSV — predict row (no obs columns)
        self._csv.writerow([self.step, round(ex,4), round(ey,4), round(eth,2),
                            round(s_xx,6), round(s_yy,6), round(s_yaw,6),
                            event, '', '', '', '', ''])
        self._csv_file.flush()

        self._publish_ekf_path(ts)
        self._publish_ekf_pose(ts)

    # ── Observations -> update ────────────────────────────────────────────────
    def _obs_cb(self, msg: Float64MultiArray):
        if not self.landmark_map:
            return
        data = msg.data
        for i in range(0, len(data) - 2, 3):
            lm_id   = int(data[i])
            r       = data[i + 1]
            bearing = data[i + 2]
            if lm_id not in self.landmark_map:
                continue
            s_before = float(self.ekf.Sigma[0, 0])
            self.ekf.update([r, bearing], self.landmark_map[lm_id])
            s_after = float(self.ekf.Sigma[0, 0])
            self.prev_sigma = s_after

            brg_deg = math.degrees(bearing)
            print(f'{_B}         -> obs L{lm_id}  r={r:.2f}m  brg={brg_deg:+.1f}°  '
                  f'σ_xx  {s_before:.5f} -> {s_after:.5f}{_RST}')

            # CSV — observation row (step repeated, obs columns filled)
            ex  = float(self.ekf.mu[0, 0])
            ey  = float(self.ekf.mu[1, 0])
            eth = math.degrees(float(self.ekf.mu[2, 0]))
            self._csv.writerow([self.step, round(ex,4), round(ey,4), round(eth,2),
                                round(s_after,6), '', '',
                                f'obs_L{lm_id}',
                                lm_id, round(r,3), round(brg_deg,2),
                                round(s_before,6), round(s_after,6)])
            self._csv_file.flush()

    # ── Publishers ───────────────────────────────────────────────────────────
    def _publish_ekf_path(self, stamp):
        x     = float(self.ekf.mu[0, 0])
        y     = float(self.ekf.mu[1, 0])
        theta = float(self.ekf.mu[2, 0])
        ps = PoseStamped()
        ps.header.stamp    = stamp
        ps.header.frame_id = 'map'
        ps.pose.position.x    = x
        ps.pose.position.y    = y
        ps.pose.position.z    = 0.0
        ps.pose.orientation.z = math.sin(theta / 2.0)
        ps.pose.orientation.w = math.cos(theta / 2.0)
        self.ekf_path_msg.header.stamp = stamp
        self.ekf_path_msg.poses.append(ps)
        self.pub_ekf_path.publish(self.ekf_path_msg)

    def _publish_ekf_pose(self, stamp):
        x     = float(self.ekf.mu[0, 0])
        y     = float(self.ekf.mu[1, 0])
        theta = float(self.ekf.mu[2, 0])
        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x    = x
        msg.pose.pose.position.y    = y
        msg.pose.pose.position.z    = 0.0
        msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(theta / 2.0)
        cov6 = [0.0] * 36
        S = self.ekf.Sigma
        cov6[0]  = float(S[0, 0]); cov6[1]  = float(S[0, 1]); cov6[5]  = float(S[0, 2])
        cov6[6]  = float(S[1, 0]); cov6[7]  = float(S[1, 1]); cov6[11] = float(S[1, 2])
        cov6[30] = float(S[2, 0]); cov6[31] = float(S[2, 1]); cov6[35] = float(S[2, 2])
        msg.pose.covariance = cov6
        self.pub_ekf_pose.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    try:
        rclpy.spin(node)
    finally:
        node._csv_file.close()
        print(f'\n{_G}CSV saved -> {LOG_PATH}{_RST}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()