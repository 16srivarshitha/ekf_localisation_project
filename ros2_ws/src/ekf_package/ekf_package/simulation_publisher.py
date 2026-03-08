"""
simulation_publisher.py
Reads pre-generated .npy data and publishes at 10 Hz for the EKF node to consume.
Fixed: frame_id='odom', non-zero covariance, twist velocity, self.odometry_data (not self.odom)
"""

import math
import os

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class SimulationPublisher(Node):

    def __init__(self):
        super().__init__('simulation_publisher')

        data_dir = SCRIPT_DIR
        self.ground_truth  = np.load(os.path.join(data_dir, 'ground_truth.npy'))
        self.odometry_data = np.load(os.path.join(data_dir, 'odometry.npy'))
        self.landmarks     = np.load(os.path.join(data_dir, 'landmarks.npy'))
        self.observations  = np.load(os.path.join(data_dir, 'observations.npy'))

        self.step        = 0
        self.total_steps = len(self.ground_truth)

        self.true_path_msg      = Path()
        self.true_path_msg.header.frame_id = 'map'
        self.odom_path_msg      = Path()
        self.odom_path_msg.header.frame_id = 'map'

        self.pub_gt_pose   = self.create_publisher(PoseStamped,       '/ground_truth/pose',     10)
        self.pub_odom      = self.create_publisher(Odometry,          '/odom',                  10)
        self.pub_lm        = self.create_publisher(MarkerArray,       '/landmarks',             10)
        self.pub_obs       = self.create_publisher(Float64MultiArray, '/landmark_observations', 10)
        self.pub_true_path = self.create_publisher(Path,              '/true_path',             10)
        self.pub_odom_path = self.create_publisher(Path,              '/odom_path',             10)

        self.timer = self.create_timer(0.1, self.publish_step)
        self.get_logger().info(f'SimulationPublisher ready — {self.total_steps} steps')

    def publish_step(self):
        if self.step >= self.total_steps:
            self.get_logger().info('Simulation complete.')
            self.timer.cancel()
            return

        ts = self.get_clock().now().to_msg()
        self._publish_ground_truth(ts)
        self._publish_odometry(ts)
        self._publish_landmarks(ts)
        self._publish_observations(ts)

        if self.step % 50 == 0:
            self.get_logger().info(f'Step {self.step}/{self.total_steps - 1}')

        self.step += 1

    def _pose_stamped(self, ts, x, y, theta):
        ps = PoseStamped()
        ps.header.stamp    = ts
        ps.header.frame_id = 'map'
        ps.pose.position.x    = float(x)
        ps.pose.position.y    = float(y)
        ps.pose.position.z    = 0.0
        ps.pose.orientation.z = float(math.sin(theta / 2.0))
        ps.pose.orientation.w = float(math.cos(theta / 2.0))
        return ps

    def _publish_ground_truth(self, ts):
        x, y, theta = self.ground_truth[self.step]
        ps = self._pose_stamped(ts, x, y, theta)
        self.pub_gt_pose.publish(ps)

        self.true_path_msg.header.stamp = ts
        self.true_path_msg.poses.append(ps)
        self.pub_true_path.publish(self.true_path_msg)

    def _publish_odometry(self, ts):
        x, y, theta = self.odometry_data[self.step]

        odom_msg = Odometry()
        odom_msg.header.stamp    = ts
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id  = 'base_link'

        odom_msg.pose.pose.position.x = float(x)
        odom_msg.pose.pose.position.y = float(y)
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.z = float(math.sin(theta / 2.0))
        odom_msg.pose.pose.orientation.w = float(math.cos(theta / 2.0))

        odom_msg.pose.covariance[0]  = 0.05
        odom_msg.pose.covariance[7]  = 0.05
        odom_msg.pose.covariance[35] = 0.02

        odom_msg.twist.twist.linear.x  = 1.0
        odom_msg.twist.twist.angular.z = 0.2
        odom_msg.twist.covariance[0]  = 0.05
        odom_msg.twist.covariance[35] = 0.02

        self.pub_odom.publish(odom_msg)

        ps = self._pose_stamped(ts, x, y, theta)
        self.odom_path_msg.header.stamp = ts
        self.odom_path_msg.poses.append(ps)
        self.pub_odom_path.publish(self.odom_path_msg)

    def _publish_landmarks(self, ts):
        ma = MarkerArray()
        for i, (lx, ly) in enumerate(self.landmarks):
            m = Marker()
            m.header.stamp    = ts
            m.header.frame_id = 'map'
            m.ns     = 'landmarks'
            m.id     = i
            m.type   = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(lx)
            m.pose.position.y = float(ly)
            m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = 0.3
            m.scale.y = 0.3
            m.scale.z = 1.0
            m.color.r = 0.0
            m.color.g = 0.0
            m.color.b = 1.0
            m.color.a = 1.0
            ma.markers.append(m)
        self.pub_lm.publish(ma)

    def _publish_observations(self, ts):
        step_obs = self.observations[self.observations[:, 0] == self.step]
        msg = Float64MultiArray()
        msg.data = [float(v) for row in step_obs for v in row[1:]]
        self.pub_obs.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimulationPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()