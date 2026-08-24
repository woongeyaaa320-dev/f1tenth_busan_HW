"""Minimal constant-speed raceline follower for the second (opponent) agent.

This node exists only to give the ego vehicle's dynamic-obstacle avoidance
something real to react to in simulation. It is NOT a racing controller: no
speed profile, no AEB, no enable/disable service. It reads ``odom_topic``
(the opponent's own ground-truth odometry published directly by gym_bridge
when ``num_agent: 2``) and drives a fixed-speed pure-pursuit lap of a
raceline CSV, publishing AckermannDriveStamped on ``drive_topic``.
"""

import csv
import math

import numpy as np
import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry


def _load_xy(csv_path):
    with open(csv_path, newline='') as stream:
        rows = [
            row for row in csv.reader(stream)
            if row and not row[0].strip().startswith('#')
        ]
    header = [cell.strip().lower() for cell in rows[0]]
    x_index = next(header.index(name) for name in ('x', 'x_m') if name in header)
    y_index = next(header.index(name) for name in ('y', 'y_m') if name in header)
    return np.asarray([
        [float(row[x_index]), float(row[y_index])] for row in rows[1:]
    ], dtype=float)


class OpponentDriverNode(Node):
    """Drive a second simulated agent at a constant speed around a raceline."""

    def __init__(self):
        super().__init__('opponent_driver_node')

        self.declare_parameter('waypoint_csv', '')
        self.declare_parameter('odom_topic', '/opp_racecar/odom')
        self.declare_parameter('drive_topic', '/opp_drive')
        self.declare_parameter('target_speed', 1.0)
        self.declare_parameter('lookahead_distance', 1.0)
        self.declare_parameter('wheelbase', 0.324)
        self.declare_parameter('max_steering_angle', 0.4189)
        self.declare_parameter('control_rate', 20.0)

        csv_path = self.get_parameter('waypoint_csv').value
        if not csv_path:
            raise RuntimeError(
                'opponent_driver_node requires waypoint_csv to be set')
        self.points = _load_xy(csv_path)
        if len(self.points) < 4:
            raise RuntimeError(f'Raceline needs at least 4 points: {csv_path}')
        segments = np.roll(self.points, -1, axis=0) - self.points
        self.segment_lengths = np.linalg.norm(segments, axis=1)
        self.cumulative = np.concatenate(
            ([0.0], np.cumsum(self.segment_lengths)))
        self.path_length = float(self.cumulative[-1])

        self.target_speed = float(self.get_parameter('target_speed').value)
        self.lookahead = max(
            0.3, float(self.get_parameter('lookahead_distance').value))
        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.max_steering_angle = float(
            self.get_parameter('max_steering_angle').value)

        self.latest_odom = None
        self.last_nearest_index = 0

        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self.odom_callback,
            10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            self.get_parameter('drive_topic').value,
            10)

        control_rate = max(1.0, float(self.get_parameter('control_rate').value))
        self.create_timer(1.0 / control_rate, self.control_step)
        self.get_logger().info(
            'Opponent driver ready: %d waypoints, %.2f m lap, speed=%.2f m/s'
            % (len(self.points), self.path_length, self.target_speed))

    def odom_callback(self, message):
        self.latest_odom = message

    @staticmethod
    def _yaw(quaternion):
        return math.atan2(
            2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
            1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2))

    def _nearest_index(self, position):
        # Search the whole lap; ~500 points at 20 Hz is cheap and avoids ever
        # getting stuck searching only a stale local window.
        distances = np.linalg.norm(self.points - position, axis=1)
        return int(np.argmin(distances))

    def _lookahead_point(self, start_index):
        start_s = self.cumulative[start_index]
        target_s = (start_s + self.lookahead) % self.path_length
        wrapped_x = np.concatenate((self.points[:, 0], [self.points[0, 0]]))
        wrapped_y = np.concatenate((self.points[:, 1], [self.points[0, 1]]))
        return np.asarray([
            np.interp(target_s, self.cumulative, wrapped_x),
            np.interp(target_s, self.cumulative, wrapped_y),
        ])

    def control_step(self):
        if self.latest_odom is None:
            return
        pose = self.latest_odom.pose.pose
        position = np.asarray([pose.position.x, pose.position.y])
        yaw = self._yaw(pose.orientation)

        nearest_index = self._nearest_index(position)
        target = self._lookahead_point(nearest_index)
        dx, dy = target - position
        local_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        local_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy
        lookahead_actual = max(0.1, math.hypot(local_x, local_y))
        curvature = 2.0 * local_y / (lookahead_actual ** 2)
        steering = math.atan(self.wheelbase * curvature)
        steering = max(
            -self.max_steering_angle,
            min(self.max_steering_angle, steering))

        message = AckermannDriveStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.drive.speed = self.target_speed
        message.drive.steering_angle = steering
        self.drive_pub.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = OpponentDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
