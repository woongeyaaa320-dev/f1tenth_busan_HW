"""kill_switch_demo_node: hardcoded constant-speed driver for a kill switch
demonstration in an unmapped environment. No localization, mapping, or path
following -- publishes a fixed small forward speed (zero steering) to /auto,
immediately zeroes it while /safety/kill_switch is engaged, and resumes
automatically once released (matching the real toggle behavior every other
controller uses, so this doubles as proof the switch is a genuine on/off
toggle, not just a one-shot stop). Also auto-stops on its own after
max_duration as a safety net independent of the kill switch, in case the
operator forgets to Ctrl+C.
"""

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool


class KillSwitchDemoNode(Node):
    def __init__(self):
        super().__init__('kill_switch_demo_node')

        self.declare_parameter('drive_topic', '/auto')
        self.declare_parameter('kill_switch_topic', '/safety/kill_switch')
        self.declare_parameter('speed', 1.0)
        self.declare_parameter('steering_angle', 0.0)
        self.declare_parameter('control_rate', 30.0)
        # Give the operator a moment after launch before it actually starts
        # moving, so there is no risk of it lurching before hands are on
        # the kill switch.
        self.declare_parameter('start_delay', 3.0)
        # Independent safety net: stop on its own even if the kill switch
        # is never pressed and nobody Ctrl+C's the node.
        self.declare_parameter('max_duration', 10.0)

        self.drive_topic = self.get_parameter('drive_topic').value
        self.kill_switch_topic = self.get_parameter('kill_switch_topic').value
        self.speed = float(self.get_parameter('speed').value)
        self.steering_angle = float(self.get_parameter('steering_angle').value)
        control_rate = float(self.get_parameter('control_rate').value)
        self.start_delay = float(self.get_parameter('start_delay').value)
        self.max_duration = float(self.get_parameter('max_duration').value)

        self.kill_switch_engaged = False
        self.start_time = None
        self.timed_out = False

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10)
        self.create_subscription(
            Bool, self.kill_switch_topic, self.kill_switch_callback, 10)
        self.create_timer(1.0 / max(control_rate, 1.0), self.control_loop)

        self.get_logger().info(
            'Kill switch demo ready: will drive speed=%.2fm/s on %s after '
            '%.1fs, watching %s. Auto-stops after %.1fs regardless.'
            % (self.speed, self.drive_topic, self.start_delay,
               self.kill_switch_topic, self.max_duration))

    def kill_switch_callback(self, msg):
        self.kill_switch_engaged = bool(msg.data)
        self.get_logger().warn(
            'Kill switch %s'
            % ('ENGAGED -> stopping' if self.kill_switch_engaged
               else 'released -> resuming'))

    def publish(self, speed, steering_angle):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)

    def control_loop(self):
        now = self.get_clock().now()
        if self.start_time is None:
            self.start_time = now
        elapsed = (now - self.start_time).nanoseconds * 1e-9

        if elapsed < self.start_delay:
            self.publish(0.0, 0.0)
            return
        if elapsed - self.start_delay >= self.max_duration:
            if not self.timed_out:
                self.timed_out = True
                self.get_logger().info('max_duration reached, stopping')
            self.publish(0.0, 0.0)
            return
        if self.kill_switch_engaged:
            self.publish(0.0, 0.0)
            return
        self.publish(self.speed, self.steering_angle)


def main(args=None):
    rclpy.init(args=args)
    node = KillSwitchDemoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
