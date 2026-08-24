"""Manual kill switch: a joystick-button software toggle.

Competition rule 3.3.1 requires the kill switch to operate as a toggle
(on/off) rather than a press-and-hold button, and to be independent from any
autonomous safety system (AEB).  joy_teleop cannot provide this by itself:
it can only re-publish a fixed message while a button is held, so a
press-and-hold gesture is the only behavior it can produce from a single
button.  This node instead reads /joy directly, remembers the last button
state itself, and flips a latched boolean on each press (the 0->1 edge),
independent of how long the button stays down.

The AEB stop published by the local obstacle planner republishes on
/safety/emergency_stop every planning cycle, so a one-shot publish on that
same topic would be overwritten within milliseconds.  This node instead owns
a dedicated topic (/safety/kill_switch) that every controller ORs into its
own stop logic, and republishes its latched state continuously so a
controller that starts late -- or a single dropped message -- still sees the
correct state immediately.
"""

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from sensor_msgs.msg import Joy
from std_msgs.msg import Bool


class KillSwitchNode(Node):
    def __init__(self):
        super().__init__('kill_switch_node')

        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('kill_switch_topic', '/safety/kill_switch')
        # -1 is an intentionally invalid sentinel.  The correct index depends
        # on the controller/driver in use and must be confirmed against the
        # real /joy stream (ros2 topic echo /joy) before this node can do
        # anything; refusing to guess is safer than silently watching the
        # wrong button.
        self.declare_parameter('kill_switch_button', -1)
        self.declare_parameter('publish_rate', 20.0)

        self.joy_topic = self.get_parameter('joy_topic').value
        self.kill_switch_topic = self.get_parameter('kill_switch_topic').value
        self.kill_switch_button = int(
            self.get_parameter('kill_switch_button').value)
        publish_rate = float(self.get_parameter('publish_rate').value)
        if publish_rate <= 0.0:
            raise RuntimeError('publish_rate must be positive')

        if self.kill_switch_button < 0:
            self.get_logger().error(
                'kill_switch_button is not configured (%d): run '
                '"ros2 topic echo %s" while pressing the kill switch button, '
                'find the index that flips 0->1 in the buttons array, and '
                'set kill_switch_button:=<that index>. This node will not '
                'toggle the kill switch until it is configured.'
                % (self.kill_switch_button, self.joy_topic))

        self.engaged = False
        self.previous_button_state = 0

        self.state_pub = self.create_publisher(
            Bool, self.kill_switch_topic, 10)
        self.create_subscription(Joy, self.joy_topic, self.joy_callback, 10)
        self.create_timer(1.0 / publish_rate, self.publish_state)

        self.get_logger().info(
            'Kill switch ready (joy=%s button=%d -> %s)' % (
                self.joy_topic, self.kill_switch_button,
                self.kill_switch_topic))

    def joy_callback(self, msg):
        if self.kill_switch_button < 0:
            return
        if len(msg.buttons) <= self.kill_switch_button:
            self.get_logger().error(
                'Joystick reports only %d buttons, but kill_switch_button=%d '
                'was requested; kill switch cannot see this button.'
                % (len(msg.buttons), self.kill_switch_button))
            return

        button_state = int(msg.buttons[self.kill_switch_button])
        if button_state == 1 and self.previous_button_state == 0:
            self.engaged = not self.engaged
            self.get_logger().warn(
                'Kill switch %s' % ('ENGAGED' if self.engaged else 'released'))
            self.publish_state()
        self.previous_button_state = button_state

    def publish_state(self):
        msg = Bool()
        msg.data = self.engaged
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = KillSwitchNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, RCLError):
            pass


if __name__ == '__main__':
    main()
