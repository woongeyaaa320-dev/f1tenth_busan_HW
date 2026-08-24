"""
Humble adapter for the UNICORN Racing Stack L1 controller strategy.

The controller structure is adapted from the MIT-licensed HMCL-UNIST
UNICORN Racing Stack (ROS 2 Jazzy):
https://github.com/HMCL-UNIST/unicorn-racing-stack

This adapter intentionally uses this project's existing nav_msgs/Path, TF,
odometry, safety, and Ackermann interfaces.  It does not pretend to be MPC:
UNICORN's current controller is an L1/Pure-Pursuit controller.

This specific variant (WoongPpNode / controller:=woong_pp) is
ported from woongeyaaa320-dev/f1tenth-obstacle-tuning (a fork of this same
project's obstacle-avoidance-tuning branch). It adds several stability
mechanisms this project's own unicorn_l1_node.py does not have: a grace
period before a single-cycle heading-error glitch triggers a full stop
(HeadingErrorFault), a position-jump reset for the nearest-index search,
a post-maneuver acceleration hold after a corner/avoidance manoeuvre, and a
capped lateral-error steering gain. Its own README documents these fixes as
validated only at low speed in sim (maximum_speed:=1.5 m/s recommended;
2.4+ m/s showed reproducible collisions on some obstacle placements) -- it
has NOT been tuned or validated for this project's real-car targets (8+
m/s). Its `t_clip_min`/`t_clip_max` (1.10/8.00) also differ from this
project's own unicorn_l1_node.py (0.70/3.00, sized down for this project's
short ~23 m test tracks) -- treat this as a separate, unvalidated-at-speed
controller to test incrementally from a low speed:=, not a drop-in
replacement.
"""

import math

import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float32
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class HeadingErrorFault(RuntimeError):
    """Raised when the tracked heading error exceeds ``max_heading_error``.

    Kept distinct from other ``RuntimeError`` causes (stale path, vehicle too
    far from path) because this one is observed to occur as a single-cycle
    localization glitch (e.g. AMCL briefly misreading pose during an
    aggressive avoidance swerve) that self-corrects within a few control
    cycles.  It is handled with the same brief hold-last-command grace period
    already used for transient TF faults, instead of an instant full stop.
    """


def build_closed_velocity_profile(
        curvature, segment_lengths, maximum_speed,
        maximum_lateral_acceleration, maximum_acceleration,
        maximum_deceleration):
    """
    Return a closed-loop, dynamics-limited speed profile.

    Lateral limits follow ``a_y = v^2 * kappa``.  Repeated forward and
    backward passes then make every transition reachable under the configured
    acceleration and braking limits.  This is the same separation used by
    global raceline optimizers: path geometry first, velocity profile second.
    """
    curvature = np.abs(np.asarray(curvature, dtype=float))
    segment_lengths = np.asarray(segment_lengths, dtype=float)
    if (len(curvature) == 0
            or len(curvature) != len(segment_lengths)):
        raise ValueError('curvature and segment lengths must be non-empty')

    maximum_speed = max(0.0, float(maximum_speed))
    lateral = max(1e-3, float(maximum_lateral_acceleration))
    acceleration = max(1e-3, float(maximum_acceleration))
    deceleration = max(1e-3, float(maximum_deceleration))
    profile = np.minimum(
        maximum_speed,
        np.sqrt(lateral / np.maximum(curvature, 1e-4)))

    # A few complete circular sweeps are sufficient for constraints to cross
    # the array boundary while keeping path-update cost deterministic.
    count = len(profile)
    for _ in range(4):
        for index in range(count):
            next_index = (index + 1) % count
            reachable = math.sqrt(max(
                0.0,
                profile[index] ** 2
                + 2.0 * acceleration * segment_lengths[index]))
            profile[next_index] = min(profile[next_index], reachable)
        for index in range(count - 1, -1, -1):
            next_index = (index + 1) % count
            reachable = math.sqrt(max(
                0.0,
                profile[next_index] ** 2
                + 2.0 * deceleration * segment_lengths[index]))
            profile[index] = min(profile[index], reachable)
    return profile


class WoongPpNode(Node):
    """L1/Pure-Pursuit tracker using UNICORN's adaptive guidance strategy."""

    controller_label = 'UNICORN L1'
    topic_prefix = '/unicorn_l1'

    def __init__(self):
        super().__init__('woong_pp_node')

        self.declare_parameter('enabled', False)
        self.declare_parameter('solve_when_disabled', True)
        self.declare_parameter('global_frame_id', 'map')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'ego_racecar/base_link')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('path_topic', '/planning/path')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('collision_topic', '/ego_racecar/collision')
        self.declare_parameter(
            'emergency_stop_topic', '/safety/emergency_stop')
        # Rule 3.3.1 manual on/off kill switch, independent of AEB.
        self.declare_parameter('kill_switch_topic', '/safety/kill_switch')
        self.declare_parameter(
            'avoidance_active_topic', '/planning/avoidance_active')
        self.declare_parameter('speed_limit_topic', '/planning/speed_limit')
        self.declare_parameter('speed_limit_timeout', 0.50)
        self.declare_parameter('use_dynamic_speed_limit', False)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('control_rate', 50.0)
        self.declare_parameter('target_speed', 1.0)
        self.declare_parameter('min_reference_speed', 0.30)
        self.declare_parameter('min_command_speed', 0.0)
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('max_lateral_acceleration', 1.50)
        self.declare_parameter('max_longitudinal_acceleration', 2.0)
        self.declare_parameter('max_longitudinal_deceleration', 4.0)
        self.declare_parameter('avoidance_speed_limit', 1.50)
        self.declare_parameter('max_steering_angle', 0.4189)
        self.declare_parameter('max_steering_delta', 0.40)
        # Physical steering slew limit. The legacy per-cycle delta is kept as
        # an upper compatibility bound, but no longer depends on loop rate.
        self.declare_parameter('max_steering_rate', 6.0)
        # Only transient TF lookup failures may hold the last safe command.
        # Collision, AEB and invalid-path conditions still stop immediately.
        self.declare_parameter('transform_fault_grace', 0.10)
        # A stationary vehicle gives AMCL no new odometry to resample on, so
        # a heading-error fault that is allowed to reach a full stop can
        # never self-correct.  Bridging brief glitches here (same grace
        # style as transform_fault_grace) keeps the vehicle moving long
        # enough for AMCL's own recovery resampling to fix the estimate.
        self.declare_parameter('heading_error_grace', 0.10)

        # UNICORN controller.yaml defaults. Distance-window curvature sampling
        # replaces its waypoint-index window so behavior is path-resolution
        # independent.
        self.declare_parameter('t_clip_min', 1.10)
        self.declare_parameter('t_clip_max', 8.00)
        self.declare_parameter('m_l1', 0.47)
        self.declare_parameter('q_l1', -0.20)
        self.declare_parameter('curvature_factor', 0.145)
        self.declare_parameter('future_constant', 0.05)
        self.declare_parameter('curvature_window_start', 0.50)
        self.declare_parameter('curvature_window_end', 1.50)
        # Extra L1 lookahead on a confirmed long straight, on top of the
        # existing speed/curvature terms below. straight_lookahead_boost=1.0
        # (default) means this has no effect at all -- a deliberately inert
        # default so it can be tested via a launch/param override without
        # changing behavior for anyone not explicitly opting in. See
        # TUNING_LOG for the reasoning and test results before raising this
        # past 1.0 as a new default.
        self.declare_parameter('straight_curvature_threshold', 0.20)
        self.declare_parameter('straight_reference_distance', 4.0)
        self.declare_parameter('straight_lookahead_boost', 1.0)
        # Read the spatial speed profile ahead of the current pose so braking
        # begins before the L1 steering transition, especially when an
        # occluded static obstacle first becomes visible near a bend.
        self.declare_parameter('speed_lookahead', 0.25)
        self.declare_parameter('lat_err_coeff', 1.0)
        self.declare_parameter('speed_factor_for_lat_err', 1.0)
        self.declare_parameter('speed_factor_for_curvature', 1.0)
        self.declare_parameter('heading_kp', 0.8)
        self.declare_parameter('heading_kd', 0.0)
        self.declare_parameter('heading_filter_alpha', 0.05)
        self.declare_parameter('heading_gain_speed', 15.0)
        self.declare_parameter('lateral_steering_gain_cap', 0.40)
        # A windowed nearest-point search (search_back/forward_points) is
        # cheap per-cycle but assumes the vehicle only moves along the path
        # between cycles.  A discontinuous position jump -- /sim_reset_pose
        # teleporting the vehicle anywhere for a manual test, or a real
        # kidnapped-robot AMCL correction -- leaves the search window
        # anchored on the old location, so it can report "far from path"
        # even when the new position is right on the raceline.  Detecting a
        # jump and forcing one full-path search re-acquires the correct
        # index regardless of where the vehicle appears.
        self.declare_parameter('position_jump_reset_distance', 1.0)
        self.declare_parameter('heading_slowdown_threshold_deg', 10.0)
        # The curvature-based speed profile only knows the path's geometry,
        # not whether the LiDAR has actually confirmed the road ahead is
        # clear -- right after a corner or an avoidance manoeuvre, the
        # sensor's view forward can still be limited by what it was just
        # looking at. Observed as a wall collision (map=True) immediately
        # after a curved/avoidance section once the variable-speed fix
        # let the vehicle accelerate hard coming out of it. Holding off on
        # *acceleration* (deceleration remains unrestricted) for a short
        # window after the last high-curvature or avoidance moment gives
        # the sensor a chance to re-confirm a clear path before speed is
        # allowed to climb again.
        self.declare_parameter('post_maneuver_hold_time', 0.6)

        self.declare_parameter('max_path_distance', 0.80)
        self.declare_parameter('max_heading_error', 1.0472)
        self.declare_parameter('odom_timeout', 0.50)
        self.declare_parameter('path_timeout', 2.00)
        self.declare_parameter('search_back_points', 5)
        self.declare_parameter('search_forward_points', 30)

        for name in (
                'enabled', 'solve_when_disabled', 'global_frame_id',
                'odom_frame_id', 'base_frame_id', 'odom_topic', 'path_topic',
                'drive_topic',
                'collision_topic', 'emergency_stop_topic',
                'kill_switch_topic',
                'avoidance_active_topic', 'speed_limit_topic',
                'use_dynamic_speed_limit'):
            setattr(self, name, self.get_parameter(name).value)
        for name in (
                'wheelbase', 'target_speed', 'min_reference_speed',
                'min_command_speed', 'max_speed',
                'max_lateral_acceleration',
                'max_longitudinal_acceleration',
                'max_longitudinal_deceleration', 'avoidance_speed_limit',
                'max_steering_angle',
                'max_steering_delta', 'max_steering_rate',
                'transform_fault_grace', 'heading_error_grace',
                't_clip_min', 't_clip_max', 'm_l1',
                'q_l1', 'curvature_factor', 'future_constant',
                'curvature_window_start', 'curvature_window_end',
                'straight_curvature_threshold', 'straight_reference_distance',
                'straight_lookahead_boost',
                'speed_lookahead',
                'lat_err_coeff', 'speed_factor_for_lat_err',
                'speed_factor_for_curvature', 'heading_kp', 'heading_kd',
                'heading_filter_alpha', 'heading_gain_speed',
                'lateral_steering_gain_cap', 'position_jump_reset_distance',
                'heading_slowdown_threshold_deg', 'post_maneuver_hold_time',
                'max_path_distance',
                'max_heading_error', 'odom_timeout', 'path_timeout',
                'speed_limit_timeout'):
            setattr(self, name, float(self.get_parameter(name).value))
        self.search_back_points = int(
            self.get_parameter('search_back_points').value)
        self.search_forward_points = int(
            self.get_parameter('search_forward_points').value)
        control_rate = float(self.get_parameter('control_rate').value)
        self.control_dt = 1.0 / max(control_rate, 1.0)

        if self.t_clip_min <= 0.0 or self.t_clip_max < self.t_clip_min:
            raise RuntimeError('invalid L1 distance limits')
        if self.curvature_window_end < self.curvature_window_start:
            raise RuntimeError('invalid curvature sampling window')
        if self.straight_lookahead_boost < 1.0:
            raise RuntimeError(
                'straight_lookahead_boost must be >= 1.0 (it only extends '
                'lookahead on a confirmed straight, never shortens it)')
        if not 0.0 <= self.min_command_speed <= self.max_speed:
            raise RuntimeError(
                'min_command_speed must be between 0 and max_speed')
        if self.max_lateral_acceleration <= 0.0:
            raise RuntimeError('max_lateral_acceleration must be positive')
        if (self.max_longitudinal_acceleration <= 0.0
                or self.max_longitudinal_deceleration <= 0.0):
            raise RuntimeError(
                'longitudinal acceleration limits must be positive')
        if self.avoidance_speed_limit <= 0.0:
            raise RuntimeError('avoidance_speed_limit must be positive')
        if self.max_steering_rate <= 0.0:
            raise RuntimeError('max_steering_rate must be positive')
        if self.transform_fault_grace < 0.0:
            raise RuntimeError('transform_fault_grace must be non-negative')
        self.current_odom = None
        self.last_odom_time = None
        self.last_path_time = None
        self.collision = False
        self.emergency_stop = False
        self.kill_switch_engaged = False
        self.avoidance_active = False
        self.dynamic_speed_limit = None
        self.last_speed_limit_time = None
        self.path_points = None
        self.path_yaw = None
        self.path_curvature = None
        self.path_speed_curvature = None
        self.path_segment_lengths = None
        self.path_cumulative = None
        self.path_length = None
        self.path_speed_profile = None
        self.yaw_lap_change = None
        self.nearest_index = None
        self.last_position = None
        self.maneuver_exit_time = None
        self.previous_steering = 0.0
        self.previous_command_speed = 0.0
        self.filtered_heading_error = None
        self.previous_heading_error = None
        self.last_solution_ok = False
        self.last_status_message = None
        self.last_status_time = None
        self.transform_fault_since = None
        self.heading_fault_since = None

        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Control needs the newest odometry sample, never a queued history.
        # A depth-10 queue can make the pose several control cycles old when
        # scan processing and RViz are active.
        self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 1)
        # The local planner can update at 20 Hz. Old paths are unsafe once a
        # newer obstacle estimate exists, so retain only the latest message.
        self.create_subscription(Path, self.path_topic, self.path_callback, 1)
        self.create_subscription(
            Bool, self.collision_topic, self.collision_callback, 10)
        self.create_subscription(
            Bool, self.emergency_stop_topic,
            self.emergency_stop_callback, 10)
        self.create_subscription(
            Bool, self.kill_switch_topic,
            self.kill_switch_callback, 10)
        self.create_subscription(
            Bool, self.avoidance_active_topic,
            self.avoidance_active_callback, 10)
        self.create_subscription(
            Float32, self.speed_limit_topic,
            self.speed_limit_callback, 10)

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10)
        self.proposed_drive_pub = self.create_publisher(
            AckermannDriveStamped,
            self.topic_prefix + '/proposed_drive', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, self.topic_prefix + '/markers', 10)
        self.enable_service = self.create_service(
            SetBool, '/control/enable', self.enable_callback)
        self.timer = self.create_timer(
            self.control_dt, self.control_loop)

        self.get_logger().info(
            '%s ready (enabled=%s, rate=%.1fHz, target=%.2fm/s)'
            % (self.controller_label, self.enabled, control_rate,
               self.target_speed))
        self.get_logger().info(
            'Humble adapter: nav_msgs/Path + TF -> Ackermann /drive '
            '(dynamic_speed_limit=%s)' % self.use_dynamic_speed_limit)

    @staticmethod
    def clamp(value, minimum, maximum):
        return max(minimum, min(value, maximum))

    @staticmethod
    def angle_difference(target, source):
        return math.atan2(
            math.sin(target - source), math.cos(target - source))

    @staticmethod
    def quaternion_to_yaw(quaternion):
        return math.atan2(
            2.0 * (quaternion.w * quaternion.z
                   + quaternion.x * quaternion.y),
            1.0 - 2.0 * (quaternion.y * quaternion.y
                         + quaternion.z * quaternion.z))

    def odom_callback(self, message):
        self.current_odom = message
        self.last_odom_time = self.get_clock().now()

    def collision_callback(self, message):
        self.collision = bool(message.data)
        if self.collision and self.enabled:
            self.enabled = False
            self.previous_command_speed = 0.0
            self.publish_stop()
            self.get_logger().error(
                self.controller_label
                + ' disabled: collision input is active')

    def emergency_stop_callback(self, message):
        self.emergency_stop = bool(message.data)
        if self.emergency_stop:
            self.previous_command_speed = 0.0
            self.publish_stop()

    def kill_switch_callback(self, message):
        self.kill_switch_engaged = bool(message.data)
        if self.kill_switch_engaged:
            self.enabled = False
            self.previous_command_speed = 0.0
            self.publish_stop()

    def avoidance_active_callback(self, message):
        self.avoidance_active = bool(message.data)

    def speed_limit_callback(self, message):
        value = float(message.data)
        if math.isfinite(value) and value >= 0.0:
            self.dynamic_speed_limit = value
            self.last_speed_limit_time = self.get_clock().now()

    def path_callback(self, message):
        if len(message.poses) < 4:
            return
        points = np.asarray([
            [pose.pose.position.x, pose.pose.position.y]
            for pose in message.poses
        ], dtype=float)
        if np.linalg.norm(points[0] - points[-1]) < 1e-4:
            points = points[:-1]
        if len(points) < 4:
            return
        previous_count = (
            None if self.path_points is None else len(self.path_points))
        changed = (
            previous_count is None
            or len(points) != len(self.path_points)
            or np.max(np.abs(points - self.path_points)) > 1e-6)
        if changed:
            self.set_closed_path(points)
            # A local obstacle planner updates the coordinates of the same
            # closed path as scan estimates settle. Preserve progress when
            # the topology is unchanged; a full-search reset at every scan
            # can jump to a nearby branch and create steering chatter.
            if previous_count != len(points):
                self.nearest_index = None
            self.get_logger().info(
                '%s received closed path: %d points, %.2f m'
                % (self.controller_label, len(points), self.path_length),
                throttle_duration_sec=2.0)
        self.last_path_time = self.get_clock().now()

    def set_closed_path(self, points):
        next_points = np.roll(points, -1, axis=0)
        segments = next_points - points
        segment_lengths = np.linalg.norm(segments, axis=1)
        if np.any(segment_lengths < 1e-4):
            raise RuntimeError('path contains duplicate adjacent points')

        yaw = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
        direction = 1.0 if np.sum(np.diff(yaw)) >= 0.0 else -1.0
        closed_yaw = yaw[0] + direction * 2.0 * math.pi
        while closed_yaw - yaw[-1] > math.pi:
            closed_yaw -= 2.0 * math.pi
        while closed_yaw - yaw[-1] < -math.pi:
            closed_yaw += 2.0 * math.pi
        yaw_lap_change = closed_yaw - yaw[0]
        previous_yaw = np.roll(yaw, 1)
        previous_yaw[0] = yaw[-1] - yaw_lap_change
        next_yaw = np.roll(yaw, -1)
        next_yaw[-1] = yaw[0] + yaw_lap_change
        arc_span = np.roll(segment_lengths, 1) + segment_lengths

        self.path_points = points
        self.path_yaw = yaw
        yaw_span = np.arctan2(
            np.sin(next_yaw - previous_yaw),
            np.cos(next_yaw - previous_yaw))
        self.path_curvature = (
            yaw_span / np.maximum(arc_span, 1e-6))
        # The Path message has no curvature field.  A short circular moving
        # average suppresses waypoint-discretization spikes without using any
        # map coordinate or corner-specific rule.
        self.path_speed_curvature = np.mean([
            np.roll(self.path_curvature, offset) for offset in range(-2, 3)
        ], axis=0)
        self.path_segment_lengths = segment_lengths
        self.path_cumulative = np.concatenate(
            ([0.0], np.cumsum(segment_lengths)))
        self.path_length = float(self.path_cumulative[-1])
        self.yaw_lap_change = float(yaw_lap_change)
        self.path_speed_profile = build_closed_velocity_profile(
            self.path_speed_curvature,
            self.path_segment_lengths,
            self.max_speed,
            self.max_lateral_acceleration,
            self.max_longitudinal_acceleration,
            self.max_longitudinal_deceleration,
        )

    def age_seconds(self, stamp):
        if stamp is None:
            return float('inf')
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def readiness_problem(self):
        if self.path_points is None:
            return 'no path'
        if self.age_seconds(self.last_path_time) > self.path_timeout:
            return 'path is stale'
        if self.current_odom is None:
            return 'no odometry'
        if self.age_seconds(self.last_odom_time) > self.odom_timeout:
            return 'odometry is stale'
        if self.collision:
            return 'collision input is active; clear it before starting'
        if self.emergency_stop:
            return 'local planner emergency stop is active'
        if self.kill_switch_engaged:
            return 'kill switch is engaged'
        return None

    def lookup_vehicle_pose(self):
        # This originally composed map->odom (TF) with odom->base_link taken
        # from the /odom *message*'s pose field, on the theory that AMCL's
        # map->odom correction publishes slower than wheel odometry and this
        # keeps the fed pose fresher at racing speed. That assumes the
        # Odometry message's pose.pose is a real, integrated position --
        # true in sim (f1tenth_gym_ros), but this project's real-car odom
        # source does not populate it (pose.pose stays fixed at the
        # all-zero default; only twist/velocity is real), which silently
        # froze the computed vehicle pose at whatever map->odom's own
        # translation was, regardless of where the car actually was. Use
        # the same single TF lookup pure_pursuit_node.py already relies on
        # successfully on this hardware instead.
        transform = self.tf_buffer.lookup_transform(
            self.global_frame_id, self.base_frame_id, Time(),
            timeout=Duration(seconds=0.03))
        translation = transform.transform.translation
        yaw = self.quaternion_to_yaw(transform.transform.rotation)
        return translation.x, translation.y, yaw

    def candidate_indices(self):
        count = len(self.path_points)
        if self.nearest_index is None:
            return range(count)
        return [
            (self.nearest_index + offset) % count
            for offset in range(-self.search_back_points,
                                self.search_forward_points + 1)
        ]

    def nearest_path_state(
            self, x, y, update_index=False, reference_yaw=None):
        position = np.asarray([x, y])
        best = None
        best_forward = None
        count = len(self.path_points)
        indices = self.candidate_indices() if update_index else range(count)
        for index in indices:
            segment = (
                self.path_points[(index + 1) % count]
                - self.path_points[index])
            relative = position - self.path_points[index]
            fraction = self.clamp(
                float(np.dot(relative, segment) / np.dot(segment, segment)),
                0.0, 1.0)
            projection = self.path_points[index] + fraction * segment
            delta = position - projection
            distance = float(np.linalg.norm(delta))
            candidate = (distance, index, fraction, projection)
            if best is None or distance < best[0]:
                best = candidate

            # A forward-only Ackermann vehicle cannot be following a path
            # segment whose tangent points behind it.  Prefer a directionally
            # compatible projection when nearby track branches run alongside
            # each other in opposite directions.  Fall back to pure distance
            # only when no forward segment exists in the search window.
            if (reference_yaw is not None
                    and math.cos(self.angle_difference(
                        float(self.path_yaw[index]), reference_yaw)) > 0.0
                    and (best_forward is None
                         or distance < best_forward[0])):
                best_forward = candidate

        if best_forward is not None:
            best = best_forward
        distance, index, fraction, projection = best
        path_s = (
            self.path_cumulative[index]
            + fraction * self.path_segment_lengths[index])
        heading = float(self.path_yaw[index])
        normal = np.asarray([-math.sin(heading), math.cos(heading)])
        signed_error = float(np.dot(position - projection, normal))
        if update_index:
            self.nearest_index = index
        return index, distance, heading, path_s, signed_error

    def interpolate_path(self, values, sample_s, lap_change=0.0):
        samples = np.atleast_1d(sample_s).astype(float)
        laps = np.floor(samples / self.path_length).astype(int)
        wrapped = np.mod(samples, self.path_length)
        closed_values = np.concatenate(
            (values, [values[0] + lap_change]))
        result = np.interp(wrapped, self.path_cumulative, closed_values)
        return result + laps * lap_change

    def straight_run_ahead(
            self, start_s, threshold, max_distance, samples=16):
        """Distance ahead (capped at max_distance) over which |curvature|
        stays below threshold, starting at start_s. Used only to scale
        straight_lookahead_boost -- inert while that stays at its default
        of 1.0."""
        if max_distance <= 0.0:
            return 0.0
        sample_s = np.linspace(
            start_s, start_s + max_distance, max(2, samples))
        curvature = np.abs(self.interpolate_path(
            self.path_curvature, sample_s))
        over = np.nonzero(curvature > threshold)[0]
        if len(over) == 0:
            return max_distance
        first = int(over[0])
        if first == 0:
            return 0.0
        return float(sample_s[first] - start_s)

    def compute_command(self, x, y, yaw, speed):
        if (self.last_position is not None
                and math.hypot(x - self.last_position[0],
                                y - self.last_position[1])
                > self.position_jump_reset_distance):
            self.nearest_index = None
        self.last_position = (x, y)
        _, distance, path_heading, path_s, _ = self.nearest_path_state(
            x, y, update_index=True, reference_yaw=yaw)
        heading_error = self.angle_difference(path_heading, yaw)
        if distance > self.max_path_distance:
            raise RuntimeError(
                'vehicle is %.2f m from path (limit %.2f m)'
                % (distance, self.max_path_distance))
        if abs(heading_error) > self.max_heading_error:
            raise HeadingErrorFault(
                'heading error is %.1f deg (limit %.1f deg)'
                % (math.degrees(abs(heading_error)),
                   math.degrees(self.max_heading_error)))

        # UNICORN predicts a short future vehicle pose before selecting L1.
        beta = math.atan(0.48 * math.tan(self.previous_steering))
        future_x = x + speed * math.cos(yaw + beta) * self.future_constant
        future_y = y + speed * math.sin(yaw + beta) * self.future_constant
        future_yaw = yaw + (
            speed / self.wheelbase * math.sin(beta) * self.future_constant)
        _, _, _, future_s, future_lateral_error = self.nearest_path_state(
            future_x, future_y, reference_yaw=future_yaw)

        curvature_s = np.linspace(
            future_s + self.curvature_window_start,
            future_s + self.curvature_window_end,
            num=8)
        mean_curvature = float(np.mean(np.abs(self.interpolate_path(
            self.path_curvature, curvature_s))))

        # The upstream UNICORN controller reads the local raceline speed before
        # calculating L1.  Use this path's dynamics-limited speed profile as
        # the ROS Path equivalent; the launch target is only the straight-line
        # ceiling and must not lengthen L1 inside a slower corner.
        speed_horizon = max(
            0.25, max(0.0, speed) * self.speed_lookahead)
        speed_sample_s = np.linspace(
            future_s, future_s + speed_horizon, 16)
        command_speed = float(np.min(self.interpolate_path(
            self.path_speed_profile, speed_sample_s)))
        curvature_norm = self.clamp(
            2.0 * (mean_curvature / 0.8) - 2.0, 0.0, 1.0)
        curvature_norm *= self.speed_factor_for_curvature
        lateral_norm = self.clamp(
            abs(future_lateral_error), 0.0, 1.0)
        lateral_norm *= self.speed_factor_for_lat_err
        command_speed *= (
            1.0 - self.lat_err_coeff
            + self.lat_err_coeff
            * math.exp(-lateral_norm * curvature_norm))
        threshold = math.radians(self.heading_slowdown_threshold_deg)
        if abs(heading_error) >= threshold:
            if abs(heading_error) < math.pi / 2.0:
                command_speed *= (
                    1.0 - 0.5 * abs(heading_error) / (math.pi / 2.0))
            else:
                command_speed *= 0.5
        command_speed = self.clamp(
            command_speed, self.min_reference_speed, self.max_speed)
        if command_speed > 0.0:
            command_speed = max(command_speed, self.min_command_speed)
        if self.avoidance_active:
            avoidance_limit = self.avoidance_speed_limit
            if (self.use_dynamic_speed_limit
                    and self.dynamic_speed_limit is not None
                    and self.age_seconds(self.last_speed_limit_time)
                    <= self.speed_limit_timeout):
                avoidance_limit = min(
                    avoidance_limit, self.dynamic_speed_limit)
            command_speed = min(command_speed, avoidance_limit)

        # A corner or an avoidance manoeuvre can still be limiting what the
        # LiDAR can see forward for a moment after the path straightens back
        # out. curvature_norm > 0 already marks "currently in a meaningful
        # curve" (see the speed-profile scaling above); combine that with
        # avoidance_active to mark "in a manoeuvre". The hold is triggered
        # only on the falling edge -- the moment the vehicle *leaves* a
        # manoeuvre -- not continuously while still inside one: the
        # in-manoeuvre speed is already the curvature/lateral-error profile
        # computed above, which legitimately varies (slows further into a
        # tighter section, eases as it opens up); freezing it at whatever
        # value it first took inside the manoeuvre would fight that, and
        # if the vehicle starts already inside a manoeuvre zone (e.g. an
        # obstacle right near the start pose) with previous_command_speed
        # at its initial 0.0, continuously re-arming the hold produces a
        # permanent zero/near-zero-speed deadlock -- both reproduced during
        # testing. Once truly straight (in_manoeuvre False), acceleration
        # is refused for post_maneuver_hold_time so the vehicle only speeds
        # back up after a moment to look ahead with the path actually clear.
        now = self.get_clock().now()
        in_manoeuvre = bool(self.avoidance_active or curvature_norm > 0.0)
        if in_manoeuvre:
            self.maneuver_exit_time = None
        else:
            if self.maneuver_exit_time is None:
                self.maneuver_exit_time = now
            if ((now - self.maneuver_exit_time).nanoseconds * 1e-9
                    < self.post_maneuver_hold_time):
                command_speed = min(command_speed, self.previous_command_speed)

        if speed < 2.0:
            speed_for_l1 = self.clamp(
                command_speed, max(0.0, speed - 1.0), speed + 1.0)
        else:
            speed_for_l1 = speed
        curvature_scaler = (
            self.curvature_factor * mean_curvature * speed * speed)
        raw_l1 = self.m_l1 * speed_for_l1 - curvature_scaler + self.q_l1
        if self.straight_lookahead_boost > 1.0:
            straight_ahead = self.straight_run_ahead(
                future_s, self.straight_curvature_threshold,
                self.t_clip_max)
            straight_fraction = self.clamp(
                straight_ahead
                / max(self.straight_reference_distance, 1e-3),
                0.0, 1.0)
            raw_l1 *= (
                1.0
                + (self.straight_lookahead_boost - 1.0) * straight_fraction)
        lower_l1 = max(
            self.t_clip_min,
            math.sqrt(2.0) * abs(future_lateral_error))
        l1_distance = self.clamp(raw_l1, lower_l1, self.t_clip_max)

        target_s = future_s + l1_distance
        target_x = float(self.interpolate_path(
            self.path_points[:, 0], target_s)[0])
        target_y = float(self.interpolate_path(
            self.path_points[:, 1], target_s)[0])
        vector_x = target_x - future_x
        vector_y = target_y - future_y
        vector_norm = max(math.hypot(vector_x, vector_y), 1e-6)
        eta = math.asin(self.clamp(
            (-math.sin(future_yaw) * vector_x
             + math.cos(future_yaw) * vector_y) / vector_norm,
            -1.0, 1.0))
        steering = math.atan(
            2.0 * self.wheelbase * math.sin(eta)
            / max(l1_distance, 1e-6))

        # UNICORN's filtered, speed-scaled heading correction.
        if self.filtered_heading_error is None:
            self.filtered_heading_error = eta
        else:
            alpha = self.clamp(self.heading_filter_alpha, 0.0, 1.0)
            # Heading is circular. Filtering +179 and -179 degrees as plain
            # numbers points toward zero and can reverse steering at the wrap.
            self.filtered_heading_error = self.angle_difference(
                self.filtered_heading_error
                + alpha * self.angle_difference(
                    eta, self.filtered_heading_error),
                0.0)
        gain = self.heading_kp * self.clamp(
            speed / max(self.heading_gain_speed, 1e-3), 0.0, 1.0)
        derivative = 0.0
        if self.previous_heading_error is not None:
            derivative = self.angle_difference(
                self.filtered_heading_error, self.previous_heading_error)
            derivative /= self.control_dt
        self.previous_heading_error = self.filtered_heading_error
        steering += (
            gain * self.filtered_heading_error
            + self.heading_kd * derivative)

        # Match UNICORN's lateral-error steering scaling and steering slew cap.
        # The multiplier is 2**|lateral_error|, applied to the *entire*
        # steering command (geometric L1 + heading P/D).  Left unbounded,
        # any overshoot past the line keeps re-amplifying itself back and
        # forth (each correction grows the next one's gain) -- an observed
        # left-right wobble.  Capping the error this scales from keeps the
        # intended "steer harder when far off-line" behaviour without
        # letting the loop's own gain feed the oscillation.
        scaling_error = min(
            abs(future_lateral_error), self.lateral_steering_gain_cap)
        steering *= math.exp(math.log(2.0) * scaling_error)
        steering = self.limit_steering(steering)

        return command_speed, steering, (
            future_x, future_y, target_x, target_y, l1_distance,
            mean_curvature, distance)

    def limit_steering(self, steering):
        """Apply frequency-independent steering slew and angle limits."""
        maximum_step = min(
            self.max_steering_delta,
            self.max_steering_rate * self.control_dt)
        steering = self.clamp(
            steering,
            self.previous_steering - maximum_step,
            self.previous_steering + maximum_step)
        return self.clamp(
            steering, -self.max_steering_angle, self.max_steering_angle)

    def publish_ackermann(self, publisher, speed, steering):
        if not rclpy.ok():
            return
        message = AckermannDriveStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.base_frame_id
        message.drive.speed = float(speed)
        message.drive.steering_angle = float(steering)
        try:
            publisher.publish(message)
        except RCLError:
            if rclpy.ok():
                raise

    def publish_stop(self):
        self.publish_ackermann(self.drive_pub, 0.0, 0.0)

    def publish_markers(self, geometry):
        future_x, future_y, target_x, target_y, l1_distance, _, _ = geometry
        message = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for marker_id, namespace, x, y, red, green, blue in (
                (0, 'future_pose', future_x, future_y, 0.2, 0.6, 1.0),
                (1, 'l1_target', target_x, target_y, 1.0, 0.8, 0.0)):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.global_frame_id
            marker.ns = namespace
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.08
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.12
            marker.scale.y = 0.12
            marker.scale.z = 0.12
            marker.color.a = 0.9
            marker.color.r = red
            marker.color.g = green
            marker.color.b = blue
            message.markers.append(marker)

        line = Marker()
        line.header.stamp = stamp
        line.header.frame_id = self.global_frame_id
        line.ns = 'l1_guidance'
        line.id = 2
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.035
        line.color.a = 0.9
        line.color.r = 1.0
        line.color.g = 0.8
        line.points = [
            Point(x=future_x, y=future_y, z=0.08),
            Point(x=target_x, y=target_y, z=0.08),
        ]
        message.markers.append(line)
        if not rclpy.ok():
            return
        try:
            self.marker_pub.publish(message)
        except RCLError:
            if rclpy.ok():
                raise

    def warn_throttled(self, message):
        now = self.get_clock().now()
        if (message != self.last_status_message
                or self.last_status_time is None
                or (now - self.last_status_time).nanoseconds
                > 2_000_000_000):
            self.get_logger().warn(message)
            self.last_status_message = message
            self.last_status_time = now

    def enable_callback(self, request, response):
        if not request.data:
            self.enabled = False
            self.previous_steering = 0.0
            self.previous_command_speed = 0.0
            self.publish_stop()
            response.success = True
            response.message = self.controller_label + ' stopped'
            self.get_logger().info(response.message)
            return response

        problem = self.readiness_problem()
        if problem is not None:
            response.success = False
            response.message = (
                'Cannot start ' + self.controller_label + ': ' + problem)
            self.get_logger().error(response.message)
            return response
        if not self.last_solution_ok:
            response.success = False
            response.message = (
                'Cannot start ' + self.controller_label
                + ': no valid dry-run command yet')
            self.get_logger().error(response.message)
            return response

        self.previous_command_speed = max(
            0.0, float(self.current_odom.twist.twist.linear.x))
        self.enabled = True
        response.success = True
        response.message = self.controller_label + ' enabled'
        self.get_logger().info(response.message)
        return response

    def control_loop(self):
        problem = self.readiness_problem()
        if problem is not None:
            self.last_solution_ok = False
            self.publish_stop()
            if self.enabled:
                self.warn_throttled(
                    self.controller_label + ' safety stop: ' + problem)
            return
        if not self.enabled and not self.solve_when_disabled:
            self.publish_stop()
            return

        try:
            x, y, yaw = self.lookup_vehicle_pose()
            speed = max(
                0.0, float(self.current_odom.twist.twist.linear.x))
            command_speed, steering, geometry = self.compute_command(
                x, y, yaw, speed)
            self.transform_fault_since = None
            self.heading_fault_since = None
            self.publish_markers(geometry)
            self.last_solution_ok = True
            if not self.enabled:
                self.publish_ackermann(
                    self.proposed_drive_pub, command_speed, steering)
                self.publish_stop()
                return
            command_speed = self.clamp(
                command_speed,
                max(0.0, self.previous_command_speed
                    - self.max_longitudinal_deceleration * self.control_dt),
                self.previous_command_speed
                + self.max_longitudinal_acceleration * self.control_dt)
            self.previous_command_speed = command_speed
            self.publish_ackermann(
                self.proposed_drive_pub, command_speed, steering)
            self.previous_steering = steering
            self.publish_ackermann(
                self.drive_pub, command_speed, steering)
        except TransformException as error:
            now = self.get_clock().now()
            if self.transform_fault_since is None:
                self.transform_fault_since = now
            fault_age = (
                now - self.transform_fault_since).nanoseconds * 1e-9
            if (self.enabled
                    and self.last_solution_ok
                    and fault_age <= self.transform_fault_grace):
                self.publish_ackermann(
                    self.drive_pub,
                    self.previous_command_speed,
                    self.previous_steering)
                self.warn_throttled(
                    self.controller_label
                    + ' holding last command during transient TF fault: '
                    + str(error))
                return
            self.last_solution_ok = False
            self.previous_command_speed = 0.0
            self.publish_stop()
            self.warn_throttled(
                self.controller_label + ' safety stop: ' + str(error))
        except HeadingErrorFault as error:
            now = self.get_clock().now()
            if self.heading_fault_since is None:
                self.heading_fault_since = now
            fault_age = (
                now - self.heading_fault_since).nanoseconds * 1e-9
            if (self.enabled
                    and self.last_solution_ok
                    and fault_age <= self.heading_error_grace):
                self.publish_ackermann(
                    self.drive_pub,
                    self.previous_command_speed,
                    self.previous_steering)
                self.warn_throttled(
                    self.controller_label
                    + ' holding last command during transient heading '
                    'fault: ' + str(error))
                return
            self.last_solution_ok = False
            self.previous_command_speed = 0.0
            self.publish_stop()
            self.warn_throttled(
                self.controller_label + ' safety stop: ' + str(error))
        except (RuntimeError, ValueError) as error:
            self.last_solution_ok = False
            self.previous_command_speed = 0.0
            self.publish_stop()
            self.warn_throttled(
                self.controller_label + ' safety stop: ' + str(error))


def main(args=None):
    rclpy.init(args=args)
    node = WoongPpNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        try:
            executor.shutdown()
            if rclpy.ok():
                node.publish_stop()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, RCLError):
            pass


if __name__ == '__main__':
    main()
