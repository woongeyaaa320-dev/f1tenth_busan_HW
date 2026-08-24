#!/usr/bin/env python3
"""Measure achieved acceleration at each commanded step-up in a
surface_grip_test.py bag (the counterpart to analyze_grip_bag.py's braking
report, which only looks at step-downs to zero)."""

import argparse
import sys

import numpy as np

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions

DRIVE_TOPICS = ('/auto', '/drive', '/teleop')


def read_bag(path):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=path, storage_id='sqlite3'),
        ConverterOptions(input_serialization_format='cdr',
                         output_serialization_format='cdr'))
    odom, drive = [], []
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        stamp *= 1e-9
        if topic == '/odom':
            message = deserialize_message(data, Odometry)
            odom.append((stamp, message.twist.twist.linear.x))
        elif topic in DRIVE_TOPICS:
            message = deserialize_message(data, AckermannDriveStamped)
            drive.append((stamp, message.drive.speed))
    return np.array(odom), np.array(drive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag')
    parser.add_argument('--rise-fraction', type=float, default=0.90,
                        help='fraction of the new commanded level counted '
                             'as "reached" when timing the rise')
    args = parser.parse_args()

    odom, drive = read_bag(args.bag)
    if not len(odom) or not len(drive):
        sys.exit('bag is missing /odom or a drive topic')

    stamps = odom[:, 0]
    speed = odom[:, 1]
    d_stamps = drive[:, 0]
    commanded = drive[:, 1]

    # Find step-up events: commanded jumps up and holds (grip test steps are
    # constant-level holds, not ramps).
    rising = np.flatnonzero(commanded[1:] > commanded[:-1] + 0.15)
    print('%d samples odom, %d samples drive, %d step-up events\n'
          % (len(odom), len(drive), len(rising)))
    print('%-8s %-8s %8s %10s' % ('from', 'to', 'rise_t', 'accel(m/s^2)'))
    print('-' * 40)

    rates = []
    for idx in rising:
        t0 = d_stamps[idx + 1]
        level_from = commanded[idx]
        level_to = commanded[idx + 1]
        target = level_from + args.rise_fraction * (level_to - level_from)

        window = (stamps >= t0) & (stamps <= t0 + 5.0)
        if not window.sum():
            continue
        t_window = stamps[window]
        v_window = speed[window]

        reached = np.flatnonzero(v_window >= target)
        if not len(reached):
            print('%-8.2f %-8.2f %8s %10s' % (
                level_from, level_to, 'n/a', 'never reached %.2f' % target))
            continue
        t_reach = t_window[reached[0]]
        v_start = v_window[0]
        rise_t = t_reach - t0
        if rise_t < 0.02:
            continue
        rate = (target - v_start) / rise_t
        rates.append(rate)
        print('%-8.2f %-8.2f %8.2f %10.2f' % (
            level_from, level_to, rise_t, rate))

    if rates:
        rates = np.array(rates)
        print('\nn=%d  median=%.2f  max=%.2f m/s^2' % (
            len(rates), np.median(rates), rates.max()))
        print('max_longitudinal_acceleration 권장: 중앙값의 80%%인 %.2f'
              % (0.8 * np.median(rates)))
    else:
        print('\nno usable step-up rise measured')


if __name__ == '__main__':
    main()
