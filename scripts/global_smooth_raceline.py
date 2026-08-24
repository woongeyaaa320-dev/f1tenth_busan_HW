#!/usr/bin/env python3
"""Apply one uniform moving-average smoothing pass to a raceline's x,y
points (closed loop), then verify both vehicle-footprint map clearance and
a curvature/steering-limit ceiling. Chosen over the earlier per-point local
patches: those fixed one constraint at one spot while regularly creating a
new curvature violation at a different spot (whack-a-mole). A single global
pass trades a small amount of optimizer-computed curvature-minimality
everywhere for uniformly better clearance and smoothness track-wide.
"""
import argparse
import csv
import math
import sys

import numpy as np

sys.path.insert(0, '/home/kimi/Downloads/f1tenth/scripts')
from generate_racetrack_bounds import load_map, is_occupied  # noqa: E402
from enforce_footprint_clearance import footprint_clear  # noqa: E402


def circle_curvature(a, b, c):
    ab, bc, ac = math.dist(a, b), math.dist(b, c), math.dist(a, c)
    denom = ab * bc * ac
    if denom < 1e-9:
        return 0.0
    twice_area = abs(
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return 2.0 * twice_area / denom


def signed_curvature(a, b, c):
    magnitude = circle_curvature(a, b, c)
    v1x, v1y = b[0] - a[0], b[1] - a[1]
    v2x, v2y = c[0] - b[0], c[1] - b[1]
    cross = v1x * v2y - v1y * v2x
    return magnitude if cross >= 0.0 else -magnitude


def recompute_geometry(xy, vx):
    count = len(xy)
    s_values = [0.0] * count
    for index in range(1, count):
        s_values[index] = s_values[index - 1] + math.dist(
            xy[index - 1], xy[index])
    lap_length = s_values[-1] + math.dist(xy[-1], xy[0])

    psi_values, kappa_values = [], []
    for index in range(count):
        previous = xy[(index - 1) % count]
        current = xy[index]
        following = xy[(index + 1) % count]
        psi_values.append(math.atan2(
            following[1] - previous[1], following[0] - previous[0]))
        kappa_values.append(signed_curvature(previous, current, following))

    ax_values = []
    for index in range(count):
        previous_v = vx[(index - 1) % count]
        next_v = vx[(index + 1) % count]
        previous_s = s_values[(index - 1) % count]
        next_s = (
            s_values[(index + 1) % count]
            if index + 1 < count else lap_length)
        ds = next_s - previous_s
        if index == 0:
            ds = s_values[1] + (lap_length - s_values[-1])
        ax_values.append(
            0.0 if ds <= 1e-6 else vx[index] * (next_v - previous_v) / ds)
    return s_values, psi_values, kappa_values, ax_values, lap_length


def moving_average_smooth(xy, window):
    count = len(xy)
    return [
        (
            sum(xy[(index + w) % count][0]
                for w in range(-window, window + 1)) / (2 * window + 1),
            sum(xy[(index + w) % count][1]
                for w in range(-window, window + 1)) / (2 * window + 1),
        )
        for index in range(count)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raceline', required=True)
    parser.add_argument('--map', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--vehicle-length', type=float, default=0.58)
    parser.add_argument('--vehicle-width', type=float, default=0.31)
    parser.add_argument('--margin', type=float, default=0.05)
    parser.add_argument(
        '--physical-curvlim', type=float, default=1.374,
        help='tan(max_steering_angle)/wheelbase for this vehicle')
    parser.add_argument('--window', type=int, default=6)
    parser.add_argument('--max-passes', type=int, default=10)
    args = parser.parse_args()

    map_data = load_map(args.map)
    with open(args.raceline, 'r', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    xy = [(float(r['x_m']), float(r['y_m'])) for r in rows]
    vx = [float(r['vx_mps']) for r in rows]
    count = len(xy)
    length = args.vehicle_length + 2.0 * args.margin
    width = args.vehicle_width + 2.0 * args.margin
    curvlim = args.physical_curvlim * 0.90  # small safety margin

    for pass_index in range(1, args.max_passes + 1):
        xy = moving_average_smooth(xy, args.window)
        _, psi_values, kappa_values, _, _ = recompute_geometry(xy, vx)
        footprint_bad = sum(
            0 if footprint_clear(
                map_data, xy[i][0], xy[i][1], psi_values[i], length, width)
            else 1
            for i in range(count))
        curvature_bad = sum(
            1 for k in kappa_values if abs(k) > curvlim)
        max_kappa = max(abs(k) for k in kappa_values)
        print('pass %d: footprint_bad=%d curvature_bad=%d max_kappa=%.3f'
              % (pass_index, footprint_bad, curvature_bad, max_kappa))
        if footprint_bad == 0 and curvature_bad == 0:
            break

    s_values, psi_values, kappa_values, ax_values, lap_length = (
        recompute_geometry(xy, vx))
    with open(args.output, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ['s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm', 'vx_mps',
             'ax_mps2'])
        for index in range(count):
            writer.writerow([
                '%.8f' % s_values[index], '%.8f' % xy[index][0],
                '%.8f' % xy[index][1], '%.8f' % psi_values[index],
                '%.8f' % kappa_values[index], '%.8f' % vx[index],
                '%.8f' % ax_values[index]])

    kappa_abs = [abs(k) for k in kappa_values]
    footprint_bad = sum(
        0 if footprint_clear(
            map_data, xy[i][0], xy[i][1], psi_values[i], length, width)
        else 1
        for i in range(count))
    print('FINAL: lap_length=%.2fm kappa mean=%.4f max=%.4f '
          'footprint_bad=%d curvlim=%.3f'
          % (lap_length, sum(kappa_abs) / count, max(kappa_abs),
             footprint_bad, curvlim))
    print('wrote %s' % args.output)


if __name__ == '__main__':
    main()
