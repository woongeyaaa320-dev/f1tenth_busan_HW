#!/usr/bin/env python3
"""Detect points on a raceline CSV (s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,
ax_mps2) where the swept vehicle footprint clips a wall on the actual map,
and pull the path toward the (known-clear, roughly centered) centerline at
those points until clear.

The min-curvature optimizer only guarantees a `width_opt`-based margin for
the path *centerline*, which does not account for the vehicle's rectangular
body swinging into a wall corner on a tight bend -- exactly what caused the
sim closed-loop test's wall collision on the first generated raceline.
"""
import argparse
import csv
import math
import sys

import numpy as np

sys.path.insert(0, '/home/kimi/Downloads/f1tenth/scripts')
from generate_racetrack_bounds import load_map, is_occupied  # noqa: E402


def footprint_corners(x, y, psi, length, width):
    half_l, half_w = length / 2.0, width / 2.0
    cos_p, sin_p = math.cos(psi), math.sin(psi)
    local_corners = [
        (half_l, half_w), (half_l, -half_w),
        (-half_l, half_w), (-half_l, -half_w)]
    return [
        (x + lx * cos_p - ly * sin_p, y + lx * sin_p + ly * cos_p)
        for lx, ly in local_corners]


def footprint_clear(map_data, x, y, psi, length, width, samples=3):
    """Check corners plus edge midpoints of the rectangular footprint."""
    half_l, half_w = length / 2.0, width / 2.0
    cos_p, sin_p = math.cos(psi), math.sin(psi)
    for lx in np.linspace(-half_l, half_l, samples):
        for ly in (-half_w, half_w):
            px = x + lx * cos_p - ly * sin_p
            py = y + lx * sin_p + ly * cos_p
            if is_occupied(map_data, px, py):
                return False
    for ly in np.linspace(-half_w, half_w, samples):
        for lx in (-half_l, half_l):
            px = x + lx * cos_p - ly * sin_p
            py = y + lx * sin_p + ly * cos_p
            if is_occupied(map_data, px, py):
                return False
    return True


def read_raceline(path):
    with open(path, 'r', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    return rows


def read_centerline(path):
    with open(path, 'r', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        return [(float(row['x']), float(row['y'])) for row in reader]


def nearest_centerline_point(point, centerline):
    return min(centerline, key=lambda c: math.dist(c, point))


def recompute_geometry(xy, vx):
    count = len(xy)

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
        ax_values.append(0.0 if ds <= 1e-6 else vx[index] * (next_v - previous_v) / ds)

    return s_values, psi_values, kappa_values, ax_values, lap_length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raceline', required=True)
    parser.add_argument('--centerline', required=True)
    parser.add_argument('--map', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--vehicle-length', type=float, default=0.58)
    parser.add_argument('--vehicle-width', type=float, default=0.31)
    parser.add_argument(
        '--margin', type=float, default=0.05,
        help='Extra clearance beyond the bare vehicle body, in meters')
    parser.add_argument('--max-iterations', type=int, default=25)
    parser.add_argument(
        '--blend-half-window', type=int, default=15,
        help='Points on each side of a flagged point pulled toward the '
             'centerline, tapering to zero at the window edge')
    args = parser.parse_args()

    map_data = load_map(args.map)
    centerline = read_centerline(args.centerline)
    rows = read_raceline(args.raceline)
    xy = [(float(r['x_m']), float(r['y_m'])) for r in rows]
    vx = [float(r['vx_mps']) for r in rows]
    count = len(xy)
    length = args.vehicle_length + 2.0 * args.margin
    width = args.vehicle_width + 2.0 * args.margin

    for iteration in range(args.max_iterations):
        _, psi_values, _, _, _ = recompute_geometry(xy, vx)
        flagged = [
            index for index in range(count)
            if not footprint_clear(
                map_data, xy[index][0], xy[index][1], psi_values[index],
                length, width)
        ]
        if not flagged:
            print('iteration %d: clear, no flagged points' % iteration)
            break
        print('iteration %d: %d flagged points, first at index %d (%.2f, %.2f)'
              % (iteration, len(flagged), flagged[0],
                 xy[flagged[0]][0], xy[flagged[0]][1]))
        # Local moving-average smoothing around each flagged point (same
        # technique used for the peak-curvature fix), tapering to zero at
        # the window edge so it reattaches smoothly -- NOT a per-point pull
        # toward the nearest raw centerline sample, which produced a
        # non-monotonic, self-crossing path (kappa max 129 rad/m) on the
        # first attempt.
        new_xy = list(xy)
        half = args.blend_half_window
        smooth_window = 6
        for flagged_index in flagged:
            for offset in range(-half, half + 1):
                idx = (flagged_index + offset) % count
                weight = 1.0 - abs(offset) / (half + 1)
                sx = sum(
                    xy[(idx + w) % count][0]
                    for w in range(-smooth_window, smooth_window + 1)
                ) / (2 * smooth_window + 1)
                sy = sum(
                    xy[(idx + w) % count][1]
                    for w in range(-smooth_window, smooth_window + 1)
                ) / (2 * smooth_window + 1)
                new_xy[idx] = (
                    xy[idx][0] * (1 - weight) + sx * weight,
                    xy[idx][1] * (1 - weight) + sy * weight,
                )
        xy = new_xy
    else:
        print('WARNING: still flagged after %d iterations' % args.max_iterations)

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

    kappa_abs = [abs(value) for value in kappa_values]
    print('final: lap_length=%.2fm kappa mean=%.4f max=%.4f'
          % (lap_length, sum(kappa_abs) / count, max(kappa_abs)))
    print('wrote %s' % args.output)


if __name__ == '__main__':
    main()
