#!/usr/bin/env python3
"""Convert the TUM/CL2-UWaterloo optimizer's minimal (x_m,y_m,vx_mps) export
into this project's s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2 format,
already used by algorithms/planning/waypoints/track03_raceline.csv and
consumed by planning/waypoint_planner_node's flexible CSV loader.

The optimizer variant used here (main_globaltraj_f110.py's
export_traj_race_f110) only writes x/y/v -- no header, no s/psi/kappa/ax.
Curvature is computed with the same three-point circle-fit method as
control/control/pure_pursuit_node.py's PurePursuitNode.circle_curvature, for
consistency with the rest of this codebase rather than introducing a second
curvature formula.
"""
import argparse
import csv
import math


def circle_curvature(first, middle, last):
    """Unsigned curvature magnitude of the circle through three XY points.
    Mirrors control/control/pure_pursuit_node.py's static method exactly."""
    a = math.dist(first, middle)
    b = math.dist(middle, last)
    c = math.dist(first, last)
    denominator = a * b * c
    if denominator < 1e-9:
        return 0.0
    twice_area = abs(
        (middle[0] - first[0]) * (last[1] - first[1])
        - (middle[1] - first[1]) * (last[0] - first[0]))
    return 2.0 * twice_area / denominator


def signed_curvature(first, middle, last):
    magnitude = circle_curvature(first, middle, last)
    # Cross product sign of the two chord vectors gives turn direction
    # (positive = left turn), matching this project's psi_rad/kappa_radpm
    # sign convention already present in track03_raceline.csv.
    v1x, v1y = middle[0] - first[0], middle[1] - first[1]
    v2x, v2y = last[0] - middle[0], last[1] - middle[1]
    cross = v1x * v2y - v1y * v2x
    return magnitude if cross >= 0.0 else -magnitude


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='x_m,y_m,vx_mps CSV, closed (first==last row)')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    points = []
    with open(args.input, 'r', encoding='utf-8') as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            points.append((float(row[0]), float(row[1]), float(row[2])))

    if len(points) > 1 and math.dist(points[0][:2], points[-1][:2]) < 1e-6:
        points = points[:-1]
    count = len(points)
    if count < 4:
        raise RuntimeError('not enough points to build a closed raceline')

    xy = [(p[0], p[1]) for p in points]
    vx = [p[2] for p in points]

    s_values = [0.0] * count
    for index in range(1, count):
        s_values[index] = s_values[index - 1] + math.dist(
            xy[index - 1], xy[index])
    lap_length = s_values[-1] + math.dist(xy[-1], xy[0])

    psi_values = []
    kappa_values = []
    for index in range(count):
        previous = xy[(index - 1) % count]
        current = xy[index]
        following = xy[(index + 1) % count]
        psi_values.append(math.atan2(
            following[1] - previous[1], following[0] - previous[0]))
        kappa_values.append(
            signed_curvature(previous, current, following))

    ax_values = []
    for index in range(count):
        previous_v = vx[(index - 1) % count]
        next_v = vx[(index + 1) % count]
        previous_s = s_values[(index - 1) % count]
        next_s = s_values[(index + 1) % count] if index + 1 < count else lap_length
        ds = next_s - previous_s
        if index == 0:
            ds = s_values[1] + (lap_length - s_values[-1])
        if ds <= 1e-6:
            ax_values.append(0.0)
            continue
        # a = v * dv/ds (constant-acceleration-per-arclength convention,
        # matching the ax_mps2 sign/units already used in track03_raceline.csv)
        ax_values.append(vx[index] * (next_v - previous_v) / ds)

    with open(args.output, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ['s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm', 'vx_mps',
             'ax_mps2'])
        for index in range(count):
            writer.writerow([
                '%.8f' % s_values[index],
                '%.8f' % xy[index][0],
                '%.8f' % xy[index][1],
                '%.8f' % psi_values[index],
                '%.8f' % kappa_values[index],
                '%.8f' % vx[index],
                '%.8f' % ax_values[index],
            ])

    kappa_abs = [abs(value) for value in kappa_values]
    print('points: %d, lap length: %.2f m' % (count, lap_length))
    print('kappa mean=%.4f max=%.4f rad/m'
          % (sum(kappa_abs) / count, max(kappa_abs)))
    print('vx min=%.2f mean=%.2f max=%.2f m/s'
          % (min(vx), sum(vx) / count, max(vx)))
    print('wrote %s' % args.output)


if __name__ == '__main__':
    main()
