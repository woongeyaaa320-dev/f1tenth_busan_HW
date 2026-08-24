#!/usr/bin/env python3
"""Generate a TUM-format track-bounds CSV (x_m,y_m,w_tr_right_m,w_tr_left_m)
for the CL2-UWaterloo/TUM global raceline optimizer, from an existing
centerline CSV (x,y,yaw,curvature,speed -- this project's
planning/waypoints/<track>_centerline.csv format) plus the occupancy-grid map
used to build it.

For every centerline point, this walks outward along the left/right path
normals in map-resolution steps until it hits an occupied cell (or a search
radius limit), the same "ray to nearest wall" idea used by any map-based
track-width extraction -- simpler than re-deriving it from the live-obstacle
EDT/clearance code in planning/local_planner_core.py, which is built for scan
clusters, not a static map.

Usage:
    python3 generate_racetrack_bounds.py \
        --centerline algorithms/planning/waypoints/track03_centerline.csv \
        --map maps/track03.yaml \
        --output /home/kimi/Raceline-Optimization/inputs/tracks/track03.csv
"""
import argparse
import csv
import math
import os

import numpy as np
import yaml
from PIL import Image


def load_map(map_yaml_path):
    with open(map_yaml_path, 'r', encoding='utf-8') as stream:
        meta = yaml.safe_load(stream)
    image_path = meta['image']
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(map_yaml_path), image_path)
    image = np.asarray(Image.open(image_path).convert('L'), dtype=np.uint8)
    negate = int(meta.get('negate', 0))
    occupied_thresh = float(meta.get('occupied_thresh', 0.65))
    if negate:
        occupancy = image.astype(float) / 255.0
    else:
        occupancy = (255.0 - image.astype(float)) / 255.0
    occupied_mask = occupancy > occupied_thresh
    return {
        'occupied': occupied_mask,
        'resolution': float(meta['resolution']),
        'origin_x': float(meta['origin'][0]),
        'origin_y': float(meta['origin'][1]),
        'height': image.shape[0],
        'width': image.shape[1],
    }


def world_to_pixel(map_data, x, y):
    col = int(round((x - map_data['origin_x']) / map_data['resolution']))
    row = int(round(
        map_data['height'] - 1
        - (y - map_data['origin_y']) / map_data['resolution']))
    return row, col


def is_occupied(map_data, x, y):
    row, col = world_to_pixel(map_data, x, y)
    if row < 0 or row >= map_data['height'] or col < 0 or col >= map_data['width']:
        return True
    return bool(map_data['occupied'][row, col])


def distance_to_wall(map_data, x, y, direction_x, direction_y,
                      max_radius, step):
    distance = 0.0
    while distance < max_radius:
        distance += step
        px = x + direction_x * distance
        py = y + direction_y * distance
        if is_occupied(map_data, px, py):
            return max(0.0, distance - step * 0.5)
    return max_radius


def read_centerline(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            rows.append((
                float(row['x']), float(row['y']), float(row['yaw']),
                float(row['curvature'])))
    return rows


def smoothed_curvature(curvatures, window=5):
    """Centered moving-average of |curvature|, wrapping around the closed
    track, to avoid clamping width against a single noisy finite-difference
    curvature sample at a raw centerline vertex."""
    values = np.abs(np.asarray(curvatures, dtype=float))
    count = len(values)
    half = window // 2
    smoothed = np.zeros(count)
    for index in range(count):
        window_values = [
            values[(index + offset) % count]
            for offset in range(-half, half + 1)]
        smoothed[index] = float(np.mean(window_values))
    return smoothed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--centerline', required=True)
    parser.add_argument('--map', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--max-radius', type=float, default=2.0,
        help='Furthest distance to search for a wall, in meters')
    parser.add_argument(
        '--min-half-width', type=float, default=0.20,
        help='Floor applied to each side to avoid a zero-width point from '
             'a single noisy map pixel')
    parser.add_argument(
        '--wall-safety-margin', type=float, default=0.10,
        help='Extra clearance subtracted from every side beyond the raw '
             'map-measured width, so the optimizer keeps a real margin '
             'between the vehicle body (0.31 m wide) and a wall rather than '
             'running the reference line right up to the boundary it was '
             'given (a first pass with no margin produced a raceline whose '
             'closest approach was only 0.14 m from a wall and the sim '
             'closed-loop test collided there).')
    parser.add_argument(
        '--curvature-safety-factor', type=float, default=0.80,
        help='Fraction of the local (smoothed) reference-line turn radius '
             'each side may claim. The min-curvature optimizer needs the '
             'left/right boundary offsets to never cross the path centre '
             'of curvature; a raw map-measured width can exceed that at a '
             'sharp bend even though the physical track is fine, so this '
             'clamps width there instead of leaving it to fail deep inside '
             'the optimizer.')
    args = parser.parse_args()

    map_data = load_map(args.map)
    centerline = read_centerline(args.centerline)
    curvature = smoothed_curvature(
        [point[3] for point in centerline], window=9)
    step = map_data['resolution'] * 0.5

    verify_bad = 0
    clamped_by_curvature = 0
    rows = []
    for (x, y, yaw, _), kappa in zip(centerline, curvature):
        if is_occupied(map_data, x, y):
            verify_bad += 1
        # Left/right normals of the path heading.
        left_dx, left_dy = -math.sin(yaw), math.cos(yaw)
        right_dx, right_dy = math.sin(yaw), -math.cos(yaw)
        w_left = distance_to_wall(
            map_data, x, y, left_dx, left_dy, args.max_radius, step)
        w_right = distance_to_wall(
            map_data, x, y, right_dx, right_dy, args.max_radius, step)
        w_left -= args.wall_safety_margin
        w_right -= args.wall_safety_margin
        floor = args.min_half_width
        if kappa > 1e-3:
            radius_limit = args.curvature_safety_factor / kappa
            if w_left > radius_limit or w_right > radius_limit:
                clamped_by_curvature += 1
            w_left = min(w_left, radius_limit)
            w_right = min(w_right, radius_limit)
            # The curvature ceiling always wins over the floor: a sharp
            # bend that geometrically cannot support min_half_width on a
            # side must end up narrower there, not re-inflated past the
            # point where the boundary offset would cross the path.
            floor = min(floor, radius_limit)
        w_left = max(floor, w_left)
        w_right = max(floor, w_right)
        rows.append((x, y, w_right, w_left))

    if verify_bad:
        raise RuntimeError(
            '%d/%d centerline points land on an occupied map cell -- map '
            'origin/resolution/negate convention is probably wrong; refusing '
            'to emit a bounds file that would be silently unsafe.'
            % (verify_bad, len(centerline)))

    widths = np.array([(row[2], row[3]) for row in rows])
    print('centerline points: %d' % len(rows))
    print('w_tr_right_m: min=%.3f mean=%.3f max=%.3f'
          % (widths[:, 0].min(), widths[:, 0].mean(), widths[:, 0].max()))
    print('w_tr_left_m:  min=%.3f mean=%.3f max=%.3f'
          % (widths[:, 1].min(), widths[:, 1].mean(), widths[:, 1].max()))
    total_width = widths[:, 0] + widths[:, 1]
    print('total width:  min=%.3f mean=%.3f max=%.3f'
          % (total_width.min(), total_width.mean(), total_width.max()))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8', newline='') as stream:
        stream.write('# x_m,y_m,w_tr_right_m,w_tr_left_m\n')
        writer = csv.writer(stream)
        for row in rows:
            writer.writerow(['%.6f' % value for value in row])
    print('wrote %s' % args.output)


if __name__ == '__main__':
    main()
