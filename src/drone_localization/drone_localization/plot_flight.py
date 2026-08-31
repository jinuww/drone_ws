#!/usr/bin/env python3
"""
Post-flight visualiser.

Reads the CSVs written by flight_recorder and shows, in the Gazebo world
frame (ENU):
  • top-down XY: planned vs actual vs LiDAR-estimated
  • altitude (Z) vs time
  • 3D trajectory
and prints a rough localization-error summary (estimated vs actual).

The planned trajectory + markers are imported from the drone_mission
package if it is on the path; otherwise that overlay is skipped.

Usage:
  ros2 run drone_localization plot_flight
  # options:
  ros2 run drone_localization plot_flight --ros-args -p output_dir:=/path -p save_only:=true
"""

import argparse
import csv
import os
import sys

import numpy as np


def _load_csv(path):
    if not os.path.isfile(path):
        return None
    t, x, y, z = [], [], [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 4:
                continue
            t.append(float(row[0]))
            x.append(float(row[1]))
            y.append(float(row[2]))
            z.append(float(row[3]))
    if not t:
        return None
    return (np.array(t), np.array(x), np.array(y), np.array(z))


def _load_planned():
    """Return (xs, ys, zs, markers_dict, home_xy) or None if unavailable."""
    try:
        from drone_mission.generate_groundtruth_trajectory import (
            generate_trajectory, MARKERS, HOME,
        )
    except Exception:
        return None
    try:
        samples = generate_trajectory()
        xs = np.array([s.x for s in samples])
        ys = np.array([s.y for s in samples])
        zs = np.array([s.z for s in samples])
        return xs, ys, zs, MARKERS, HOME
    except Exception:
        return None


def _localization_error(actual, estimate):
    """RMS/median error by interpolating actual onto estimate timestamps.

    The two log streams may use different clocks / start times (PX4 logs
    from boot on the ground; the LiDAR estimate only starts once the drone
    lifts off and is detected). So we FIRST recover the time offset δ that
    best aligns them (minimising horizontal error), then report the error
    at that alignment. Returns a dict with per-axis breakdown.
    """
    ta, xa, ya, za = actual
    te, xe, ye, ze = estimate
    if len(ta) < 2 or len(te) < 2:
        return None

    def err_at(delta):
        ts = te + delta
        m = (ts >= ta[0]) & (ts <= ta[-1])
        if m.sum() < 5:
            return None, None
        xa_i = np.interp(ts[m], ta, xa)
        ya_i = np.interp(ts[m], ta, ya)
        za_i = np.interp(ts[m], ta, za)
        exy = np.sqrt((xe[m] - xa_i) ** 2 + (ye[m] - ya_i) ** 2)
        ez = ze[m] - za_i
        return exy, ez

    # Search the time offset that minimises median horizontal error.
    # Range ±30s covers the boot-vs-takeoff gap; 0.1s step.
    best = None
    for delta in np.arange(-30.0, 30.01, 0.1):
        exy, ez = err_at(delta)
        if exy is None:
            continue
        score = np.median(exy)
        if best is None or score < best[0]:
            best = (score, delta, exy, ez)
    if best is None:
        return None

    _, delta, exy, ez = best
    return {
        'offset': delta,
        'n': len(exy),
        'xy_median': float(np.median(exy)),
        'xy_rms': float(np.sqrt(np.mean(exy ** 2))),
        'xy_max': float(np.max(exy)),
        'z_bias': float(np.mean(ez)),        # signed: + = estimate reads high
        'z_std': float(np.std(ez)),
    }


def main():
    # Minimal arg parsing (also tolerate ROS args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        default=os.path.expanduser('~/drone_project/flight_logs'))
    parser.add_argument('--save_only', action='store_true')
    args, _ = parser.parse_known_args()

    # allow -p output_dir:=... style too
    for a in sys.argv:
        if a.startswith('output_dir:='):
            args.output_dir = a.split(':=', 1)[1]
        if a.startswith('save_only:=') and a.split(':=', 1)[1].lower() == 'true':
            args.save_only = True

    out_dir = args.output_dir
    actual = _load_csv(os.path.join(out_dir, 'actual_path.csv'))
    estimate = _load_csv(os.path.join(out_dir, 'estimated_path.csv'))
    planned = _load_planned()

    if actual is None and estimate is None:
        print(f'기록된 CSV가 없어: {out_dir}\n'
              'flight_recorder를 먼저 켜고 비행한 뒤 다시 실행해.')
        return

    import matplotlib
    if args.save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 3D needs mpl_toolkits.mplot3d, which can clash with a stale system
    # copy. Try it; if unavailable, fall back to a 2D-only layout so the
    # important top-down + altitude views always render.
    ax_3d = None
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(15, 8))
        ax_xy = fig.add_subplot(1, 2, 1)
        ax_z = fig.add_subplot(2, 2, 2)
        ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
    except Exception as e:
        print(f'3D 뷰 비활성화(mpl_toolkits 충돌): {e}\n→ 2D(평면+고도)만 표시.')
        fig = plt.figure(figsize=(14, 6))
        ax_xy = fig.add_subplot(1, 2, 1)
        ax_z = fig.add_subplot(1, 2, 2)

    # --- Top-down XY ---
    if planned is not None:
        pxs, pys, pzs, markers, home = planned
        ax_xy.plot(pxs, pys, '--', color='0.6', lw=1.5,
                   label='Planned (ground-truth)')
        for mid, (mx, my) in markers.items():
            ax_xy.plot(mx, my, 's', color='orange', ms=12, mec='k', zorder=5)
            ax_xy.annotate(str(mid), (mx, my), color='k', fontsize=9,
                           ha='center', va='center', zorder=6)
        ax_xy.plot(home[0], home[1], 'o', color='green', ms=12, mec='k',
                   label='Home', zorder=5)

    if actual is not None:
        ax_xy.plot(actual[1], actual[2], '-', color='tab:blue', lw=1.8,
                   label='Actual flight (PX4)')
    if estimate is not None:
        ax_xy.plot(estimate[1], estimate[2], ':', color='tab:red', lw=1.8,
                   label='LiDAR estimate')

    ax_xy.set_xlabel('world X [m]')
    ax_xy.set_ylabel('world Y [m]')
    ax_xy.set_title('Top-down trajectory (Gazebo world / ENU)')
    ax_xy.set_aspect('equal', adjustable='datalim')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc='best', fontsize=8)

    # --- Altitude vs time ---
    if actual is not None:
        ax_z.plot(actual[0], actual[3], '-', color='tab:blue',
                  label='Actual Z')
    if estimate is not None:
        ax_z.plot(estimate[0], estimate[3], ':', color='tab:red',
                  label='Estimate Z')
    ax_z.set_xlabel('time [s]')
    ax_z.set_ylabel('altitude Z [m]')
    ax_z.set_title('Altitude vs time')
    ax_z.grid(True, alpha=0.3)
    ax_z.legend(loc='best', fontsize=8)

    # --- 3D (optional) ---
    if ax_3d is not None:
        if planned is not None:
            ax_3d.plot(pxs, pys, pzs, '--', color='0.6', lw=1.0)
        if actual is not None:
            ax_3d.plot(actual[1], actual[2], actual[3], '-', color='tab:blue', lw=1.5)
        if estimate is not None:
            ax_3d.plot(estimate[1], estimate[2], estimate[3], ':', color='tab:red', lw=1.5)
        ax_3d.set_xlabel('X [m]')
        ax_3d.set_ylabel('Y [m]')
        ax_3d.set_zlabel('Z [m]')
        ax_3d.set_title('3D trajectory')

    # --- Error summary (time-aligned) ---
    if actual is not None and estimate is not None:
        e = _localization_error(actual, estimate)
        if e is not None:
            txt = (
                'LiDAR localization error vs PX4 (time-aligned, '
                f'offset {e["offset"]:+.1f}s, n={e["n"]}):\n'
                f'  HORIZONTAL(XY): median={e["xy_median"]:.3f} m  '
                f'RMS={e["xy_rms"]:.3f} m  max={e["xy_max"]:.3f} m   |   '
                f'Z bias={e["z_bias"]:+.3f} m (std {e["z_std"]:.3f})')
            print(txt)
            fig.suptitle(txt, fontsize=9)
    else:
        missing = 'actual' if actual is None else 'estimate'
        print(f'주의: {missing} 경로가 비어있어 오차 비교는 생략함.')

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_png = os.path.join(out_dir, 'flight_plot.png')
    fig.savefig(out_png, dpi=130)
    print(f'그림 저장: {out_png}')

    if not args.save_only:
        try:
            plt.show()
        except Exception as e:
            print(f'창을 못 열었어({e}). 저장된 PNG를 열어봐: {out_png}')


if __name__ == '__main__':
    main()
