#!/usr/bin/env python3
"""런 하나를 그림 네 장으로 확인한다.

GUI가 ~/competition_results/<팀>_<ID>/run_NN/ 에 남긴 CSV를 읽어서 평면 궤적,
고도, 필터 불확실성, 샘플 간격을 한 화면에 그리고 같은 폴더에 trajectory.png
로 저장한다. 인자 없이 부르면 가장 최근 런을 집는다.

    ros2 run drone_localization plot_run
    ros2 run drone_localization plot_run --list
    ros2 run drone_localization plot_run --run ~/competition_results/팀_ID/run_01
    ros2 run drone_localization plot_run --save-only

ROI 사각형과 LiDAR 위치는 params yaml에서 읽는다. 하드코딩하면 파라미터를
바꾼 뒤 그림이 조용히 거짓말을 하게 된다.

읽는 파일 (drone_trajectory.csv 만 필수):
    drone_trajectory.csv    KF 출력. 이 그림의 본체
    waypoint_estimates.csv  있으면 평면도에 경로점 추정을 겹쳐 그린다

네 장을 각각 어떻게 읽는지:

  평면 궤적   점 색이 시간이다. ROI 점선 밖으로 나가는 구간이 있으면 거기서
              관측이 끊긴 것이므로 ROI부터 의심한다.
  고도        회색 띠가 ROI z 범위다. 띠 경계에 붙어 다니면 z를 넓혀야 한다.
  불확실성    로그 축이다. 0.01을 넘는 구간이 코스팅 -- 관측 없이 등속으로
              밀고 있다는 뜻이라 그 구간 위치는 측정이 아니라 추정이다.
  샘플 간격   100ms(10Hz)가 목표선. 빨간 막대가 많으면 파이프라인이 밀리는
              것이고, 간격이 벌어진 만큼 속도 추정이 과장된다.
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

DEFAULT_RESULTS = os.path.expanduser('~/competition_results')
TRAJ = 'drone_trajectory.csv'
WAYPOINTS = 'waypoint_estimates.csv'


def find_runs(base):
    runs = glob.glob(os.path.join(base, '*', 'run_*'))
    runs = [r for r in runs if os.path.isfile(os.path.join(r, TRAJ))]
    return sorted(runs, key=os.path.getmtime, reverse=True)


def read_traj(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r.get('time_s')]
    if len(rows) < 2:
        return None
    col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    d = {k: col(k) for k in
         ('time_s', 'x_m', 'y_m', 'z_m', 'vx_mps', 'vy_mps', 'vz_mps',
          'cov_x', 'cov_y', 'cov_z')}
    d['t'] = d['time_s'] - d['time_s'][0]
    return d


def read_waypoints(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [(int(r['marker_id']), int(r['visit']),
                 float(r['x']), float(r['y']), float(r['z']))
                for r in csv.DictReader(f) if r.get('marker_id')]


def read_params(path):
    """ROI와 LiDAR 좌표만 뽑는다. 없으면 오버레이를 생략한다."""
    try:
        import yaml
        with open(path) as f:
            p = yaml.safe_load(f)['lidar_drone_tracker']['ros__parameters']
    except Exception:
        return None
    try:
        return {
            'roi': [(p['roi_x_min'], p['roi_x_max']),
                    (p['roi_y_min'], p['roi_y_max']),
                    (p['roi_z_min'], p['roi_z_max'])],
            'lidars': [(p[f'lidar{i}_x'], p[f'lidar{i}_y'])
                       for i in (1, 2) if f'lidar{i}_x' in p],
        }
    except KeyError:
        return None


def default_params_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('drone_localization')
        return os.path.join(share, 'params', 'lidar_localization_os1.yaml')
    except Exception:
        return None


def plot(run_dir, params, save_only):
    import matplotlib
    if save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    d = read_traj(os.path.join(run_dir, TRAJ))
    if d is None:
        print(f'샘플이 부족합니다: {run_dir}')
        return None

    t, x, y, z = d['t'], d['x_m'], d['y_m'], d['z_m']
    cov = d['cov_x']
    speed = np.hypot(d['vx_mps'], d['vy_mps'])
    gaps = np.diff(t)
    wps = read_waypoints(os.path.join(run_dir, WAYPOINTS))

    team = os.path.basename(os.path.dirname(run_dir))
    run = os.path.basename(run_dir)
    coasting = int((cov > 0.01).sum())

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        f'{team} / {run}   ·   {len(t)} samples · {t[-1]:.1f}s · '
        f'{1.0 / np.mean(gaps):.1f} Hz avg · coasting {coasting}/{len(t)}',
        fontsize=13)

    # --- 평면 궤적 -------------------------------------------------
    ax = fig.add_subplot(2, 2, 1)
    if params:
        (rx0, rx1), (ry0, ry1), _ = params['roi']
        ax.add_patch(plt.Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0,
                                   fc='none', ec='0.55', ls='--', lw=1.2,
                                   label='ROI'))
        for i, (lx, ly) in enumerate(params['lidars'], start=1):
            ax.plot(lx, ly, 'r^', ms=12, zorder=5)
            ax.annotate(f'lidar{i}', (lx, ly), textcoords='offset points',
                        xytext=(8, 4), color='r', fontsize=8)
    seg = np.stack([np.c_[x[:-1], y[:-1]], np.c_[x[1:], y[1:]]], axis=1)
    lc = LineCollection(seg, cmap='viridis', array=t[:-1], lw=2)
    ax.add_collection(lc)
    ax.scatter(x, y, c=t, cmap='viridis', s=16, zorder=3,
               edgecolor='w', linewidth=0.4)
    ax.plot(0, 0, 'k+', ms=12, mew=2, zorder=5)
    ax.annotate('origin', (0, 0), textcoords='offset points', xytext=(6, 6),
                fontsize=8)
    ax.plot(x[0], y[0], 'o', c='#00a000', ms=11, zorder=6)
    ax.plot(x[-1], y[-1], 's', c='#d62728', ms=10, zorder=6)
    for mid, visit, wx, wy, _wz in wps:
        ax.plot(wx, wy, '*', c='m', ms=15, zorder=7)
        ax.annotate(f'{mid}#{visit}', (wx, wy), textcoords='offset points',
                    xytext=(7, -10), color='m', fontsize=8)
    fig.colorbar(lc, ax=ax, label='time (s)')
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Top-down XY  (green=start, red=end)')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)

    # --- 고도 ------------------------------------------------------
    ax = fig.add_subplot(2, 2, 2)
    if params:
        z0, z1 = params['roi'][2]
        ax.axhspan(z0, z1, color='0.87', label='ROI z-band')
        ax.legend(fontsize=8)
    ax.plot(t, z, '-o', ms=3.5, lw=1.4, color='#1f77b4')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('z (m)')
    ax.set_title('Altitude')
    ax.grid(alpha=0.3)

    # --- 불확실성 --------------------------------------------------
    ax = fig.add_subplot(2, 2, 3)
    ax.semilogy(t, cov, '-o', ms=3.5, lw=1.4, color='#d62728')
    ax.axhline(0.01, ls='--', c='0.5', lw=1)
    ax.annotate('coasting threshold', (t[-1], 0.01), ha='right', va='bottom',
                fontsize=8, color='0.4')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('cov_x (m^2), log')
    ax.set_title('Filter uncertainty')
    ax.grid(alpha=0.3, which='both')

    # --- 샘플 간격 --------------------------------------------------
    ax = fig.add_subplot(2, 2, 4)
    width = max(0.05, 0.6 * float(np.median(gaps)))
    ax.bar(t[1:], gaps * 1000.0, width=width,
           color=np.where(gaps > 0.3, '#d62728', '#2ca02c'))
    ax.axhline(100, ls='--', c='k', lw=1)
    ax.annotate('100 ms (10 Hz target)', (0, 100), va='bottom', fontsize=8)
    ax.set_xlabel('t (s)')
    ax.set_ylabel('gap to previous sample (ms)')
    ax.set_title(f'Sample spacing — median {np.median(gaps) * 1000:.0f} ms, '
                 f'max {gaps.max() * 1000:.0f} ms')
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(run_dir, 'trajectory.png')
    fig.savefig(out, dpi=130)

    print(f'런        : {team} / {run}')
    print(f'샘플      : {len(t)}개, {t[-1]:.1f}초, 평균 {1.0/np.mean(gaps):.1f} Hz')
    print(f'간격      : 중앙 {np.median(gaps)*1000:.0f} ms, 최대 {gaps.max()*1000:.0f} ms')
    print(f'범위      : x {x.min():.2f}~{x.max():.2f}  '
          f'y {y.min():.2f}~{y.max():.2f}  z {z.min():.2f}~{z.max():.2f}')
    print(f'속도      : 중앙 {np.median(speed):.2f} m/s, 최대 {speed.max():.2f} m/s')
    print(f'코스팅    : {coasting}/{len(t)} 샘플 (cov_x > 0.01)')
    if wps:
        print(f'경로점    : {len(wps)}건')
    print(f'저장      : {out}')

    if not save_only:
        plt.show()
    return out


def main():
    ap = argparse.ArgumentParser(
        description='런 CSV를 궤적 그림으로 만든다.')
    ap.add_argument('--run', help='런 폴더 경로 (기본: 가장 최근 런)')
    ap.add_argument('--results-dir', default=DEFAULT_RESULTS,
                    help=f'런들이 모인 폴더 (기본: {DEFAULT_RESULTS})')
    ap.add_argument('--params', default=None,
                    help='ROI/LiDAR를 읽을 params yaml')
    ap.add_argument('--list', action='store_true', help='런 목록만 출력')
    ap.add_argument('--save-only', action='store_true',
                    help='창을 띄우지 않고 PNG만 저장')
    # ros2 run 이 항상 붙이는 --ros-args 이후는 이 툴과 무관하므로 버린다.
    args, _ = ap.parse_known_args(
        [a for a in sys.argv[1:] if a != '--ros-args'])

    runs = find_runs(args.results_dir)
    if not runs:
        print(f'{args.results_dir} 아래에 {TRAJ} 를 가진 런이 없습니다.')
        return 1

    if args.list:
        print(f'{len(runs)}개 (최근순):')
        for r in runs:
            n = sum(1 for _ in open(os.path.join(r, TRAJ))) - 1
            print(f'  {os.path.basename(os.path.dirname(r))}/'
                  f'{os.path.basename(r)}  ({n} samples)')
        return 0

    run_dir = os.path.expanduser(args.run) if args.run else runs[0]
    if not os.path.isfile(os.path.join(run_dir, TRAJ)):
        print(f'{TRAJ} 가 없습니다: {run_dir}')
        return 1

    params_path = args.params or default_params_path()
    params = read_params(params_path) if params_path else None
    if params is None:
        print('params yaml 을 못 읽어 ROI/LiDAR 오버레이를 생략합니다.')

    return 0 if plot(run_dir, params, args.save_only) else 1


if __name__ == '__main__':
    sys.exit(main())
