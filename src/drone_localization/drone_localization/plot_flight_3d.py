#!/usr/bin/env python3
"""
전체 비행 경로 3D 시각화 (인터랙티브).

plot_flight.py는 평면/고도/3D를 한 화면에 요약해 보여주는 반면, 이 툴은
3D 뷰 하나에 집중해서 현장 전체(격자 필드, 마커, LiDAR 2대, ROI z-band)를
함께 그린다. 마우스로 돌려가며 고도 프로파일과 수평 궤적을 동시에
확인할 때 쓴다.

그리는 것:
  • 계획(ground-truth) / 실제(PX4) / LiDAR 추정 궤적
  • 30x20m 필드 외곽 + 격자선, ArUco 마커 4개, HOME 패드
  • 지상 LiDAR 2대 위치(지지대 포함) — params yaml에서 읽음
  • ROI z-band (측위가 유효한 고도 구간) 반투명 표시
  • 오프라인 분석 결과(waypoint_estimates_offline.csv)가 있으면 마커별 추정점

사용법:
  ros2 run drone_localization plot_flight_3d
  # 옵션:
  #   --output_dir DIR     로그 디렉토리 (기본 ~/drone_project/flight_logs)
  #   --params FILE        LiDAR 좌표를 읽을 yaml (기본: OS1 설정)
  #   --color_by_error     추정 궤적을 실제 대비 오차 크기로 색칠
  #   --no_field           필드/격자/마커 오버레이 끄기
  #   --z_exag N           Z축 표시 배율 (기본 4.0, 1.0=실제 비율)
  #   --save_only          창을 띄우지 않고 PNG만 저장
"""

import argparse
import csv
import os

import numpy as np

# 필드 사양 (generate_competition_map.py와 동일하게 유지할 것)
AREA_X = 30.0
AREA_Y = 20.0
N_CELLS_X = 7
N_CELLS_Y = 5

# ---------------------------------------------------------------------------
# 팔레트 — dataviz 레퍼런스 팔레트에서 역할별로 가져온 값 (hex 변형 없음).
#
# 데이터 계열(identity)에는 categorical 슬롯 1~3만 쓴다. 이 3개 조합은
# 레퍼런스에 "all-pairs 검증 통과(CVD ΔE 9.2 light)"로 문서화된 세트라
# 3D 산점처럼 모든 쌍이 동시에 보이는 형태에서도 안전하다.
# 씬 요소(필드/마커/LiDAR/HOME)는 데이터가 아니므로 계열 색을 쓰지 않고
# chrome ink + 모양으로만 구분한다 — 계열 색을 씬에 뺏기지 않게.
# ---------------------------------------------------------------------------
SURFACE = '#fcfcfb'      # chart surface
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'    # axis/labels
GRIDLINE = '#e1e0d9'     # hairline grid
BASELINE = '#c3c2b7'     # axis / field outline

SERIES_ACTUAL = '#2a78d6'    # slot 1 blue  — 실제 비행(PX4)
SERIES_ESTIMATE = '#eb6834'  # slot 2 orange — LiDAR 추정
SERIES_WAYPOINT = '#1baf7a'  # slot 3 aqua  — 경로점 재추정

# 오차(magnitude) 전용 단일색 ramp. 기본 sequential 색(blue)은 '실제 비행'
# 계열과 겹치므로, 오차가 속한 계열(추정=orange)의 단일 hue를 light→dark로
# 쓴다. 무지개 대신 한 색 — 밝을수록 오차 작음.
#
# 가장 밝은 끝은 surface(#fcfcfb)에 묻히지 않을 만큼만 밝게 잡는다. 순수
# sequential이면 최저값이 표면으로 물러나도 되지만, 여기서는 그 점들이
# '추정 궤적' 자체라서 안 보이면 곤란하다.
ERROR_RAMP = ['#f7c9ae', '#f3a97f', '#ee8753', '#eb6834', '#c94b16', '#8a300a']


def _fix_mpl_toolkits():
    """낡은 시스템 mpl_toolkits가 새 matplotlib을 가로채는 문제를 교정.

    이 PC에는 pip(~/.local)의 matplotlib 3.10과 apt(/usr/lib)의 3.5.1이
    함께 있는데, dist-packages의 matplotlib-3.5.1-nspkg.pth가 mpl_toolkits를
    네임스페이스 패키지로 선점해서 3.5.1쪽이 로드된다. 그 버전은 이미
    삭제된 matplotlib.docstring을 import하려다 깨지므로 Axes3D를 못 쓴다.
    matplotlib과 같은 디렉토리에 있는(=버전이 맞는) mpl_toolkits를
    __path__ 앞에 끼워넣어 그쪽이 먼저 잡히게 한다."""
    try:
        import matplotlib
        import mpl_toolkits
        sibling = os.path.join(
            os.path.dirname(os.path.dirname(matplotlib.__file__)), 'mpl_toolkits')
        if os.path.isdir(sibling) and sibling not in mpl_toolkits.__path__:
            mpl_toolkits.__path__.insert(0, sibling)
    except Exception:
        pass  # 교정 실패해도 아래 import에서 정상 여부가 다시 판정된다


def _load_csv(path):
    if not os.path.isfile(path):
        return None
    t, x, y, z = [], [], [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            t.append(float(row[0]))
            x.append(float(row[1]))
            y.append(float(row[2]))
            z.append(float(row[3]))
    if not t:
        return None
    return np.array(t), np.array(x), np.array(y), np.array(z)


def _load_planned():
    try:
        from drone_mission.generate_groundtruth_trajectory import (
            generate_trajectory, MARKERS, HOME,
        )
        samples = generate_trajectory()
        xs = np.array([s.x for s in samples])
        ys = np.array([s.y for s in samples])
        zs = np.array([s.z for s in samples])
        return xs, ys, zs, MARKERS, HOME
    except Exception:
        return None


def _load_waypoint_estimates(out_dir):
    """오프라인 분석 결과 우선, 없으면 실시간 트래커가 쓴 것을 사용."""
    for name in ('waypoint_estimates_offline.csv', 'waypoint_estimates.csv'):
        path = os.path.join(out_dir, name)
        if not os.path.isfile(path):
            continue
        rows = []
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    rows.append((int(r['marker_id']), int(r['visit']),
                                 float(r['x']), float(r['y']), float(r['z'])))
                except (KeyError, ValueError):
                    continue
        if rows:
            return rows, name
    return None, None


def _load_lidars(params_path):
    """params yaml에서 LiDAR 2대의 (x, y, z)를 읽는다. 실패하면 None."""
    try:
        import yaml
        with open(params_path) as f:
            data = yaml.safe_load(f)
        p = data['lidar_drone_tracker']['ros__parameters']
        return [
            (p['lidar1_x'], p['lidar1_y'], p['lidar1_z']),
            (p['lidar2_x'], p['lidar2_y'], p['lidar2_z']),
        ], (p.get('roi_z_min'), p.get('roi_z_max'))
    except Exception:
        return None, (None, None)


def _default_params_path():
    """설치된 share/params를 먼저 찾고, 없으면 소스 트리를 시도."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('drone_localization')
        cand = os.path.join(share, 'params', 'lidar_localization_os1.yaml')
        if os.path.isfile(cand):
            return cand
    except Exception:
        pass
    return os.path.expanduser(
        '~/drone_project/DronePositioningSystem/ros2_ws/src/drone_localization/'
        'params/lidar_localization_os1.yaml')


def _style_axes(ax):
    """3D 기본 크롬(회색 패널 + 굵은 격자)은 데이터보다 시선을 끈다.
    패널을 없애고 격자를 hairline으로 낮춰 데이터가 앞에 오게 한다."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(GRIDLINE)
        axis.pane.set_alpha(1.0)
        axis.line.set_color(BASELINE)
        axis._axinfo['grid'].update(color=GRIDLINE, linewidth=0.6,
                                    linestyle='-')
        axis.set_tick_params(colors=INK_MUTED, labelsize=8)
    ax.set_facecolor(SURFACE)


def _draw_field(ax, markers, home, z_floor=0.0):
    """필드 외곽 + 격자선 + 마커 + HOME을 바닥면에 그린다.

    전부 chrome ink로만 그린다 — 씬 참조물이지 데이터 계열이 아니므로
    categorical 슬롯을 쓰지 않는다. 서로는 모양(사각/삼각/원)으로 구분."""
    xs = [i * AREA_X / N_CELLS_X for i in range(N_CELLS_X + 1)]
    ys = [j * AREA_Y / N_CELLS_Y for j in range(N_CELLS_Y + 1)]

    for x in xs:
        ax.plot([x, x], [ys[0], ys[-1]], [z_floor, z_floor],
                color=GRIDLINE, lw=0.6, zorder=1)
    for y in ys:
        ax.plot([xs[0], xs[-1]], [y, y], [z_floor, z_floor],
                color=GRIDLINE, lw=0.6, zorder=1)

    ax.plot([0, AREA_X, AREA_X, 0, 0], [0, 0, AREA_Y, AREA_Y, 0],
            [z_floor] * 5, color=BASELINE, lw=1.2, zorder=2)

    if markers:
        mx = [p[0] for p in markers.values()]
        my = [p[1] for p in markers.values()]
        ax.scatter(mx, my, [z_floor] * len(mx), marker='s', s=54,
                   facecolors=INK_SECONDARY, edgecolors=SURFACE, linewidths=1.2,
                   depthshade=False, zorder=4, label='ArUco marker')
        # 직접 라벨은 마커 4개에만 (선택적 라벨링) — 방문마다 붙이면 겹친다.
        for mid, (x, y) in markers.items():
            ax.text(x, y, z_floor + 0.30, f'id{mid}', fontsize=8,
                    ha='center', color=INK_SECONDARY)

    if home is not None:
        ax.scatter([home[0]], [home[1]], [z_floor], marker='o', s=64,
                   facecolors=SURFACE, edgecolors=INK_SECONDARY, linewidths=1.4,
                   depthshade=False, zorder=4, label='Home pad')


def _draw_lidars(ax, lidars):
    for i, (lx, ly, lz) in enumerate(lidars):
        ax.plot([lx, lx], [ly, ly], [0.0, lz], color=BASELINE, lw=1.2, zorder=3)
        ax.scatter([lx], [ly], [lz], marker='^', s=70,
                   facecolors=INK_MUTED, edgecolors=SURFACE, linewidths=1.2,
                   depthshade=False, zorder=5,
                   label='Ground LiDAR' if i == 0 else None)


def _draw_roi_band(ax, z_min, z_max):
    """측위가 유효한 고도 구간. 데이터를 가리지 않게 아주 옅게."""
    xx, yy = np.meshgrid([0, AREA_X], [0, AREA_Y])
    for z in (z_min, z_max):
        ax.plot_surface(xx, yy, np.full_like(xx, z, dtype=float),
                        color=INK_MUTED, alpha=0.05, shade=False, zorder=0)


def _set_equal_aspect(ax, xlim, ylim, zlim, z_exag=1.0):
    """matplotlib 3D는 축 스케일이 기본적으로 왜곡되므로, 데이터 범위
    비율을 box_aspect로 넘겨 실제 형상이 유지되게 한다.

    다만 필드는 30x20m인데 비행고도는 2m 남짓이라 실제 비율 그대로면
    Z축이 납작해져서 고도 변화가 안 보인다. z_exag로 Z만 늘려 고도
    프로파일을 읽을 수 있게 한다 (좌표값 자체는 그대로, 표시 비율만 조정)."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    try:
        ax.set_box_aspect((xlim[1] - xlim[0],
                           ylim[1] - ylim[0],
                           max(zlim[1] - zlim[0], 1e-6) * z_exag))
    except Exception:
        pass  # 구버전 matplotlib


def _interp_actual(actual, te):
    """추정 타임스탬프에 맞춰 실제 궤적을 보간 (오차 색칠용)."""
    ta, xa, ya, za = actual
    m = (te >= ta[0]) & (te <= ta[-1])
    if m.sum() < 2:
        return None
    return (m,
            np.interp(te[m], ta, xa),
            np.interp(te[m], ta, ya),
            np.interp(te[m], ta, za))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        default=os.path.expanduser('~/drone_project/flight_logs'))
    parser.add_argument('--params', default=None)
    parser.add_argument('--color_by_error', action='store_true')
    parser.add_argument('--no_field', action='store_true')
    parser.add_argument('--save_only', action='store_true')
    parser.add_argument('--z_exag', type=float, default=4.0,
                        help='Z축 표시 배율 (1.0=실제 비율). 필드가 30x20m인데 '
                             '고도는 2m라 기본값으로 Z를 늘려 고도 변화를 보이게 함.')
    args, _ = parser.parse_known_args()

    out_dir = args.output_dir
    actual = _load_csv(os.path.join(out_dir, 'actual_path.csv'))
    estimate = _load_csv(os.path.join(out_dir, 'estimated_path.csv'))
    planned = _load_planned()
    waypoints, wp_src = _load_waypoint_estimates(out_dir)

    if actual is None and estimate is None and planned is None:
        print(f'그릴 데이터가 없습니다: {out_dir}\n'
              'flight_recorder로 비행을 기록한 뒤 다시 실행하세요.')
        return

    params_path = args.params or _default_params_path()
    lidars, (roi_z_min, roi_z_max) = _load_lidars(params_path)

    _fix_mpl_toolkits()
    import matplotlib
    if args.save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:
        print(f'3D 렌더링 불가(mpl_toolkits 문제): {e}\n'
              '두 버전이 섞여 있습니다. 다음으로 정리할 수 있습니다:\n'
              '  pip3 install --user --force-reinstall matplotlib')
        return

    # 3D 축은 subplot + tight_layout 조합에서 캔버스를 절반밖에 못 쓰므로
    # 위치를 직접 지정해 제목/범례 아래 영역을 꽉 채운다.
    fig = plt.figure(figsize=(12, 8.5), facecolor=SURFACE)
    ax = fig.add_axes((-0.02, -0.04, 0.92, 0.92), projection='3d')
    _style_axes(ax)

    if not args.no_field:
        markers = planned[3] if planned else None
        home = planned[4] if planned else None
        if roi_z_min is not None and roi_z_max is not None:
            _draw_roi_band(ax, roi_z_min, roi_z_max)
        _draw_field(ax, markers, home)
        if lidars:
            _draw_lidars(ax, lidars)

    # 계획 궤적은 '측정값'이 아니라 참조선이므로 계열 색을 쓰지 않고
    # muted ink + 파선으로 뒤로 물린다 (파선은 CVD 대비 2차 인코딩 역할도 함).
    if planned is not None:
        ax.plot(planned[0], planned[1], planned[2], '--', color=INK_MUTED,
                lw=1.1, zorder=6, label='Planned (ground-truth)')

    if actual is not None:
        ax.plot(actual[1], actual[2], actual[3], '-', color=SERIES_ACTUAL,
                lw=2.0, zorder=7, label='Actual flight (PX4)')

    if estimate is not None:
        te, xe, ye, ze = estimate
        interp = _interp_actual(actual, te) if (args.color_by_error and actual is not None) else None
        if interp is not None:
            m, xa_i, ya_i, za_i = interp
            err = np.sqrt((xe[m] - xa_i) ** 2 + (ye[m] - ya_i) ** 2
                          + (ze[m] - za_i) ** 2)
            cmap = LinearSegmentedColormap.from_list('err_orange', ERROR_RAMP)
            # 오차 분포는 소수의 큰 이상치(추적 상실 구간)가 꼬리를 길게
            # 끌어서, 최대값에 스케일을 맞추면 정상 구간이 전부 최저 색으로
            # 뭉개진다. p95에서 자르고 그 위는 같은 색으로 포화시킨다.
            vmax = float(np.percentile(err, 95))
            if vmax <= 0:
                vmax = float(err.max()) or 1.0
            sc = ax.scatter(xe[m], ye[m], ze[m], c=err, cmap=cmap, s=9,
                            vmin=0.0, vmax=vmax, linewidths=0,
                            depthshade=False, zorder=8,
                            label='LiDAR estimate (shaded by error)')
            cax = fig.add_axes((0.90, 0.16, 0.014, 0.34))
            cb = fig.colorbar(sc, cax=cax, extend='max')
            cb.set_label('3D error vs PX4  [m]', color=INK_SECONDARY, fontsize=9)
            cb.ax.tick_params(colors=INK_MUTED, labelsize=8)
            cb.outline.set_edgecolor(GRIDLINE)
            cax.text(0.5, -0.10, f'scale cut at p95\n(max {err.max():.2f} m)',
                     transform=cax.transAxes, ha='center', va='top',
                     fontsize=7.5, color=INK_MUTED)
        else:
            ax.plot(xe, ye, ze, '-', color=SERIES_ESTIMATE, lw=1.6, alpha=0.9,
                    zorder=8, label='LiDAR estimate')

    # 경로점 재추정값: 이 그림의 결론에 해당하는 마크. 궤적 위에 겹치므로
    # surface 링(2px)으로 분리하고, 3D는 zorder가 깊이 정렬에 밀려 산점에
    # 가려지므로 바닥까지 수직 stem을 내려 위치를 고정한다 (아래 마커
    # 사각형과 이어져 "어느 경로점의 추정인지"도 같이 읽힌다).
    # 값 자체는 CSV가 테이블 뷰 역할을 하므로 점마다 숫자를 붙이지 않는다.
    if waypoints:
        wx = [w[2] for w in waypoints]
        wy = [w[3] for w in waypoints]
        wz = [w[4] for w in waypoints]
        for x, y, z in zip(wx, wy, wz):
            ax.plot([x, x], [y, y], [0.0, z], color=SERIES_WAYPOINT,
                    lw=1.0, alpha=0.55, zorder=9)
        ax.scatter(wx, wy, wz, marker='o', s=120,
                   facecolors=SERIES_WAYPOINT, edgecolors=SURFACE,
                   linewidths=2.0, depthshade=False, zorder=10,
                   label=f'Waypoint estimate  (n={len(waypoints)})')

    # 축 범위: 필드를 항상 포함하고, 궤적이 벗어나면 그만큼 넓힌다.
    all_x, all_y, all_z = [0.0, AREA_X], [0.0, AREA_Y], [0.0, 3.0]
    for d in (actual, estimate):
        if d is not None:
            all_x += [float(d[1].min()), float(d[1].max())]
            all_y += [float(d[2].min()), float(d[2].max())]
            all_z += [float(d[3].min()), float(d[3].max())]
    if planned is not None:
        all_z += [float(planned[2].min()), float(planned[2].max())]
    if lidars:
        for lx, ly, lz in lidars:
            all_x.append(lx)
            all_y.append(ly)
            all_z.append(lz)

    pad = 1.5
    _set_equal_aspect(
        ax,
        (min(all_x) - pad, max(all_x) + pad),
        (min(all_y) - pad, max(all_y) + pad),
        (min(all_z) - 0.3, max(all_z) + 0.8),
        z_exag=args.z_exag,
    )

    ax.set_xlabel('world X  [m]', color=INK_SECONDARY, fontsize=9, labelpad=6)
    ax.set_ylabel('world Y  [m]', color=INK_SECONDARY, fontsize=9, labelpad=6)
    ax.set_zlabel('altitude Z  [m]', color=INK_SECONDARY, fontsize=9, labelpad=-2)
    ax.view_init(elev=24, azim=-62)

    # 제목/부제는 축 위가 아니라 figure 상단에 둬서 플롯 영역을 침범하지 않게.
    fig.text(0.02, 0.965, 'Flight trajectory — planned vs actual vs LiDAR estimate',
             fontsize=13, color=INK_PRIMARY, ha='left', va='top')
    sub = 'Gazebo world frame (ENU)'
    if abs(args.z_exag - 1.0) > 1e-6:
        sub += f'   ·   Z exaggerated x{args.z_exag:g} for readability'
    if wp_src:
        sub += f'   ·   waypoints from {wp_src}'
    fig.text(0.02, 0.925, sub, fontsize=9, color=INK_MUTED, ha='left', va='top')

    # 범례는 플롯 위에 겹치지 않게 상단 가로 배치.
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = fig.legend(handles, labels, loc='upper left',
                         bbox_to_anchor=(0.02, 0.90), ncol=3, frameon=False,
                         fontsize=8.5, handlelength=1.6, columnspacing=1.4)
        for txt in leg.get_texts():
            txt.set_color(INK_SECONDARY)

    # 계획 궤적은 패키지에서 오므로 로그 디렉토리가 없어도 여기까지 온다.
    # flight_recorder와 동일하게 필요하면 만들어 준다.
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, 'flight_plot_3d.png')
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    print(f'그림 저장: {out_png}')

    if not args.save_only:
        try:
            plt.show()
        except Exception as e:
            print(f'창을 못 열었습니다({e}). 저장된 PNG를 열어보세요: {out_png}')


if __name__ == '__main__':
    main()
