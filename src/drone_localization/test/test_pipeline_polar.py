"""파이프라인 전 구간 검증 — ROS 통신만 스텁, 나머지는 전부 실제 코드.

  합성 점구름 → [실제] 프레임 짝짓기 → [실제] ROI → [실제] sklearn DBSCAN
              → [실제] _select_drone_cluster (적응 게이트)
              → [실제] _estimate_detection_covariance → [실제] _fuse_detections
              → [실제] _apply_measurement (극좌표 EKF 또는 융합+선형)

즉 노드의 _process 를 그대로 태운다. 같은 입력으로 use_polar_ekf True/False 를
돌려 직접 비교한다.

합성 점구름은 LiDAR가 드론의 **자기 쪽 면만** 본다는 성질을 그대로 흉내낸다.
그래서 각 센서의 센트로이드는 실제 중심보다 자기 쪽으로 당겨져 찍힌다 —
극좌표 EKF가 상쇄하려는 바로 그 편향이다.

  python3 test/test_pipeline_polar.py     (또는 pytest)
"""
import math
import os
import sys
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for name in ('rclpy', 'rclpy.node', 'rclpy.qos', 'geometry_msgs',
             'geometry_msgs.msg', 'sensor_msgs', 'sensor_msgs.msg',
             'std_msgs', 'std_msgs.msg', 'visualization_msgs',
             'visualization_msgs.msg', 'sensor_msgs_py',
             'sensor_msgs_py.point_cloud2'):
    sys.modules.setdefault(name, MagicMock())


class _Node:
    def __init__(self, *a, **k):
        pass


sys.modules['rclpy.node'].Node = _Node
sys.modules['std_msgs.msg'].Header = lambda: SimpleNamespace(
    stamp=SimpleNamespace(sec=0, nanosec=0), frame_id='')

import numpy as np  # noqa: E402

from drone_localization.ekf import EKF6D  # noqa: E402
from drone_localization.lidar_drone_tracker import (  # noqa: E402
    LidarDroneTracker, SensorCfg)

BODY_RADIUS = 0.22      # 기체 반경 [m]
NEAR_FACE = 0.12        # 근접면 편향: 중심에서 센서 쪽으로 [m]
BEAM_RES = math.radians(0.35)


def build_tracker(use_polar_ekf):
    t = object.__new__(LidarDroneTracker)
    t._sensors = [
        SensorCfg('lidar1', '', 12.0, -1.0, 2.0, 0.0, 0.0, math.pi / 2, 0.05),
        SensorCfg('lidar2', '', 12.0, 16.0, 2.0, 0.0, 0.0, -math.pi / 2, 0.05),
    ]

    # ROI / 필드
    t.roi_x_min, t.roi_x_max = 0.0, 24.0
    t.roi_y_min, t.roi_y_max = 0.0, 15.0
    t.roi_z_min, t.roi_z_max = 0.8, 3.2
    t.world_frame = 'world'
    t.remove_boundary_surfaces = False
    t.boundary_surface_margin = 0.10

    # 배경·필터링: 합성 입력에는 배경이 없고, 점이 적어 깎지 않는다
    t.learn_background = False
    t.use_background = False
    t._bg_trees = [None, None]
    t._bg_points = [None, None]
    t.background_distance = 0.08
    t.voxel_leaf_size = 0.0
    t.sor_k = 0
    t.sor_std = 2.0

    # 군집화 / 후보 검증
    t.eps = 0.22
    t.min_pts = 5
    t.max_pts = 2000
    t.cluster_max_extent = 1.20
    t.cluster_min_extent = 0.02

    # 검출 공분산
    t.centroid_noise_floor = 0.025
    t.range_noise_per_meter = 0.002
    t.min_detection_std = 0.03
    t.max_detection_std = 0.50
    t.meas_noise_pos = 0.10

    # 연관 게이트
    t.max_assoc_dist = 1.75
    t.gate_sigma_k = 3.0
    t.max_assoc_dist_cap = 4.0

    # 타이밍
    t.update_rate = 10.0
    t._dt = 0.1
    t.buffer_stale_timeout = 0.30
    t.queue_depth = 8
    t.sync_tolerance = 0.010
    t.async_tolerance = 0.06
    t.sync_stats_window = 200
    t.sensor_agreement_dist = 0.35
    t._cloud_queues = [deque(maxlen=8), deque(maxlen=8)]
    t._last_processed_stamp_ns = [None, None]
    t._sync_samples_ms = deque(maxlen=200)
    t._sync_last_log = time.monotonic()
    t._last_process_monotonic = time.monotonic()

    # EKF
    t.use_polar_ekf = use_polar_ekf
    t.ekf_sigma_range = 0.10
    t.ekf_beam_res = BEAM_RES
    t.ekf_sigma_body = 0.14
    t.accel_noise_std = 2.0
    t._kf = EKF6D(accel_noise_std=2.0, default_meas_noise_std=0.10)

    # 트랙 유지 / 런 상태
    t.coast_max = 10
    t.reset_after = 50
    t._coast_count = 0
    t._run_active = False
    t.enable_waypoint_tracking = False
    t._raw_csv_w = t._traj_csv_w = t._wp_csv_w = t._ekf_csv_w = None

    # 통계 / ROS 스텁
    t._stat_period = 1e9        # 테스트 중 로그 억제
    t._stat_t0 = time.monotonic()
    t._reset_stats()
    t.get_logger = MagicMock()
    t.get_clock = MagicMock()
    t._pub_world = [MagicMock(), MagicMock()]
    t._pub_filtered = [MagicMock(), MagicMock()]
    t._pub_sensor_pose = [MagicMock(), MagicMock()]
    t._pub_markers = MagicMock()
    t._pub_pose = MagicMock()
    return t


def face_points(p, sensor_xyz, rng, n=20):
    """센서 쪽을 향한 면에서만 반사된 점들을 만든다.

    시선에 수직인 원판 위에 점을 뿌리고, 원판을 중심보다 NEAR_FACE 만큼
    센서 쪽으로 당긴다 — 실제 LiDAR가 보는 그림이다. 거리에 비례하는 빔
    양자화 잡음을 얹는다."""
    los = np.asarray(sensor_xyz, dtype=float) - p
    dist = float(np.linalg.norm(los))
    los /= dist
    # 시선에 수직인 정규직교 두 축
    tmp = np.array([0.0, 0.0, 1.0]) if abs(los[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(los, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(los, u)

    r = BODY_RADIUS * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    face = p + los * NEAR_FACE
    pts = face + np.outer(r * np.cos(th), u) + np.outer(r * np.sin(th), v)
    # 빔 양자화 (시선 수직) + 거리 잡음 (시선 방향)
    pts += np.outer(rng.normal(0, BEAM_RES * dist, n), u)
    pts += np.outer(rng.normal(0, BEAM_RES * dist, n), v)
    pts += np.outer(rng.normal(0, 0.01, n), los)
    return pts


def push_frame(t, p, rng, stamp_s):
    stamp = SimpleNamespace(sec=int(stamp_s),
                            nanosec=int(round((stamp_s % 1.0) * 1e9)))
    now = time.monotonic()
    for i, cfg in enumerate(t._sensors):
        pts = face_points(p, cfg.t_s2w, rng)
        t._cloud_queues[i].append((stamp_s, pts, now, stamp))


def fly(t, p, rng, frames=40, t0=100.0):
    """호버 중인 드론을 frames 프레임 동안 추적한다."""
    for k in range(frames):
        push_frame(t, p, rng, t0 + 0.1 * k)
        # 실시간이 아니라 프레임 간격 0.1 s 로 예측하도록 맞춘다.
        t._last_process_monotonic = time.monotonic() - 0.1
        t._process()
    return t._kf.position.copy()


# ---------------------------------------------------------------------------


def test_pipeline_detects_and_uses_both_sensors():
    t = build_tracker(use_polar_ekf=True)
    p = np.array([8.0, 6.0, 2.0])
    est = fly(t, p, np.random.default_rng(1))

    assert t._stat_frames > 0, '프레임이 하나도 처리되지 않았다'
    assert t._stat_detect / t._stat_frames > 0.9, '검출률이 너무 낮다'
    assert t._stat_ekf_frames > 0, '극좌표 EKF 경로를 한 번도 안 탔다'
    per_frame = t._stat_ekf_sensors / t._stat_ekf_frames
    print(f'  검출 {t._stat_detect}/{t._stat_frames} 프레임, '
          f'센서 {per_frame:.2f}대/프레임, 최종오차 '
          f'{np.linalg.norm(est - p) * 100:.1f} cm')
    assert per_frame > 1.9, '두 센서가 매 프레임 반영돼야 한다'
    assert t._stat_ekf_fallback == 0
    assert np.linalg.norm(est - p) < 0.15


def test_nis_is_consistent_with_sigmas():
    """NIS 평균이 3(자유도 3)에서 크게 벗어나면 σ 설정이 틀린 것이다."""
    t = build_tracker(use_polar_ekf=True)
    fly(t, np.array([8.0, 6.0, 2.0]), np.random.default_rng(2))
    nis = float(np.mean(t._stat_nis))
    print(f'  평균 NIS {nis:.2f} (이상적 3.0)')
    assert 0.5 < nis < 8.0


def test_polar_beats_fused_linear_on_waypoints():
    """경로점 위치(호버 median)를 두 경로로 각각 재고 비교한다."""
    markers = {1: (5.0, 4.0), 2: (5.0, 11.0), 3: (19.0, 4.0), 4: (12.0, 11.0)}
    errs = {True: [], False: []}

    for use_ekf in (True, False):
        for mid, (mx, my) in markers.items():
            p = np.array([mx, my, 2.0])
            for trial in range(5):
                t = build_tracker(use_polar_ekf=use_ekf)
                est = fly(t, p, np.random.default_rng(9000 + 100 * mid + trial))
                errs[use_ekf].append(np.linalg.norm(est[:2] - p[:2]))

    ekf = float(np.sqrt(np.mean(np.square(errs[True]))))
    lin = float(np.sqrt(np.mean(np.square(errs[False]))))
    print(f'  경로점 수평 RMSE — 극좌표 EKF {ekf * 100:.1f} cm, '
          f'융합+선형 {lin * 100:.1f} cm')
    assert ekf < lin, '극좌표 EKF가 근접면 편향을 더 잘 상쇄해야 한다'
    # 채점기준(<0.5 m)에는 양쪽 다 들어가야 정상이다.
    assert ekf < 0.5 and lin < 0.5


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(f'── {name}')
            try:
                fn()
                print('   PASS')
            except AssertionError as exc:
                fails += 1
                print(f'   FAIL: {exc}')
    print('PASS' if fails == 0 else f'FAIL ({fails})')
    sys.exit(1 if fails else 0)
