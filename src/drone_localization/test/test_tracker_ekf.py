"""트래커의 센서별 갱신 경로를 실제 코드로 검증한다.

ROS만 스텁이고 나머지는 전부 실제 lidar_drone_tracker 코드다:
_estimate_detection_covariance → _fuse_detections → _apply_measurement.

확인하는 것:
  1. 두 LiDAR가 정말 하나의 필터에 각각 반영되는가
  2. use_polar_ekf=false 면 예전 경로(융합 후 선형 갱신)로 돌아가는가
  3. 검출이 하나도 안 남으면 융합 경로로 안전하게 빠지는가
  4. 진단 CSV에 센서별 기여가 남는가
  5. **근접면 편향이 기하로 상쇄되는가** — 극좌표 EKF를 쓰는 실제 이유

  python3 test/test_tracker_ekf.py     (또는 pytest)
"""
import csv
import io
import math
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Node 는 상속 대상이라 진짜 클래스여야 한다 (MagicMock 이면 서브클래싱 불가)
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

import numpy as np  # noqa: E402

from drone_localization.ekf import EKF6D  # noqa: E402
from drone_localization.lidar_drone_tracker import (  # noqa: E402
    Detection, FrameInfo, LidarDroneTracker, SensorCfg)

STAMP = SimpleNamespace(sec=100, nanosec=0)
HEADER = SimpleNamespace(stamp=STAMP)


def build_tracker(use_polar_ekf=True, ekf_csv=None):
    """노드를 띄우지 않고 필요한 속성만 채운 껍데기 인스턴스."""
    t = object.__new__(LidarDroneTracker)
    t._sensors = [
        SensorCfg('lidar1', '', 12.0, -1.0, 2.0, 0.0, 0.0, math.pi / 2, 0.05),
        SensorCfg('lidar2', '', 12.0, 16.0, 2.0, 0.0, 0.0, -math.pi / 2, 0.05),
    ]
    t.use_polar_ekf = use_polar_ekf
    t.ekf_sigma_range = 0.10
    t.ekf_beam_res = math.radians(0.35)
    t.ekf_sigma_body = 0.14
    t.centroid_noise_floor = 0.025
    t.range_noise_per_meter = 0.002
    t.min_detection_std = 0.03
    t.max_detection_std = 0.50
    t.sync_tolerance = 0.010
    t.async_tolerance = 0.06
    t.accel_noise_std = 2.0
    t.sensor_agreement_dist = 0.35
    t._kf = EKF6D(accel_noise_std=2.0, default_meas_noise_std=0.10)
    t._reset_stats()
    t._ekf_csv_w = ekf_csv
    t.get_logger = MagicMock()
    return t


def make_detection(t, sensor_index, centroid, n_pts=20, rng=None, spread=0.12):
    """센트로이드 주변에 점을 뿌려 실제 공분산 추정기를 태운 Detection."""
    rng = rng or np.random.default_rng(0)
    cfg = t._sensors[sensor_index]
    cluster = np.asarray(centroid) + rng.normal(0.0, spread / 2.0, size=(n_pts, 3))
    sensor_range = float(np.linalg.norm(np.asarray(centroid) - cfg.t_s2w))
    cov = t._estimate_detection_covariance(
        cfg, cluster, np.asarray(centroid), spread, sensor_range)
    return Detection(
        sensor_index=sensor_index, sensor_name=cfg.name,
        position=np.asarray(centroid, dtype=float), covariance=cov,
        cluster_points=n_pts, spread=spread, quality=1.0,
        sensor_range=sensor_range, stamp=STAMP, candidate_count=1)


def run_frame(t, detections):
    """_process 안에서 벌어지는 일을 그대로 재현한다."""
    predicted = t._kf.position if t._kf.initialized else None
    (z, cov, _stamp, pts, cands, n_sensors, sync_dt, used) = \
        t._fuse_detections(detections, predicted)
    if z is None:
        return None
    frame = FrameInfo(n_clouds=len(detections), n_cand=cands, sel_pts=pts,
                      sync_dt=sync_dt)
    if not t._kf.initialized:
        t._kf.init(z, cov)
    else:
        t._apply_measurement(used, z, cov, HEADER, frame)
    return n_sensors


# ---------------------------------------------------------------------------


def test_both_sensors_reach_the_filter():
    buf = io.StringIO()
    t = build_tracker(ekf_csv=csv.writer(buf))
    truth = np.array([8.0, 6.0, 2.0])
    t._kf.init(truth + np.array([0.25, 0.10, 0.0]), np.eye(3) * 0.25)

    dets = [make_detection(t, 0, truth), make_detection(t, 1, truth)]
    assert run_frame(t, dets) == 2

    assert t._stat_ekf_frames == 1
    assert t._stat_ekf_sensors == 2, '두 센서가 모두 반영돼야 한다'
    assert len(t._stat_nis) == 2
    err = float(np.linalg.norm(t._kf.position - truth))
    print(f'  두 센서 갱신 후 오차 {err * 100:.1f} cm')
    assert err < 0.10

    row = buf.getvalue().strip().split('\r\n')[-1].split(',')
    assert row[1] == 'per_sensor'
    assert row[2] == '2'
    # 센서별 점 개수 열(lidar1_pts, lidar2_pts)이 둘 다 채워져야 한다
    assert int(row[7]) == 20 and int(row[11]) == 20


def test_linear_path_when_disabled():
    buf = io.StringIO()
    t = build_tracker(use_polar_ekf=False, ekf_csv=csv.writer(buf))
    truth = np.array([8.0, 6.0, 2.0])
    t._kf.init(truth + np.array([0.25, 0.0, 0.0]), np.eye(3) * 0.25)

    run_frame(t, [make_detection(t, 0, truth), make_detection(t, 1, truth)])

    assert t._stat_ekf_frames == 0
    assert t._stat_nis == []
    assert buf.getvalue().strip().split('\r\n')[-1].split(',')[1] == 'linear'


def test_fused_fallback_without_usable_detections():
    buf = io.StringIO()
    t = build_tracker(ekf_csv=csv.writer(buf))
    truth = np.array([8.0, 6.0, 2.0])
    t._kf.init(truth, np.eye(3) * 0.25)

    # used 가 비면(양쪽 다 버려진 프레임) 융합 측정으로 안전하게 갱신한다.
    t._apply_measurement([], truth + 0.05, np.eye(3) * 0.01, HEADER, FrameInfo())
    assert t._stat_ekf_fallback == 1
    assert buf.getvalue().strip().split('\r\n')[-1].split(',')[1] == 'fused'


def test_scan_offset_keeps_both_sensors():
    """PTP 없이 도는 두 OS1은 최대 반 주기(50 ms) 어긋난다.

    극좌표 EKF는 두 관측을 평균하지 않고 하나씩 넣으므로 동시일 필요가 없다.
    늦은 쪽을 필터 속도로 끌어온 뒤 둘 다 쓴다 — 예전엔 여기서 한쪽을
    통째로 버려서 두 시선이 상쇄해 주던 이점이 사라졌다."""
    t = build_tracker()
    truth = np.array([8.0, 6.0, 2.0])
    t._kf.init(truth, np.eye(3) * 0.25)
    t._kf.x[3:] = np.array([1.5, 0.0, 0.0])      # 1.5 m/s 로 이동 중

    d0 = make_detection(t, 0, truth)
    d1 = make_detection(t, 1, truth - np.array([1.5 * 0.03, 0, 0]))
    d0.stamp = SimpleNamespace(sec=100, nanosec=int(30e6))   # 30 ms 늦게 찍힘
    d1.stamp = SimpleNamespace(sec=100, nanosec=0)

    assert run_frame(t, [d0, d1]) == 2
    assert t._stat_ekf_sensors == 2, '30 ms 차이로 센서를 버리면 안 된다'


def test_stale_sensor_is_still_dropped():
    """반 주기를 훨씬 넘으면 어긋난 게 아니라 밀린 것이다 — 버린다."""
    t = build_tracker()
    truth = np.array([8.0, 6.0, 2.0])
    t._kf.init(truth, np.eye(3) * 0.25)

    d0 = make_detection(t, 0, truth)
    d1 = make_detection(t, 1, truth)
    d1.stamp = SimpleNamespace(sec=100, nanosec=int(80e6))   # 80 ms 차이

    assert run_frame(t, [d0, d1]) == 1
    assert t._stat_ekf_sensors == 1


def test_hovering_alignment_is_a_no_op():
    """호버 중(속도 0)에는 시각차 보정이 위치를 건드리지 않아야 한다.

    채점 대상인 경로점은 전부 이 조건에서 재기 때문이다."""
    t = build_tracker()
    truth = np.array([8.0, 6.0, 2.0])
    t._kf.init(truth, np.eye(3) * 0.25)     # 속도 0 으로 초기화

    d = make_detection(t, 0, truth)
    d.stamp = SimpleNamespace(sec=100, nanosec=0)
    d2 = make_detection(t, 1, truth)
    before = d.position.copy()
    aligned = t._time_align_detections([d, d2])
    moved = max(float(np.linalg.norm(a.position - before)) for a in aligned)
    assert moved < 1e-6


def test_position_disagreement_returns_one_detection():
    """두 센서가 공간적으로 어긋나면(상대 캘리브 오류) 한쪽만 쓴다."""
    t = build_tracker()
    truth = np.array([8.0, 6.0, 2.0])
    t._kf.init(truth, np.eye(3) * 0.25)

    d0 = make_detection(t, 0, truth)
    d1 = make_detection(t, 1, truth + np.array([1.0, 0.0, 0.0]))   # 1 m 어긋남

    assert run_frame(t, [d0, d1]) == 1
    assert t._stat_ekf_sensors == 1


def test_near_face_bias_cancels_geometrically():
    """극좌표 EKF를 쓰는 이유 그 자체.

    LiDAR는 드론의 자기 쪽 면만 맞히므로 센트로이드가 실제 중심보다 기체
    두께의 절반쯤 **자기 쪽으로** 당겨져 찍힌다. 시선 방향으로 길쭉한 R을
    주면 필터가 그 방향 정보를 덜 믿으므로 편향이 상쇄된다. 등방 R로 두
    관측을 평균하는 기존 경로와 같은 입력으로 비교한다."""
    delta = 0.12          # 근접면 편향 [m]
    trials = 60
    errs = {True: [], False: []}

    for use_ekf in (True, False):
        for trial in range(trials):
            rng = np.random.default_rng(4000 + trial)
            t = build_tracker(use_polar_ekf=use_ekf)
            # 필드 구석 — 두 시선이 180°가 아니라 크게 어긋나는 자리
            truth = np.array([4.0, 3.0, 2.0])
            t._kf.init(truth + rng.normal(0, 0.05, 3), np.eye(3) * 0.25)
            for _ in range(25):
                dets = []
                for i, cfg in enumerate(t._sensors):
                    los = cfg.t_s2w - truth
                    los /= np.linalg.norm(los)
                    # 센서 쪽으로 delta 만큼 당겨진 센트로이드 + 관측잡음
                    c = truth + los * delta + rng.normal(0, 0.02, 3)
                    dets.append(make_detection(t, i, c, n_pts=20, rng=rng))
                t._kf.predict(0.1)
                run_frame(t, dets)
            errs[use_ekf].append(np.linalg.norm(t._kf.position - truth))

    ekf_rmse = float(np.sqrt(np.mean(np.square(errs[True]))))
    lin_rmse = float(np.sqrt(np.mean(np.square(errs[False]))))
    print(f'  근접면 편향 {delta * 100:.0f} cm  →  '
          f'극좌표 EKF RMSE {ekf_rmse * 100:.1f} cm, '
          f'융합+선형 RMSE {lin_rmse * 100:.1f} cm')
    assert ekf_rmse < lin_rmse, '극좌표 EKF가 편향을 더 잘 상쇄해야 한다'


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
