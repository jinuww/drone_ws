#!/usr/bin/env python3
"""
관측은 센서별로, 칼만필터는 하나 — 순차 갱신 (B).

이 파일 하나로 돈다. A(lidar_drone_tracker.py)도 C(lidar_drone_tracker_dual.py)도
import하지 않는다.

세 방식의 가운데다.

    A  점을 이어붙여 중심 하나 → 필터 하나        (합치기가 필터 앞)
    B  센서별 중심 두 개 → 같은 필터에 차례로     (합치기가 필터 안)
    C  센서별 중심 → 필터 두 개 → 신뢰도로 합침   (합치기가 필터 뒤)

A에서 가져오는 생각: 필터가 하나라 정보가 쪼개지지 않는다. 상태가 하나뿐이니
교차공분산도, 겹침 보정도 필요 없다.

C에서 가져오는 생각: 점을 합치지 않으므로 어느 관측이 어느 센서 것인지 알고,
거리에 따라 R을 따로 준다. 동시성 검사가 없어 관측을 버리지 않는다.

    R(d) = (σ_ref · d / d_ref)²

== 왜 순차 갱신이 최적인가 ==

선형·가우시안 칼만필터에서 **관측 두 개를 차례로 넣는 것은 한꺼번에 쌓아
넣는 것과 정확히 같다.** 한 프레임 예시(1차원, 예측 9.90±0.20):

    ① 남쪽 9.94 (σ0.110) → 이득 0.768 → 9.9307 ±0.096
    ② 북쪽 10.09 (σ0.160) → 이득 0.266 → 9.9731 ±0.083
    한꺼번에 넣으면        →              9.9731 ±0.083   (동일)

첫 관측을 반영해 확신이 커진 뒤 두 번째를 받으므로, 두 번째가 더 신중하게
들어간다. 그래서 B는 Shin(2005)이 SFF로 근사하려던 **중앙집중 KF** 그
자체다. C는 그 근사이고, 논문은 둘의 차이를 negligible이라고 본다.

== A와 갈리는 지점 ==

A는 점을 합치는 순간 출처를 잃어 R을 고정할 수밖에 없다. 20점짜리 가까운
관측과 2점짜리 먼 관측이 한 median에 섞여 같은 취급을 받는다. B는 둘을
따로 넣으면서 각자의 거리로 이득을 정한다.

A와 달리 voxel 다운샘플링과 통계적 이상치 제거(SOR)를 쓰지 않는다. 센서당
점이 중앙값 6개뿐이라 깎을 여유가 없기 때문이다. 시뮬에는 잡음이 없어
문제가 없었지만, 실기로 옮길 때는 여기를 먼저 의심해야 한다.

== 왜 파일마다 통째로 갖고 있나 ==

예전에는 A가 '변형 A'이면서 동시에 '공통 토대'를 겸했고, B와 C가 A를,
다시 B가 C를 상속했다. 그 구조에서는 A의 파이프라인을 손보면 B와 C가
조용히 따라 바뀌었고, 어느 변화가 어느 방식의 것인지 구분되지 않았다.
**비교 실험을 하는 코드에서 이건 치명적이다** — 무엇을 재고 있는지 모르게
된다.

그래서 KF, 센서 변환, 파라미터, ROI, 경로점 감지까지 전부 이 파일 안에
둔다. A·B·C가 서로를 모르므로, 한 파일을 고쳐도 다른 둘은 그대로다.
중복은 그 대가이고, 세 방식이 앞으로 서로 다른 방향으로 갈 것이므로
치를 만한 값이다.

== 예전 시도와의 차이 ==

이 구조는 lidar_drone_tracker_async.py로 한 번 만들었다가 접었다. 접은
이유는 성능이 아니라 "갱신한 센서 쪽으로 추정치가 끌린다"는 현상이었다
(Y축 진폭 16mm — 각 라이다가 드론의 자기 쪽 면만 보기 때문). 그런데
그것을 없애려고 만든 C가 A를 넘지 못했고, 당시 B의 측정은 코스팅 버그와
스캔 40% 유실이 섞인 값이라 근거가 약하다. 그래서 깨끗한 조건에서 다시
잰다.

사용:
  ros2 launch drone_localization lidar_localization.launch.py \\
      tracker:=lidar_drone_tracker_seq

  # 관측별 상세 로그
      seq_debug_csv:=/tmp/seq.csv

Dependencies (install once):
    pip3 install open3d scipy
"""

import csv
import math
import os
import re
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy, QoSProfile, ReliabilityPolicy
)
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    raise ImportError('scipy가 설치되지 않았습니다. pip3 install scipy 실행하세요.')

try:
    import open3d as o3d
except ImportError:
    raise ImportError('open3d가 설치되지 않았습니다. pip3 install open3d 실행하세요.')

try:
    from sensor_msgs_py import point_cloud2 as pc2_util
except ImportError:
    from sensor_msgs import point_cloud2 as pc2_util


# 관측 한 줄이 한 행. C의 헤더에서 both_live/corr를 뺐다 — 트랙이 하나뿐이라
# "둘 다 살아있나"도 "두 트랙의 상관"도 정의되지 않는다.
DEBUG_HEADER = [
    't', 'sensor', 'n_pts', 'dist', 'dt', 'sigma', 'zx', 'zy', 'zz',
    'innov', 'mahal', 'accepted', 'est_x', 'est_y', 'est_z', 'est_sigma']


# ---------------------------------------------------------------------------
# KF: 6-state (x, y, z, vx, vy, vz), constant-velocity model (standard KF —
# F and H are both linear/constant, so there is nothing "extended" here)
# ---------------------------------------------------------------------------
class KF6D:
    def __init__(self, proc_noise_pos: float, proc_noise_vel: float, meas_noise_pos: float):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 9.0  # high initial uncertainty = 3m std

        # Process noise: position changes by ≤ proc_noise_pos per frame,
        # velocity changes by ≤ proc_noise_vel per frame at 10Hz
        self.Q = np.diag([
            proc_noise_pos**2, proc_noise_pos**2, proc_noise_pos**2,
            proc_noise_vel**2, proc_noise_vel**2, proc_noise_vel**2,
        ])

        # Measurement noise: raw LiDAR range noise is 0.8cm, but centroid
        # estimation from a sparse side-view cluster of a 50cm drone adds
        # 5–15cm uncertainty (fewer points → larger centroid variance).
        # Using 15cm std is a conservative but realistic choice.
        self.R = np.eye(3) * meas_noise_pos**2

        # Observation matrix: only position is measured
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.initialized = False

    def init(self, pos: np.ndarray):
        self.x[:3] = pos
        self.x[3:] = 0.0
        self.P = np.eye(6) * 0.25  # 0.5m initial uncertainty after first detection
        self.initialized = True

    def predict(self, dt: float):
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray):
        y = z - self.H @ self.x                         # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(3))  # Kalman gain
        self.x = self.x + K @ y
        # Joseph form: guarantees P stays symmetric/positive-definite even
        # under floating-point roundoff (the (I-KH)@P form can lose that).
        IKH = np.eye(6) - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()

    @property
    def cov_pos(self) -> np.ndarray:
        return self.P[:3, :3].copy()


class SensorCfg:
    """One ground-fixed LiDAR: its topic and sensor→world transform."""

    def __init__(self, name, topic, x, y, z, roll, pitch, yaw):
        self.name = name
        self.topic = topic
        self.x, self.y, self.z = x, y, z
        self.roll, self.pitch, self.yaw = roll, pitch, yaw
        self.R_s2w = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        self.t_s2w = np.array([x, y, z])

    def to_world(self, pts_sensor: np.ndarray) -> np.ndarray:
        return (self.R_s2w @ pts_sensor.T).T + self.t_s2w


class SeqTracker(Node):
    """센서별 관측 → 단일 칼만필터에 순차 갱신. 단독으로 돈다.

    두 콜백이 같은 상태를 건드리지만 ROS2 기본 실행기가 단일 스레드라 콜백은
    직렬화된다. 멀티스레드 실행기로 바꾸려면 락이 필요하다.
    """

    def __init__(self):
        super().__init__('lidar_drone_tracker')

        self._declare_parameters()
        self._read_parameters()
        self._build_sensors()
        self._declare_seq_parameters()

        self._kf = KF6D(
            self.proc_noise_pos, self.proc_noise_vel, self.meas_noise_pos)
        # A는 seed_at_home으로 필터를 패드에 미리 심지만, B는 심지 않는다.
        # 첫 관측이 곧 초기화다 — 관측 하나로 상태가 정해지므로 사전 시드가
        # 필요 없고, 시드해두면 드론이 패드에서 먼 곳에 처음 나타났을 때
        # 게이트가 그것을 거부해 리셋까지 5초를 버린다.
        self._kf.initialized = False
        self._coast_count = 0

        # 센서 원점(월드). 관측 거리를 재는 데만 쓴다.
        self._sensor_xyz = [np.array([c.x, c.y, c.z]) for c in self._sensors]

        # 마지막으로 필터를 전진시킨 관측 시각. 벽시계가 아니라 스탬프 기준이다.
        self._last_t = None
        # 마지막으로 KF가 실제 갱신된 벽시계 시각. 코스팅 판정의 기준.
        self._last_update_wall = time.monotonic()

        # Waypoint (마커) 호버 감지 상태
        self._wp_active_id = None
        self._wp_samples = []
        self._wp_break_count = 0
        self._wp_visit_counts = {}
        self._wp_markers = MarkerArray()

        os.makedirs(self.waypoint_log_dir, exist_ok=True)
        self._wp_csv_path = os.path.join(
            self.waypoint_log_dir, 'waypoint_estimates.csv')
        self._wp_csv_f = open(self._wp_csv_path, 'w', newline='')
        self._wp_csv_w = csv.writer(self._wp_csv_f)
        self._wp_csv_w.writerow(
            ['marker_id', 'visit', 'x', 'y', 'z', 'n_samples', 'duration_s'])

        # 2초마다 찍는 진단 카운터
        self._stat_period = 2.0
        self._a_scans = [0] * len(self._sensors)     # 들어온 스캔
        self._a_clusters = [0] * len(self._sensors)  # 클러스터가 선 스캔
        self._a_updates = [0] * len(self._sensors)   # KF를 갱신한 관측
        self._a_gated = [0] * len(self._sensors)     # 게이트가 거부한 관측
        self._a_pts = [[] for _ in self._sensors]    # 선택된 클러스터 점 수
        self._a_dist = [[] for _ in self._sensors]   # 관측 거리
        self._a_t0 = time.monotonic()

        # depth=1이면 콜백이 도는 동안 도착한 스캔이 큐에서 밀려난다. 라이다
        # 두 대가 거의 동시에 발행하므로 한쪽을 처리하는 사이 다른 쪽이 버려져,
        # 실측에서 10Hz 발행 중 6~7Hz만 받았다. 큐를 늘리면 처리가 밀려도
        # 받아둔다. 늦게 처리해도 관측 시각은 msg.header.stamp로 읽으므로
        # 정확도에는 영향이 없다 — 버리는 것보다 늦게라도 쓰는 편이 낫다.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._subs = [
            self.create_subscription(
                PointCloud2, self._sensors[0].topic, self._cb_lidar1, sensor_qos),
            self.create_subscription(
                PointCloud2, self._sensors[1].topic, self._cb_lidar2, sensor_qos),
        ]

        self._pub_pose = self.create_publisher(
            PoseWithCovarianceStamped, '/drone/estimated_pose', reliable_qos)
        self._pub_filtered = self.create_publisher(
            PointCloud2, '/lidar/filtered_points', sensor_qos)
        self._pub_markers = self.create_publisher(
            MarkerArray, '/lidar/cluster_markers', reliable_qos)
        self._pub_waypoints = self.create_publisher(
            MarkerArray, '/drone/waypoint_estimates', reliable_qos)

        # 융합이 아니라 관측 공백 감시만 하는 타이머.
        self._proc_timer = self.create_timer(self._dt, self._process)

        self._dbg_f = self._dbg_w = None
        self._dbg_path = self._dbg_path_param
        if self._dbg_path:
            self._open_debug()

        names = ', '.join(
            f'{c.name}({c.topic} @ {c.x:.1f},{c.y:.1f},{c.z:.1f})'
            for c in self._sensors)
        self.get_logger().info(
            f'순차 갱신 (B) · 칼만필터 1개 · {names} · '
            f'R(d)=({self.mn_ref}·d/{self.mn_ref_dist})² 하한 {self.mn_floor}m · '
            f'게이트 χ²<{self.gate_chi2}')

    def _declare_parameters(self):
        # LiDAR poses — OS1-128, ns(남북) layout. 기본값을 실제 배치와 같게
        # 두어, params_file 없이 노드만 띄워도 좌표가 맞도록 한다. 값 자체는
        # gazebo/worlds/generate_competition_map.py 의 LIDARS와 동일해야 한다.
        #
        # --- LiDAR 1: 남쪽, 필드 X중앙, 북향(필드 중심) ---
        self.declare_parameter('lidar1_topic', '/lidar1/points')
        self.declare_parameter('lidar1_x',     15.0)
        self.declare_parameter('lidar1_y',     -1.5)
        self.declare_parameter('lidar1_z',      2.0)
        self.declare_parameter('lidar1_roll',   0.0)
        self.declare_parameter('lidar1_pitch',  0.0)
        self.declare_parameter('lidar1_yaw',    math.pi / 2)

        # --- LiDAR 2: 북쪽, 필드 X중앙, 남향(필드 중심) ---
        self.declare_parameter('lidar2_topic', '/lidar2/points')
        self.declare_parameter('lidar2_x',     15.0)
        self.declare_parameter('lidar2_y',     21.5)
        self.declare_parameter('lidar2_z',      2.0)
        self.declare_parameter('lidar2_roll',   0.0)
        self.declare_parameter('lidar2_pitch',  0.0)
        self.declare_parameter('lidar2_yaw',   -math.pi / 2)

        # Max age (s) of a sensor's buffered cloud before it's ignored in
        # fusion (2× the frame period tolerates one dropped frame).
        self.declare_parameter('buffer_stale_timeout', 0.25)

        self.declare_parameter('voxel_leaf_size', 0.05)

        # ROI in world frame. X/Y: 2m margin around the 30×20 field.
        # Z: hover-altitude band only (1.0–3.0m) — this is the ONLY height
        # filter in the pipeline now. Ground-state (pre-liftoff/post-landing)
        # tracking is out of scope, so points near the floor never need to be
        # separated from the drone; the ROI simply never lets them through.
        self.declare_parameter('roi_x_min', -2.0)
        self.declare_parameter('roi_x_max', 32.0)
        self.declare_parameter('roi_y_min', -2.0)
        self.declare_parameter('roi_y_max', 22.0)
        self.declare_parameter('roi_z_min',  1.0)
        self.declare_parameter('roi_z_max',  3.0)

        # 각 LiDAR 자신의 마스트를 ROI에서 파낸다. 센서 모델은 지지대
        # (반지름 0.04m, z 0~2.0m)와 하우징(반지름 0.09m, z 1.92~2.08m)으로
        # 되어 있고, 하우징 높이가 목표 호버고도(2.0m)와 **같다**. 그래서
        # 반대편 센서가 이걸 그대로 본다.
        #
        # 실제로 이것 때문에 추적이 통째로 실패했다: 24x15 맵으로 줄이며
        # 센서 간격이 23m -> 18m로 가까워지고 ROI 하단이 1.0 -> 0.3m로
        # 내려가면서, 마스트에 찍히는 점(약 15~19개)이 드론(약 7개)보다
        # 많아졌다. 콜드스타트가 "가장 큰 클러스터"를 고르므로 마스트를
        # 잡았고, 정지 물체라 매 프레임 자기와 일치해 영원히 놓지 않았다
        # (run065/066: 추정치가 (12.0,16.4,2.0) = 북쪽 센서 위치에 고정).
        #
        # 수평거리로만 자른다 — 마스트는 z 전체에 걸쳐 있어서 z로는 못
        # 가른다. 센서는 경기장 밖에 있으므로 드론이 이 원 안에 들어올
        # 일은 없다. 0 이하면 이 기능을 끈다.
        self.declare_parameter('sensor_exclude_radius', 0.5)

        # Statistical outlier removal (rain / scatter noise)
        self.declare_parameter('sor_mean_k',     20)
        self.declare_parameter('sor_std_ratio',   2.0)

        # DBSCAN clustering (PCL EuclideanClusterExtraction equivalent)
        # eps=25cm: connects points within the same drone body (50cm–1m)
        # min_pts=3: side-view LiDAR gives few hits on a small drone at distance
        self.declare_parameter('cluster_tolerance',  0.40)
        self.declare_parameter('cluster_min_points',  2)
        self.declare_parameter('cluster_max_points',  400)

        # Home-seed: seeds the KF near where the drone should first appear so
        # early-frame candidate selection (nearest-to-track) has a prior
        # instead of falling back to cold-start (largest cluster). The ROI
        # only sees the hover band, so home_z must be a hover altitude
        # (~2.0m) — NOT the pad height, which now sits outside the ROI
        # entirely and would just fail the association gate on first detection.
        self.declare_parameter('seed_at_home', True)
        self.declare_parameter('home_x', 0.0)
        self.declare_parameter('home_y', 0.0)
        self.declare_parameter('home_z', 2.0)

        # KF noise
        self.declare_parameter('proc_noise_pos',  0.05)
        self.declare_parameter('proc_noise_vel',  0.30)
        self.declare_parameter('meas_noise_pos',  0.15)

        # Gating: reject a candidate farther than this from the KF prediction.
        # This is the BASE (well-tracked) radius in metres; the effective gate
        # widens with the filter's own uncertainty — see _select_drone_cluster.
        self.declare_parameter('max_assoc_dist', 1.5)
        # How many sigma of position uncertainty to add to the base radius.
        self.declare_parameter('gate_sigma_k', 3.0)
        # Hard ceiling on the widened gate, so a long dropout can never open it
        # far enough to latch onto unrelated clutter.
        self.declare_parameter('max_assoc_dist_cap', 4.0)

        # Coasting: max frames to continue pure-predict when no detection
        self.declare_parameter('coast_max_frames', 15)
        # Prolonged loss: after this many coast frames, drop the KF entirely
        # (cold-start) so the next detection is free to re-lock on the
        # largest cluster instead of being gated against a stale prediction.
        self.declare_parameter('reset_after_frames', 50)  # 5초 @ 10Hz

        self.declare_parameter('lidar_update_rate', 10.0)

        # ---------------------------------------------------------------
        # Waypoint(마커) 호버 감지 + 위치 재추정
        #
        # 대회 규정에 마커 위 호버 시간이 정해져 있지 않고 팀마다 체류
        # 패턴이 다르므로, 고정 시간이 아니라 "느림 + 근접" 조건으로
        # 판정한다. 채점기준 "경로점 위치추정" 하 등급이 <4m이므로,
        # 게이트 반경은 이보다 여유있게 잡아 항법이 부정확한 팀도
        # 표본을 놓치지 않게 한다 (좁게 잡으면 그런 팀은 표본이 아예
        # 안 잡혀서 채점 자체가 불가능해짐).
        # 선택된 드론 클러스터의 원본 점들을 파일로 남긴다(비었으면 끔).
        # 클러스터가 실제로 어떤 모양·분포로 잡히는지 사후에 확인하는 용도.
        self.declare_parameter('cluster_dump_path', '')

        self.declare_parameter('waypoint_gate_radius', 4.5)       # m
        self.declare_parameter('waypoint_speed_threshold', 0.25)  # m/s
        # 최소 연속 프레임(4프레임 @ 10Hz ≈ 0.4초). 이보다 짧으면 노이즈로
        # 버림 — 아주 잠깐 스치듯 지나가는 팀도 놓치지 않게 여유있게(짧게) 잡음.
        self.declare_parameter('waypoint_min_frames', 4)
        # 짧은 이탈 유예 프레임(5프레임 @ 10Hz = 0.5초). 속도추정이 정지
        # 중에도 미세하게 흔들려 문턱값을 순간적으로 넘나드는 것 때문에
        # 하나의 호버가 여러 조각으로 쪼개지는 걸 방지 — 이 프레임 수
        # 이내로 짧게 이탈하면 무시하고 같은 방문으로 이어붙인다.
        self.declare_parameter('waypoint_grace_frames', 5)
        self.declare_parameter(
            'waypoint_log_dir', os.path.expanduser('~/drone_project/flight_logs'))

        # 마커 좌표. generate_competition_map.py의 marker_coords와 항상
        # 동일하게 유지할 것 (LIDARS 좌표와 같은 방식으로 수동 동기화).
        self.declare_parameter('marker1_x', 4.2857)
        self.declare_parameter('marker1_y', 4.0)
        self.declare_parameter('marker2_x', 4.2857)
        self.declare_parameter('marker2_y', 16.0)
        self.declare_parameter('marker3_x', 25.7143)
        self.declare_parameter('marker3_y', 4.0)
        self.declare_parameter('marker4_x', 12.8571)
        self.declare_parameter('marker4_y', 12.0)

    def _read_parameters(self):
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.buffer_stale_timeout = gp('buffer_stale_timeout')
        self.voxel_leaf_size = gp('voxel_leaf_size')
        self.roi_x_min = gp('roi_x_min')
        self.roi_x_max = gp('roi_x_max')
        self.roi_y_min = gp('roi_y_min')
        self.roi_y_max = gp('roi_y_max')
        self.roi_z_min = gp('roi_z_min')
        self.roi_z_max = gp('roi_z_max')
        self.sensor_exclude_radius = gp('sensor_exclude_radius')
        self.sor_k        = gp('sor_mean_k')
        self.sor_std      = gp('sor_std_ratio')
        self.eps          = gp('cluster_tolerance')
        self.min_pts      = gp('cluster_min_points')
        self.max_pts      = gp('cluster_max_points')
        self.seed_at_home = gp('seed_at_home')
        self.home_xyz     = np.array(
            [gp('home_x'), gp('home_y'), gp('home_z')])
        self.max_assoc_dist = gp('max_assoc_dist')
        self.gate_sigma_k = gp('gate_sigma_k')
        self.max_assoc_dist_cap = gp('max_assoc_dist_cap')
        self.proc_noise_pos = gp('proc_noise_pos')
        self.proc_noise_vel = gp('proc_noise_vel')
        self.meas_noise_pos = gp('meas_noise_pos')
        self.cluster_dump_path = gp('cluster_dump_path')
        self.waypoint_gate_radius = gp('waypoint_gate_radius')
        self.waypoint_speed_threshold = gp('waypoint_speed_threshold')
        self.waypoint_min_frames = gp('waypoint_min_frames')
        self.waypoint_grace_frames = gp('waypoint_grace_frames')
        self.waypoint_log_dir = gp('waypoint_log_dir')
        self._marker_xy = {
            1: (gp('marker1_x'), gp('marker1_y')),
            2: (gp('marker2_x'), gp('marker2_y')),
            3: (gp('marker3_x'), gp('marker3_y')),
            4: (gp('marker4_x'), gp('marker4_y')),
        }
        self.coast_max    = gp('coast_max_frames')
        self.reset_after  = gp('reset_after_frames')
        self.update_rate  = gp('lidar_update_rate')
        self._dt = 1.0 / self.update_rate

    def _build_sensors(self):
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        specs = [
            ('lidar1', gp('lidar1_topic'), gp('lidar1_x'), gp('lidar1_y'),
             gp('lidar1_z'), gp('lidar1_roll'), gp('lidar1_pitch'),
             gp('lidar1_yaw')),
            ('lidar2', gp('lidar2_topic'), gp('lidar2_x'), gp('lidar2_y'),
             gp('lidar2_z'), gp('lidar2_roll'), gp('lidar2_pitch'),
             gp('lidar2_yaw')),
        ]

        self._sensors = []
        for name, topic, x, y, z, roll, pitch, yaw in specs:
            self._sensors.append(SensorCfg(name, topic, x, y, z, roll, pitch, yaw))

    # ------------------------------------------------------------------
    # Per-sensor callback: store the latest cloud in WORLD frame
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _declare_seq_parameters(self):
        """B 전용 파라미터. A의 파라미터 집합에 얹는다.

        A가 선언하지만 B가 쓰지 않는 것들이 있다 — voxel_leaf_size, sor_*,
        seed_at_home/home_*, max_assoc_dist/gate_sigma_k(B는 χ² 게이트를
        쓴다), cluster_dump_path. yaml에 남아 있어도 무시된다."""
        self.declare_parameter('meas_noise_ref', 0.12)       # d_ref에서의 표준편차(m)
        self.declare_parameter('meas_noise_ref_dist', 12.0)  # 기준 거리(m)
        self.declare_parameter('meas_noise_floor', 0.03)     # 근거리 하한
        # 마할라노비스 게이트 (자유도 3 카이제곱: 95%=7.81, 99%=11.34)
        self.declare_parameter('gate_chi2', 11.34)
        self.declare_parameter('seq_debug_csv', '')
        # 예전 이름. launch가 아직 이걸 넘기므로 받아만 준다 — 조용히 무시하면
        # "켰는데 파일이 안 생긴다"는 종류의 실패가 된다.
        self.declare_parameter('dual_debug_csv', '')

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.mn_ref = gp('meas_noise_ref')
        self.mn_ref_dist = gp('meas_noise_ref_dist')
        self.mn_floor = gp('meas_noise_floor')
        self.gate_chi2 = gp('gate_chi2')
        self._dbg_path_param = gp('seq_debug_csv') or gp('dual_debug_csv')

    def _cb_lidar1(self, msg: PointCloud2):
        self._handle_scan(0, msg)

    def _cb_lidar2(self, msg: PointCloud2):
        self._handle_scan(1, msg)

    def _handle_scan(self, idx: int, msg: PointCloud2):
        self._a_scans[idx] += 1
        pts = self._pc2_to_numpy(msg)
        if pts is None or len(pts) == 0:
            return

        # ROI는 월드 기준 개념(호버 고도 띠, 마스트 배제)이라 센서 좌표에서는
        # 걸 수 없다. A의 _roi_mask를 그대로 쓴다.
        pts_w = self._sensors[idx].to_world(pts)
        pts_w = pts_w[self._roi_mask(pts_w)]
        if len(pts_w) < self.min_pts:
            return

        centroid, n_sel = self._cluster_one(pts_w)
        if centroid is None:
            return
        self._a_clusters[idx] += 1
        self._a_pts[idx].append(n_sel)

        stamp = msg.header.stamp
        self._ingest(idx, centroid, n_sel,
                     stamp.sec + stamp.nanosec * 1e-9, stamp)

    def _cluster_one(self, pts_w: np.ndarray):
        """한 센서의 점만으로 군집화 → 드론 클러스터의 median.

        후보는 현재 추정 위치와 가까운 것으로 고른다. 아직 트랙이 없으면
        가장 큰 클러스터를 고른다(콜드스타트).

        중심을 평균이 아니라 median으로 잡는 이유는 A와 같다 — 평균은 군집에
        섞인 이상점 하나에 끌려가지만 median은 거의 움직이지 않는다."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_w.astype(np.float64))
        labels = np.array(pcd.cluster_dbscan(
            eps=self.eps, min_points=self.min_pts, print_progress=False))

        best = None
        for lab in np.unique(labels):
            if lab < 0:
                continue
            cluster = pts_w[labels == lab]
            n = len(cluster)
            if n < self.min_pts or n > self.max_pts:
                continue
            c = np.median(cluster, axis=0)
            if self._kf.initialized:
                score = float(np.linalg.norm(c - self._kf.position))
                if best is None or score < best[0]:
                    best = (score, c, n)
            elif best is None or n > best[0]:
                best = (n, c, n)
        return (None, 0) if best is None else (best[1], best[2])

    def _meas_sigma(self, dist: float) -> float:
        """거리 기반 측정 표준편차. 하한을 둬 근거리에서 R이 0에 붙는 걸 막는다.

        근거는 기하다 — 점 개수 n ∝ 1/d², 중심의 표준편차 ≈ 기체크기/√n ∝ d."""
        return max(self.mn_ref * dist / self.mn_ref_dist, self.mn_floor)

    # ------------------------------------------------------------------
    # 필터
    # ------------------------------------------------------------------
    def _advance(self, dt: float):
        if dt > 0.0 and self._kf.initialized:
            self._kf.predict(dt)

    def _ingest(self, idx, z, n_pts, t_obs, stamp):
        kf = self._kf
        dist = float(np.linalg.norm(z - self._sensor_xyz[idx]))
        sigma = self._meas_sigma(dist)

        # 관측 시각까지만 예측한다. 두 센서가 어긋나 들어오면 그 차이만큼만
        # 굴러가므로, 스탬프가 벌어져도 관측을 버릴 이유가 없다.
        if self._last_t is None:
            dt = 0.0
        else:
            dt = t_obs - self._last_t
            if dt < 0.0 or dt > 1.0:
                dt = self._dt          # 시각이 튀면 기본 주기로 대체
        self._advance(dt)
        self._last_t = t_obs

        if not kf.initialized:
            kf.init(z)
            self._last_update_wall = time.monotonic()
            self._a_updates[idx] += 1
            self._a_dist[idx].append(dist)
            self._finish(idx, z, n_pts, dist, dt, sigma, 0.0, 0.0, True, stamp)
            return

        y = z - kf.H @ kf.x
        R = np.eye(3) * sigma ** 2
        S = kf.H @ kf.P @ kf.H.T + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        mahal = float(y @ Sinv @ y)

        if mahal > self.gate_chi2:
            self._a_gated[idx] += 1
            self._coast_count += 1
            if self._coast_count > self.reset_after:
                self._reset_all('게이트 연속 기각')
            self._finish(idx, z, n_pts, dist, dt, sigma,
                         float(np.linalg.norm(y)), mahal, False, stamp)
            return

        K = kf.P @ kf.H.T @ Sinv
        kf.x = kf.x + K @ y
        # Joseph 형식: 부동소수 반올림에도 P가 대칭·양정부호로 남는다.
        IKH = np.eye(6) - K @ kf.H
        kf.P = IKH @ kf.P @ IKH.T + K @ R @ K.T

        self._coast_count = 0
        self._last_update_wall = time.monotonic()
        self._a_updates[idx] += 1
        self._a_dist[idx].append(dist)
        self._finish(idx, z, n_pts, dist, dt, sigma,
                     float(np.linalg.norm(y)), mahal, True, stamp)

    def _finish(self, idx, z, n_pts, dist, dt, sigma, innov, mahal,
                accepted, stamp):
        """C와 달리 짝을 기다리지 않는다.

        C가 기다리는 이유는 '한쪽만 갱신된 중간 상태'가 나가는 걸 막기
        위해서인데, B는 상태가 하나라 그런 중간 상태가 없다. 관측을 반영한
        직후가 언제나 그 시점의 최선이다."""
        if not self._kf.initialized:
            return
        if accepted:
            self._update_waypoint_tracking(z)
        self._log_row(idx, z, n_pts, dist, dt, sigma, innov, mahal,
                      accepted, stamp)
        self._emit(stamp)

    def _emit(self, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = 'map'
        self._publish_pose(header)

    def _reset_all(self, why: str):
        self.get_logger().warn(f'Tracking LOST ({why}) → 리셋')
        self._kf.initialized = False
        self._last_t = None
        self._coast_count = 0
        self._finalize_waypoint_episode()

    # ------------------------------------------------------------------
    # 타이머 — 융합이 아니라 관측 공백 감시만
    # ------------------------------------------------------------------
    def _process(self):
        """A의 처리 타이머 자리를 대신한다. 여기서는 아무것도 융합하지 않는다.

        기준은 '마지막 스캔'이 아니라 '마지막 KF 갱신'이어야 한다. 스캔은
        오는데 점이 모자라 클러스터가 안 서는 구간이 실측에서 28%였는데,
        스캔 도착만 보면 그동안 코스팅이 돌지 않아 발행이 통째로 끊겼다
        (A는 같은 상황에서 예측값이라도 계속 낸다)."""
        now = time.monotonic()
        self._maybe_log_stats(now)

        if not self._kf.initialized:
            return
        if now - self._last_update_wall <= self.buffer_stale_timeout:
            return

        # 관측이 끊긴 동안만 예측을 이어간다. _last_t도 같이 밀어야 다음
        # 관측이 왔을 때 이미 지나간 구간을 두 번 예측하지 않는다.
        self._advance(self._dt)
        if self._last_t is not None:
            self._last_t += self._dt
        self._coast_count += 1

        if self._coast_count > self.reset_after:
            self._reset_all(f'{self._coast_count}프레임 관측 없음')
            return
        if self._coast_count > self.coast_max:
            self.get_logger().warn(
                f'Tracking LOST: {self._coast_count} coast frames.',
                throttle_duration_sec=2.0)
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        self._publish_pose(header)

    # ------------------------------------------------------------------
    # 로그
    # ------------------------------------------------------------------
    def _maybe_log_stats(self, now):
        if now - self._a_t0 < self._stat_period:
            return
        el = now - self._a_t0
        parts = []
        for i, cfg in enumerate(self._sensors):
            pts, dist = self._a_pts[i], self._a_dist[i]
            parts.append(
                f'{cfg.name}: 스캔{self._a_scans[i]} 클러스터{self._a_clusters[i]} '
                f'갱신{self._a_updates[i]} 기각{self._a_gated[i]}'
                + (f' 점{np.median(pts):.0f} 거리{np.median(dist):.0f}m'
                   if pts and dist else ''))
        self.get_logger().info(
            f'[순차] 트랙 {"O" if self._kf.initialized else "X"}  '
            f'갱신 {sum(self._a_updates) / el:.1f}Hz  '
            f'코스팅 {self._coast_count}  ' + '  |  '.join(parts))
        n = len(self._sensors)
        self._a_scans = [0] * n
        self._a_clusters = [0] * n
        self._a_updates = [0] * n
        self._a_gated = [0] * n
        self._a_pts = [[] for _ in range(n)]
        self._a_dist = [[] for _ in range(n)]
        self._a_t0 = now

    def _open_debug(self):
        """비행마다 _001, _002 … 로 갈아 연다. 한 파일에 두 비행이 섞이면
        실측과 관측을 잘못 맞대 엉뚱한 결론이 나온다(예전에 한 번 헛짚었다)."""
        base = os.path.expanduser(self._dbg_path)
        d = os.path.dirname(base) or '.'
        os.makedirs(d, exist_ok=True)
        stem = os.path.basename(base)[:-4] if base.endswith('.csv') \
            else os.path.basename(base)
        pat = re.compile(re.escape(stem) + r'_(\d+)\.csv')
        used = [int(m.group(1)) for m in map(pat.fullmatch, os.listdir(d)) if m]
        p = os.path.join(d, f'{stem}_{max(used) + 1 if used else 1:03d}.csv')
        self._dbg_f = open(p, 'w', newline='')
        self._dbg_w = csv.writer(self._dbg_f)
        self._dbg_w.writerow(DEBUG_HEADER)
        self._dbg_path = p
        self.get_logger().info(f'순차 디버그 CSV: {p}')

    def _log_row(self, idx, z, n_pts, dist, dt, sigma, innov, mahal,
                 accepted, stamp):
        if self._dbg_w is None:
            return
        p = self._kf.position
        sig = float(np.sqrt(np.trace(self._kf.cov_pos) / 3.0))
        self._dbg_w.writerow([
            f'{stamp.sec + stamp.nanosec * 1e-9:.3f}', idx + 1, n_pts,
            f'{dist:.2f}', f'{dt:.4f}', f'{sigma:.4f}',
            f'{z[0]:.4f}', f'{z[1]:.4f}', f'{z[2]:.4f}',
            f'{innov:.4f}', f'{mahal:.2f}', int(accepted),
            f'{p[0]:.4f}', f'{p[1]:.4f}', f'{p[2]:.4f}', f'{sig:.4f}'])
        self._dbg_f.flush()

    def close(self):
        """CSV 마무리 저장. main()의 종료 처리에서 호출."""
        self._finalize_waypoint_episode()
        try:
            self._wp_csv_f.close()
        except Exception:
            pass
        self.get_logger().info(f'경로점 추정 결과 저장: {self._wp_csv_path}')
        if self._dbg_f is not None:
            try:
                self._dbg_f.close()
                self.get_logger().info(f'순차 디버그 저장: {self._dbg_path}')
            except Exception:
                pass

    def _roi_mask(self, pts_w: np.ndarray) -> np.ndarray:
        """ROI 상자 통과 여부. 센서 마스트 주변은 파낸다.

        상자 비교만으로도 inf/nan은 자동으로 걸러진다(비교가 모두 False).

        마스트 배제가 왜 필요한지는 sensor_exclude_radius 선언부 주석 참고 —
        하우징 높이가 목표 호버고도와 같아 z로는 가를 수 없다. 두 파이프라인
실시간 경로와 재생 경로가 같은 필터를 쓰도록 여기 한 곳에 둔다."""
        mask = (
            (pts_w[:, 0] >= self.roi_x_min) & (pts_w[:, 0] <= self.roi_x_max) &
            (pts_w[:, 1] >= self.roi_y_min) & (pts_w[:, 1] <= self.roi_y_max) &
            (pts_w[:, 2] >= self.roi_z_min) & (pts_w[:, 2] <= self.roi_z_max)
        )
        r = self.sensor_exclude_radius
        if r > 0.0:
            r2 = r * r
            for cfg in self._sensors:
                dx = pts_w[:, 0] - cfg.x
                dy = pts_w[:, 1] - cfg.y
                mask &= (dx * dx + dy * dy) > r2
        return mask

    def _publish_pose(self, header: Header):
        pos = self._kf.position
        cov_pos = self._kf.cov_pos

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        msg.pose.pose.orientation.w = 1.0  # yaw unknown from position-only observations

        # 6×6 flat covariance (pose: x,y,z,rx,ry,rz)
        cov = [0.0] * 36
        for i in range(3):
            cov[i * 6 + i] = float(cov_pos[i, i])
        msg.pose.covariance = cov

        self._pub_pose.publish(msg)

    # ------------------------------------------------------------------
    # Waypoint(마커) 호버 감지 + 위치 재추정
    # ------------------------------------------------------------------
    def _update_waypoint_tracking(self, centroid: np.ndarray):
        """가장 가까운 마커 반경 안에서 정지 상태가 유지되는 동안 raw
        centroid 표본을 모은다. KF 평활값이 아니라 raw 값을 쓰는 이유:
        정지 전환 구간에서 KF는 관성으로 인해 실제 위치보다 살짝 지연된
        값을 내므로, 여러 프레임의 raw 표본을 median으로 묶는 쪽이 더
        정확하다. 속도 판정만 KF의 평활 속도를 쓴다 (raw centroid 프레임간
        노이즈만으로는 속도 추정이 너무 불안정함).

        속도추정이 정지 중에도 프레임 단위로 미세하게 흔들려서 문턱값을
        순간적으로 넘나드는 경우가 실측에서 확인됨 — 그때마다 즉시
        끊어버리면 진짜 하나의 호버가 여러 조각으로 쪼개진다. 그래서
        이탈이 waypoint_grace_frames 이내로 짧으면 무시하고 같은 방문으로
        이어붙이고, 그보다 길게 지속돼야 실제로 떠난 것으로 본다."""
        speed = float(np.linalg.norm(self._kf.velocity[:2]))  # 수평 속도만
        slow = speed <= self.waypoint_speed_threshold

        if self._wp_active_id is not None:
            # 이미 특정 마커를 추적 중이면, "가장 가까운 마커"를 새로 찾지
            # 않고 지금 추적 중인 그 마커와의 거리만 본다 (경계에서 다른
            # 마커로 순간 전환되는 걸 방지).
            mx, my = self._marker_xy[self._wp_active_id]
            d = math.hypot(centroid[0] - mx, centroid[1] - my)
            near = d <= self.waypoint_gate_radius
            if near and slow:
                self._wp_samples.append(centroid.copy())
                self._wp_break_count = 0
                return
            self._wp_break_count += 1
            if self._wp_break_count <= self.waypoint_grace_frames:
                return  # 짧은 이탈 — 유예, 같은 방문 유지 (이 프레임 표본은 버림)
            self._finalize_waypoint_episode()
            # 이 프레임에 새 방문이 시작될 수도 있으니 아래로 계속 진행

        # 활성 방문이 없는 상태 — 새 방문 시작 여부 판정
        best_id, best_d = None, None
        for mid, (mx, my) in self._marker_xy.items():
            d = math.hypot(centroid[0] - mx, centroid[1] - my)
            if best_d is None or d < best_d:
                best_id, best_d = mid, d

        near = best_d is not None and best_d <= self.waypoint_gate_radius
        if near and slow:
            self._wp_active_id = best_id
            self._wp_samples = [centroid.copy()]
            self._wp_break_count = 0

    def _finalize_waypoint_episode(self):
        mid = self._wp_active_id
        samples = self._wp_samples
        self._wp_active_id = None
        self._wp_samples = []
        self._wp_break_count = 0
        if mid is None or len(samples) < self.waypoint_min_frames:
            return

        arr = np.array(samples)
        med = np.median(arr, axis=0)
        n = len(samples)
        duration = n / self.update_rate

        self._wp_visit_counts[mid] = self._wp_visit_counts.get(mid, 0) + 1
        visit = self._wp_visit_counts[mid]

        self.get_logger().info(
            f'[경로점] id{mid} 방문#{visit}: '
            f'({med[0]:.3f}, {med[1]:.3f}, {med[2]:.3f})  '
            f'n={n}샘플({duration:.1f}초)')

        self._wp_csv_w.writerow([
            mid, visit, f'{med[0]:.4f}', f'{med[1]:.4f}', f'{med[2]:.4f}',
            n, f'{duration:.2f}'])
        self._wp_csv_f.flush()

        self._publish_waypoint_marker(mid, visit, med)

    def _publish_waypoint_marker(self, marker_id: int, visit: int, pos: np.ndarray):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'waypoint_estimates'
        m.id = marker_id * 100 + visit
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
        m.pose.position.z = float(pos[2])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.3
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 1.0
        m.color.a = 0.9
        self._wp_markers.markers.append(m)
        # close()가 종료 경로에서 마지막 에피소드를 마무리하는데, 그 시점엔
        # 컨텍스트가 이미 내려가 있다. 여기서 예외가 나면 close()의 나머지
        # (CSV 닫기)가 통째로 건너뛰어진다.
        if not rclpy.ok():
            return
        self._pub_waypoints.publish(self._wp_markers)

    def _pc2_to_numpy(self, msg: PointCloud2) -> np.ndarray | None:
        """Return Nx3 float32 in sensor frame, non-finite points stripped.

        sensor_msgs_py.read_points returns a *structured* numpy array
        (fields 'x','y','z'), which cannot be cast to float32 directly.
        We stack the named fields into an Nx3 array instead.

        Non-returning LiDAR rays come back as +/-inf (not NaN), so
        skip_nans does NOT remove them — we filter with np.isfinite.
        """
        try:
            arr = pc2_util.read_points(
                msg, field_names=('x', 'y', 'z'), skip_nans=True)
            if arr is None or arr.size == 0:
                return None
            pts = np.stack(
                [arr['x'], arr['y'], arr['z']], axis=-1).astype(np.float32)
            pts = pts.reshape(-1, 3)
            # Drop inf (rays that hit nothing within range)
            finite = np.isfinite(pts).all(axis=1)
            pts = pts[finite]
            if pts.shape[0] == 0:
                return None
            return pts
        except Exception as e:
            self.get_logger().error(
                f'PC2 read error: {e}', throttle_duration_sec=5.0)
            return None

    def _numpy_to_pc2(self, pts: np.ndarray, ref_header: Header) -> PointCloud2:
        h = Header()
        h.stamp = ref_header.stamp
        h.frame_id = 'map'
        return pc2_util.create_cloud_xyz32(h, pts.tolist())


# ---------------------------------------------------------------------------

def main():
    rclpy.init()
    node = None
    try:
        node = SeqTracker()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
