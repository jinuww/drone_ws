#!/usr/bin/env python3
"""
Final dual-Ouster OS1-128 drone tracker for a 24 m x 15 m competition field.

Architecture
------------
Each LiDAR is processed independently:

  /lidarN/points
      -> calibrated sensor-to-world transform
      -> ROI crop
      -> optional static-background subtraction
      -> optional voxel + statistical outlier filtering
      -> DBSCAN
      -> cluster validation
      -> per-LiDAR position + covariance

The two POSITION measurements are then time-aligned and consistency-checked.
Whichever detections survive those checks are fed to one 6-state
constant-velocity filter, by either of two paths (use_polar_ekf):

  true   per-sensor sequential EKF update in each sensor's polar frame
         (range, azimuth, elevation). Anisotropic R: the error ellipsoid is
         long along that sensor's line of sight, so the weak axis of one
         LiDAR is the strong axis of the other. Because the two detections
         are folded in separately rather than averaged, they do not have to
         be simultaneous: the older scan is carried forward along the
         estimated velocity first (async_tolerance_sec). See ekf.py.
  false  covariance/information-weighted fusion into one world-frame
         position, then a single linear KF update (previous behaviour).

Important design properties
---------------------------
* Point clouds are NEVER fused before drone detection.
* Each input frame is processed at most once.
* Small per-sensor timestamp queues are used to select the closest pair.
* Per-sensor world/filtered clouds and per-sensor pose estimates are published.
* Background files include an extrinsic-calibration signature and are rejected
  automatically if either sensor pose or the ROI changes.
* PTP is not configured by this node, but inter-sensor timestamp differences
  are continuously measured and reported. Without phase locking the two scans
  are offset by up to half a period; that offset is corrected, not tolerated.
* Fusion uses 3x3 measurement covariance, not point-count-only weighting.
* Per-frame EKF diagnostics (which sensors contributed, NIS, sigma) are written
  to ekf_diagnostics.csv in the run folder; check_ekf_fusion.py grades them.

Nominal physical layout
-----------------------
World frame:
  x: 0 ... 24 m
  y: 0 ... 15 m
  z: up

Nominal LiDAR mounting:
  lidar1 = (12, -1, 2.0), facing +Y
  lidar2 = (12, 16, 2.0), facing -Y

These nominal extrinsics are ONLY defaults. For final operation, replace them
through the YAML file with Kabsch/SVD-calibrated x,y,z,roll,pitch,yaw values.

Inputs
------
/lidar1/points                       sensor_msgs/msg/PointCloud2
/lidar2/points                       sensor_msgs/msg/PointCloud2
/competition/run_config              std_msgs/msg/String (JSON)
/competition/mission_event           std_msgs/msg/String

Outputs
-------
/lidar1/points_world                 sensor_msgs/msg/PointCloud2
/lidar2/points_world                 sensor_msgs/msg/PointCloud2
/lidar1/filtered_points              sensor_msgs/msg/PointCloud2
/lidar2/filtered_points              sensor_msgs/msg/PointCloud2
/drone/lidar1_pose                   geometry_msgs/msg/PoseWithCovarianceStamped
/drone/lidar2_pose                   geometry_msgs/msg/PoseWithCovarianceStamped
/drone/estimated_pose                geometry_msgs/msg/PoseWithCovarianceStamped
/lidar/cluster_markers               visualization_msgs/msg/MarkerArray
/drone/waypoint_estimates            visualization_msgs/msg/MarkerArray

Background learning
-------------------
Run once AFTER PTP setup and AFTER extrinsic calibration, with an empty field:

  ros2 run <pkg> <exe> --ros-args \
      -p learn_background:=true \
      -p use_background:=true

The node stores a calibration signature in the .npz background file. If the
extrinsics or ROI later change, the old background is ignored and must be
regenerated.
"""

import csv
import hashlib
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as SciPyRotation
from sensor_msgs.msg import PointCloud2
from sklearn.cluster import DBSCAN
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray

from drone_localization.ekf import EKF6D

try:
    from sensor_msgs_py import point_cloud2 as pc2_util
except ImportError:
    from sensor_msgs import point_cloud2 as pc2_util


@dataclass
class Detection:
    sensor_index: int
    sensor_name: str
    position: np.ndarray
    covariance: np.ndarray
    cluster_points: int
    spread: float
    quality: float
    sensor_range: float
    stamp: object
    candidate_count: int
    # Seconds this detection was extrapolated forward to line up with the
    # other sensor's scan time, and what that carry could have got wrong.
    # Both 0.0 when it is the reference (newest) scan.
    time_offset: float = 0.0
    extrapolation_std: float = 0.0


@dataclass
class FrameInfo:
    """What happened on one processing tick, for the EKF diagnostics log.

    n_clouds is the discriminator that matters when a sensor goes missing:
    1 means that sensor never delivered a cloud (driver, cabling, topic, or
    a stale timestamp), 2 means it delivered one but no cluster came out of
    it (ROI, background subtraction, or simply too few returns).
    """
    n_clouds: int = 0
    n_cand: int = 0
    sel_pts: int = 0
    sync_dt: float | None = None


class SensorCfg:
    """Fixed LiDAR pose and sensor-to-world rigid transform."""

    def __init__(self, name, topic, x, y, z, roll, pitch, yaw, extrinsic_rmse):
        self.name = str(name)
        self.topic = str(topic)
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.roll, self.pitch, self.yaw = float(roll), float(pitch), float(yaw)
        self.extrinsic_rmse = max(0.0, float(extrinsic_rmse))
        self.R_s2w = SciPyRotation.from_euler(
            'xyz', [self.roll, self.pitch, self.yaw]
        ).as_matrix()
        self.t_s2w = np.array([self.x, self.y, self.z], dtype=np.float64)

    def to_world(self, pts_sensor: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_sensor, dtype=np.float64).reshape(-1, 3)
        return (self.R_s2w @ pts.T).T + self.t_s2w

    def signature_values(self):
        return [
            self.x, self.y, self.z,
            self.roll, self.pitch, self.yaw,
            self.extrinsic_rmse,
        ]


class LidarDroneTracker(Node):

    def __init__(self):
        super().__init__('lidar_drone_tracker')

        self._declare_parameters()
        self._read_parameters()
        self._build_sensors()

        self._kf = EKF6D(
            accel_noise_std=self.accel_noise_std,
            default_meas_noise_std=self.meas_noise_pos,
        )
        if self.seed_at_home:
            self._kf.init(
                self.home_xyz.copy(),
                np.eye(3) * self.home_seed_std**2,
            )

        self._coast_count = 0
        self._last_process_monotonic = time.monotonic()

        # Timestamped queues: each item = (stamp_sec, points_world, recv_time, stamp_msg)
        self._cloud_queues = [
            deque(maxlen=self.queue_depth),
            deque(maxlen=self.queue_depth),
        ]
        self._last_processed_stamp_ns = [None, None]

        # Background, separately per LiDAR.
        self._bg_points = [None, None]
        self._bg_trees = [None, None]
        self._bg_counts = [dict(), dict()]
        self._bg_frames_seen = [0, 0]
        self._background_saved = False
        self._background_signature = self._make_calibration_signature()
        self._load_background_if_available()

        # Timestamp diagnostics.
        self._sync_samples_ms = deque(maxlen=self.sync_stats_window)
        self._sync_last_log = time.monotonic()

        # Waypoint hover state.
        self._wp_active_id = None
        self._wp_samples = []
        self._wp_break_count = 0
        self._wp_visit_counts = {}
        self._wp_markers = MarkerArray()

        # Competition run / CSV state.
        self._run_active = False
        self._run_output_dir = None
        self._team_name = ''
        self._team_id = ''
        self._run_id = 0
        self._raw_csv_f = self._raw_csv_w = None
        self._traj_csv_f = self._traj_csv_w = None
        self._wp_csv_f = self._wp_csv_w = None
        self._ekf_csv_f = self._ekf_csv_w = None
        self._raw_csv_path = self._traj_csv_path = self._wp_csv_path = None
        self._ekf_csv_path = None

        # Stats.
        self._stat_period = 2.0
        self._stat_t0 = time.monotonic()
        self._reset_stats()

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
                PointCloud2, self._sensors[0].topic,
                lambda msg: self._store_cloud(0, msg), sensor_qos),
            self.create_subscription(
                PointCloud2, self._sensors[1].topic,
                lambda msg: self._store_cloud(1, msg), sensor_qos),
        ]

        self._run_config_sub = self.create_subscription(
            String, '/competition/run_config', self._run_config_callback, 10)
        self._event_sub = self.create_subscription(
            String, '/competition/mission_event', self._mission_event_callback, 20)

        # Final pose and per-sensor diagnostic outputs.
        self._pub_pose = self.create_publisher(
            PoseWithCovarianceStamped, '/drone/estimated_pose', reliable_qos)
        self._pub_sensor_pose = [
            self.create_publisher(
                PoseWithCovarianceStamped, '/drone/lidar1_pose', reliable_qos),
            self.create_publisher(
                PoseWithCovarianceStamped, '/drone/lidar2_pose', reliable_qos),
        ]
        self._pub_world = [
            self.create_publisher(PointCloud2, '/lidar1/points_world', sensor_qos),
            self.create_publisher(PointCloud2, '/lidar2/points_world', sensor_qos),
        ]
        self._pub_filtered = [
            self.create_publisher(PointCloud2, '/lidar1/filtered_points', sensor_qos),
            self.create_publisher(PointCloud2, '/lidar2/filtered_points', sensor_qos),
        ]
        self._pub_markers = self.create_publisher(
            MarkerArray, '/lidar/cluster_markers', reliable_qos)
        self._pub_waypoints = self.create_publisher(
            MarkerArray, '/drone/waypoint_estimates', reliable_qos)

        self._proc_timer = self.create_timer(self._dt, self._process)

        names = ', '.join(
            f'{c.name}({c.topic} @ {c.x:.3f},{c.y:.3f},{c.z:.3f}; '
            f'rpy={c.roll:.5f},{c.pitch:.5f},{c.yaw:.5f}; '
            f'extrinsic_rmse={c.extrinsic_rmse:.3f}m)'
            for c in self._sensors)
        self.get_logger().info(
            f'Final dual-LiDAR tracker ready: {names}; update={self.update_rate:.1f} Hz')
        self.get_logger().info(
            f'Field ROI: x=[{self.roi_x_min:.2f},{self.roi_x_max:.2f}], '
            f'y=[{self.roi_y_min:.2f},{self.roi_y_max:.2f}], '
            f'z=[{self.roi_z_min:.2f},{self.roi_z_max:.2f}] frame={self.world_frame}')
        pair_tol = (self.async_tolerance if self.use_polar_ekf
                    else self.sync_tolerance)
        self.get_logger().info(
            f'Scan pairing limit={pair_tol*1000.0:.1f} ms '
            f'(fused-path sync tolerance={self.sync_tolerance*1000.0:.1f} ms); '
            f'agreement gate={self.sensor_agreement_dist:.2f} m')
        if self.use_polar_ekf:
            self.get_logger().info(
                'Filter update: per-sensor polar EKF '
                f'(sigma_range={self.ekf_sigma_range:.3f} m, '
                f'beam={math.degrees(self.ekf_beam_res):.2f} deg, '
                f'sigma_body={self.ekf_sigma_body:.3f} m)')
        else:
            self.get_logger().info(
                'Filter update: covariance-fused linear KF (use_polar_ekf=false)')
        self.get_logger().info(
            f'Background signature={self._background_signature[:12]}...; '
            f'use_background={self.use_background}; learn_background={self.learn_background}')
        if self.learn_background:
            self.get_logger().warn(
                'BACKGROUND LEARNING MODE: field must be static and empty; '
                f'collecting {self.background_frames} frames PER LiDAR.')
        if self.warn_nominal_extrinsics:
            self._warn_if_nominal_extrinsics()

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def _declare_parameters(self):
        # Nominal competition-field extrinsics. Replace via YAML after Kabsch calibration.
        self.declare_parameter('lidar1_topic', '/lidar1/points')
        self.declare_parameter('lidar1_x', 12.0)
        self.declare_parameter('lidar1_y', -1.0)
        self.declare_parameter('lidar1_z', 2.0)
        self.declare_parameter('lidar1_roll', 0.0)
        self.declare_parameter('lidar1_pitch', 0.0)
        self.declare_parameter('lidar1_yaw', math.pi / 2.0)
        self.declare_parameter('lidar1_extrinsic_rmse', 0.05)

        self.declare_parameter('lidar2_topic', '/lidar2/points')
        self.declare_parameter('lidar2_x', 12.0)
        self.declare_parameter('lidar2_y', 16.0)
        self.declare_parameter('lidar2_z', 2.0)
        self.declare_parameter('lidar2_roll', 0.0)
        self.declare_parameter('lidar2_pitch', 0.0)
        self.declare_parameter('lidar2_yaw', -math.pi / 2.0)
        self.declare_parameter('lidar2_extrinsic_rmse', 0.05)
        self.declare_parameter('warn_nominal_extrinsics', True)

        # Timing and timestamp pairing.
        self.declare_parameter('lidar_update_rate', 10.0)
        self.declare_parameter('buffer_stale_timeout', 0.30)
        self.declare_parameter('queue_depth', 8)
        self.declare_parameter('sync_tolerance_sec', 0.010)
        # Two free-running OS1s are not phase locked, so their scans are taken
        # up to half a period apart (50 ms at 10 Hz). Averaging positions from
        # two different instants is wrong, which is what sync_tolerance_sec
        # guards -- but the polar EKF never averages: it folds each detection
        # in separately. So it only needs the two scans close enough that
        # extrapolating the older one over the gap is accurate, which is a far
        # looser requirement. Beyond this second limit a sensor really is
        # stale (dropped frames, a dead sensor) and is still discarded.
        self.declare_parameter('async_tolerance_sec', 0.06)
        self.declare_parameter('sync_stats_window', 200)
        self.declare_parameter('sensor_agreement_dist', 0.35)

        # 24 x 15 m competition field and 1-3 m flight envelope.
        self.declare_parameter('roi_x_min', 0.0)
        self.declare_parameter('roi_x_max', 24.0)
        self.declare_parameter('roi_y_min', 0.0)
        self.declare_parameter('roi_y_max', 15.0)
        self.declare_parameter('roi_z_min', 0.8)
        self.declare_parameter('roi_z_max', 3.2)
        self.declare_parameter('world_frame', 'world')

        # A competition field is generally not a closed rectangular room.
        self.declare_parameter('remove_boundary_surfaces', False)
        self.declare_parameter('boundary_surface_margin', 0.10)

        # Static background model. Regenerate after any extrinsic/ROI change.
        self.declare_parameter('use_background', True)
        self.declare_parameter('learn_background', False)
        self.declare_parameter(
            'background_file',
            str(Path('~/competition_results/calibration/dual_ouster_background.npz').expanduser()))
        self.declare_parameter('background_frames', 200)
        self.declare_parameter('background_voxel_size', 0.05)
        self.declare_parameter('background_distance', 0.08)
        self.declare_parameter('background_min_observation_ratio', 0.70)

        # Point filtering.
        self.declare_parameter('voxel_leaf_size', 0.04)
        self.declare_parameter('sor_mean_k', 10)
        self.declare_parameter('sor_std_ratio', 2.0)

        # DBSCAN / cluster shape.
        self.declare_parameter('cluster_tolerance', 0.22)
        self.declare_parameter('cluster_min_points', 5)
        self.declare_parameter('cluster_max_points', 2000)
        self.declare_parameter('cluster_max_extent', 1.20)
        self.declare_parameter('cluster_min_extent', 0.02)

        # Per-detection covariance model.
        self.declare_parameter('meas_noise_pos', 0.10)
        self.declare_parameter('centroid_noise_floor', 0.025)
        self.declare_parameter('range_noise_per_meter', 0.002)
        self.declare_parameter('min_detection_std', 0.03)
        self.declare_parameter('max_detection_std', 0.50)

        # Per-sensor polar EKF update.
        # When true, each surviving detection is folded into the filter in its
        # own sensor's polar frame instead of being averaged into one world
        # position first. Sequential updates are mathematically equivalent to a
        # stacked update for a linear/Gaussian filter, so nothing is lost; what
        # is gained is an honest, anisotropic R per sensor. Turn it off to fall
        # back to the covariance-weighted fusion path. Rationale: ekf.py.
        self.declare_parameter('use_polar_ekf', True)
        # Line-of-sight sigma. Dominated by the near-face bias (the LiDAR only
        # hits the side of the airframe facing it), NOT by range precision.
        # Systematic, so it does not shrink with point count.
        self.declare_parameter('ekf_sigma_range', 0.10)
        # OS1-128 angular sampling. 1024x10 Hz -> 0.35 deg horizontally, and
        # 45 deg / 128 = 0.35 deg vertically. Change this if the mode changes.
        self.declare_parameter('ekf_beam_res_deg', 0.35)
        # Expected airframe scale: how far off centre a hit on the near face
        # can be. Unknown per team, same character as ekf_sigma_range.
        self.declare_parameter('ekf_sigma_body', 0.14)

        # KF / association.
        self.declare_parameter('accel_noise_std', 2.0)
        self.declare_parameter('seed_at_home', True)
        self.declare_parameter('home_x', 0.0)
        self.declare_parameter('home_y', 0.0)
        self.declare_parameter('home_z', 1.0)
        self.declare_parameter('home_seed_std', 0.50)
        self.declare_parameter('max_assoc_dist', 1.75)
        self.declare_parameter('gate_sigma_k', 3.0)
        self.declare_parameter('max_assoc_dist_cap', 4.0)
        self.declare_parameter('coast_max_frames', 10)
        self.declare_parameter('reset_after_frames', 50)

        # Waypoint estimation (optional; override with actual competition markers).
        self.declare_parameter('enable_waypoint_tracking', True)
        self.declare_parameter('waypoint_gate_radius', 1.0)
        self.declare_parameter('waypoint_speed_threshold', 0.25)
        self.declare_parameter('waypoint_min_frames', 4)
        self.declare_parameter('waypoint_grace_frames', 5)
        # Default marker coordinates are only valid placeholders inside 24x15 field.
        self.declare_parameter('marker1_x', 5.0)
        self.declare_parameter('marker1_y', 4.0)
        self.declare_parameter('marker2_x', 5.0)
        self.declare_parameter('marker2_y', 11.0)
        self.declare_parameter('marker3_x', 19.0)
        self.declare_parameter('marker3_y', 4.0)
        self.declare_parameter('marker4_x', 12.0)
        self.declare_parameter('marker4_y', 11.0)

    def _read_parameters(self):
        gp = lambda n: self.get_parameter(n).value
        self.update_rate = float(gp('lidar_update_rate'))
        self._dt = 1.0 / max(1e-3, self.update_rate)
        self.buffer_stale_timeout = float(gp('buffer_stale_timeout'))
        self.queue_depth = max(2, int(gp('queue_depth')))
        self.sync_tolerance = max(0.0, float(gp('sync_tolerance_sec')))
        self.async_tolerance = max(
            self.sync_tolerance, float(gp('async_tolerance_sec')))
        self.sync_stats_window = max(20, int(gp('sync_stats_window')))
        self.sensor_agreement_dist = float(gp('sensor_agreement_dist'))
        self.warn_nominal_extrinsics = bool(gp('warn_nominal_extrinsics'))

        self.roi_x_min = float(gp('roi_x_min'))
        self.roi_x_max = float(gp('roi_x_max'))
        self.roi_y_min = float(gp('roi_y_min'))
        self.roi_y_max = float(gp('roi_y_max'))
        self.roi_z_min = float(gp('roi_z_min'))
        self.roi_z_max = float(gp('roi_z_max'))
        self.world_frame = str(gp('world_frame'))
        self.remove_boundary_surfaces = bool(gp('remove_boundary_surfaces'))
        self.boundary_surface_margin = float(gp('boundary_surface_margin'))

        self.use_background = bool(gp('use_background'))
        self.learn_background = bool(gp('learn_background'))
        self.background_file = str(gp('background_file'))
        self.background_frames = max(1, int(gp('background_frames')))
        self.background_voxel_size = float(gp('background_voxel_size'))
        self.background_distance = float(gp('background_distance'))
        self.background_min_obs_ratio = float(gp('background_min_observation_ratio'))

        self.voxel_leaf_size = float(gp('voxel_leaf_size'))
        self.sor_k = int(gp('sor_mean_k'))
        self.sor_std = float(gp('sor_std_ratio'))

        self.eps = float(gp('cluster_tolerance'))
        self.min_pts = int(gp('cluster_min_points'))
        self.max_pts = int(gp('cluster_max_points'))
        self.cluster_max_extent = float(gp('cluster_max_extent'))
        self.cluster_min_extent = float(gp('cluster_min_extent'))

        self.meas_noise_pos = float(gp('meas_noise_pos'))
        self.centroid_noise_floor = float(gp('centroid_noise_floor'))
        self.range_noise_per_meter = float(gp('range_noise_per_meter'))
        self.min_detection_std = float(gp('min_detection_std'))
        self.max_detection_std = float(gp('max_detection_std'))

        self.use_polar_ekf = bool(gp('use_polar_ekf'))
        self.ekf_sigma_range = float(gp('ekf_sigma_range'))
        self.ekf_beam_res = math.radians(float(gp('ekf_beam_res_deg')))
        self.ekf_sigma_body = float(gp('ekf_sigma_body'))

        self.accel_noise_std = float(gp('accel_noise_std'))
        self.seed_at_home = bool(gp('seed_at_home'))
        self.home_xyz = np.array(
            [gp('home_x'), gp('home_y'), gp('home_z')], dtype=np.float64)
        self.home_seed_std = float(gp('home_seed_std'))
        self.max_assoc_dist = float(gp('max_assoc_dist'))
        self.gate_sigma_k = float(gp('gate_sigma_k'))
        self.max_assoc_dist_cap = float(gp('max_assoc_dist_cap'))
        self.coast_max = int(gp('coast_max_frames'))
        self.reset_after = int(gp('reset_after_frames'))

        self.enable_waypoint_tracking = bool(gp('enable_waypoint_tracking'))
        self.waypoint_gate_radius = float(gp('waypoint_gate_radius'))
        self.waypoint_speed_threshold = float(gp('waypoint_speed_threshold'))
        self.waypoint_min_frames = int(gp('waypoint_min_frames'))
        self.waypoint_grace_frames = int(gp('waypoint_grace_frames'))
        self._marker_xy = {
            1: (float(gp('marker1_x')), float(gp('marker1_y'))),
            2: (float(gp('marker2_x')), float(gp('marker2_y'))),
            3: (float(gp('marker3_x')), float(gp('marker3_y'))),
            4: (float(gp('marker4_x')), float(gp('marker4_y'))),
        }

    def _build_sensors(self):
        gp = lambda n: self.get_parameter(n).value
        self._sensors = [
            SensorCfg(
                'lidar1', gp('lidar1_topic'),
                gp('lidar1_x'), gp('lidar1_y'), gp('lidar1_z'),
                gp('lidar1_roll'), gp('lidar1_pitch'), gp('lidar1_yaw'),
                gp('lidar1_extrinsic_rmse')),
            SensorCfg(
                'lidar2', gp('lidar2_topic'),
                gp('lidar2_x'), gp('lidar2_y'), gp('lidar2_z'),
                gp('lidar2_roll'), gp('lidar2_pitch'), gp('lidar2_yaw'),
                gp('lidar2_extrinsic_rmse')),
        ]

    def _warn_if_nominal_extrinsics(self):
        nominal = [
            np.array([12.0, -1.0, 2.0, 0.0, 0.0, math.pi / 2.0]),
            np.array([12.0, 16.0, 2.0, 0.0, 0.0, -math.pi / 2.0]),
        ]
        for i, s in enumerate(self._sensors):
            cur = np.array([s.x, s.y, s.z, s.roll, s.pitch, s.yaw])
            if np.allclose(cur, nominal[i], atol=1e-7, rtol=0.0):
                self.get_logger().warn(
                    f'{s.name}: using NOMINAL extrinsics. Replace YAML values with '
                    'Kabsch-calibrated x,y,z,roll,pitch,yaw before final scoring.')

    # ------------------------------------------------------------------
    # Background model with calibration signature
    # ------------------------------------------------------------------
    def _make_calibration_signature(self) -> str:
        payload = {
            'world_frame': self.world_frame,
            'roi': [
                self.roi_x_min, self.roi_x_max,
                self.roi_y_min, self.roi_y_max,
                self.roi_z_min, self.roi_z_max,
            ],
            'sensors': [s.signature_values() for s in self._sensors],
            'background_voxel_size': self.background_voxel_size,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()

    def _load_background_if_available(self):
        if not self.use_background or self.learn_background:
            return
        p = Path(self.background_file).expanduser()
        if not p.exists():
            self.get_logger().warn(
                f'Background file not found: {p}. Continuing without learned background.')
            return
        try:
            data = np.load(str(p), allow_pickle=False)
            stored_sig = ''
            if 'calibration_signature' in data:
                stored_sig = str(np.asarray(data['calibration_signature']).item())
            if stored_sig != self._background_signature:
                self.get_logger().error(
                    'Background model calibration signature does not match current '
                    'extrinsics/ROI. Ignoring old background; regenerate it.')
                return
            for i, cfg in enumerate(self._sensors):
                key = cfg.name
                if key not in data:
                    self.get_logger().warn(f'No {key} background in {p}')
                    continue
                pts = np.asarray(data[key], dtype=np.float32).reshape(-1, 3)
                if len(pts):
                    self._bg_points[i] = pts
                    self._bg_trees[i] = cKDTree(pts)
                    self.get_logger().info(
                        f'Loaded {len(pts)} background voxels for {cfg.name}')
        except Exception as exc:
            self.get_logger().error(f'Failed to load background model {p}: {exc}')

    def _accumulate_background(self, sensor_index, pts_world):
        if len(pts_world) == 0:
            return
        voxel = max(1e-3, self.background_voxel_size)
        keys = np.floor(pts_world / voxel).astype(np.int32)
        keys = np.unique(keys, axis=0)
        counts = self._bg_counts[sensor_index]
        for k in map(tuple, keys.tolist()):
            counts[k] = counts.get(k, 0) + 1
        self._bg_frames_seen[sensor_index] += 1

    def _background_learning_complete(self):
        return all(n >= self.background_frames for n in self._bg_frames_seen)

    def _finalize_background_learning(self):
        if self._background_saved or not self._background_learning_complete():
            return
        voxel = max(1e-3, self.background_voxel_size)
        payload = {
            'calibration_signature': np.array(self._background_signature),
        }
        for i, cfg in enumerate(self._sensors):
            min_count = max(
                1,
                int(math.ceil(
                    self.background_min_obs_ratio * self._bg_frames_seen[i])))
            kept = [k for k, c in self._bg_counts[i].items() if c >= min_count]
            if kept:
                arr = np.asarray(kept, dtype=np.float32)
                pts = (arr + 0.5) * voxel
            else:
                pts = np.empty((0, 3), dtype=np.float32)
            payload[cfg.name] = pts.astype(np.float32)
            self._bg_points[i] = payload[cfg.name]
            if len(pts):
                self._bg_trees[i] = cKDTree(pts)
            self.get_logger().info(
                f'Background {cfg.name}: {len(pts)} persistent voxels retained.')
        p = Path(self.background_file).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(p), **payload)
        self._background_saved = True
        self.learn_background = False
        self.get_logger().info(f'Background calibration COMPLETE: {p}')

    def _subtract_background(self, sensor_index, points):
        tree = self._bg_trees[sensor_index]
        if not self.use_background or tree is None or len(points) == 0:
            return points
        dists, _ = tree.query(points, k=1)
        return points[dists > self.background_distance]

    # ------------------------------------------------------------------
    # Competition run management
    # ------------------------------------------------------------------
    def _run_config_callback(self, msg):
        try:
            cfg = json.loads(msg.data)
            team_name = str(cfg['team_name']).strip()
            team_id = str(cfg['team_id']).strip()
            run_id = int(cfg['run_id'])
            output_dir = Path(cfg['output_dir']).expanduser()
            if not team_name or not team_id:
                raise ValueError('team_name and team_id must not be empty')
        except Exception as exc:
            self.get_logger().error(f'Invalid /competition/run_config: {exc}')
            return

        if self._run_active:
            self._finalize_waypoint_episode()
        self._close_run_logs()

        output_dir.mkdir(parents=True, exist_ok=True)
        self._run_output_dir = output_dir
        self._team_name = team_name
        self._team_id = team_id
        self._run_id = run_id

        try:
            self._raw_csv_path = output_dir / 'raw_detections.csv'
            self._raw_csv_f = open(self._raw_csv_path, 'w', newline='')
            self._raw_csv_w = csv.writer(self._raw_csv_f)
            self._raw_csv_w.writerow([
                'time_s', 'x_m', 'y_m', 'z_m',
                'cov_x', 'cov_y', 'cov_z',
                'cluster_points', 'candidate_clusters',
                'sensor_count', 'sync_dt_ms'])

            self._traj_csv_path = output_dir / 'drone_trajectory.csv'
            self._traj_csv_f = open(self._traj_csv_path, 'w', newline='')
            self._traj_csv_w = csv.writer(self._traj_csv_f)
            self._traj_csv_w.writerow([
                'time_s', 'x_m', 'y_m', 'z_m',
                'vx_mps', 'vy_mps', 'vz_mps',
                'cov_x', 'cov_y', 'cov_z'])

            self._wp_csv_path = output_dir / 'waypoint_estimates.csv'
            self._wp_csv_f = open(self._wp_csv_path, 'w', newline='')
            self._wp_csv_w = csv.writer(self._wp_csv_f)
            self._wp_csv_w.writerow([
                'marker_id', 'visit', 'x', 'y', 'z', 'n_samples', 'duration_s'])

            # EKF fusion diagnostics. /drone/estimated_pose only carries the
            # position and its covariance, so after a flight there is no way to
            # tell from it whether both LiDARs actually contributed. Record the
            # per-sensor contribution every frame instead; check_ekf_fusion.py
            # grades it.
            self._ekf_csv_path = output_dir / 'ekf_diagnostics.csv'
            self._ekf_csv_f = open(self._ekf_csv_path, 'w', newline='')
            self._ekf_csv_w = csv.writer(self._ekf_csv_f)
            head = ['t', 'mode', 'n_sensors', 'n_clouds', 'n_cand',
                    'sel_pts', 'sync_dt_ms']
            for cfg in self._sensors:
                head += [f'{cfg.name}_pts', f'{cfg.name}_dist',
                         f'{cfg.name}_sigma_ang_deg', f'{cfg.name}_nis']
            head.append('sigma_pos')
            self._ekf_csv_w.writerow(head)

            self._raw_csv_f.flush()
            self._traj_csv_f.flush()
            self._wp_csv_f.flush()
            self._ekf_csv_f.flush()
        except Exception as exc:
            self.get_logger().error(f'Cannot open run CSV files: {exc}')
            self._close_run_logs()
            return

        self._wp_active_id = None
        self._wp_samples = []
        self._wp_break_count = 0
        self._wp_visit_counts = {}
        self._wp_markers = MarkerArray()
        self._coast_count = 0
        if self.seed_at_home:
            self._kf.init(
                self.home_xyz.copy(),
                np.eye(3) * self.home_seed_std**2)
        else:
            self._kf.initialized = False
        self._run_active = False
        self.get_logger().info(
            f'Run configured: team={team_name}, id={team_id}, run={run_id}; '
            f'output={output_dir}')

    def _mission_event_callback(self, msg):
        event = msg.data.strip().upper()
        if event == 'START':
            if self._run_output_dir is None:
                self.get_logger().warn(
                    'START received before /competition/run_config; CSV logging inactive.')
            else:
                self._run_active = True
                self.get_logger().info(
                    f'Logging STARTED for {self._team_name} ({self._team_id}), '
                    f'run {self._run_id}')
        elif event in ('FINISH', 'ABORT'):
            if self._run_active:
                self._finalize_waypoint_episode()
            self._run_active = False
            self._close_run_logs()
            self.get_logger().info(
                f'Run logging closed on {event}: {self._team_name} '
                f'({self._team_id}), run {self._run_id}')

    def _close_run_logs(self):
        for attr in ('_raw_csv_f', '_traj_csv_f', '_wp_csv_f', '_ekf_csv_f'):
            f = getattr(self, attr, None)
            if f is not None:
                try:
                    f.flush()
                    f.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        self._raw_csv_w = self._traj_csv_w = self._wp_csv_w = None
        self._ekf_csv_w = None

    # ------------------------------------------------------------------
    # Input buffering and timestamp pairing
    # ------------------------------------------------------------------
    def _store_cloud(self, sensor_index, msg):
        pts = self._pc2_to_numpy(msg)
        if pts is None or len(pts) == 0:
            return
        pts_w = self._sensors[sensor_index].to_world(pts)
        stamp_s = self._stamp_to_seconds(msg.header.stamp)
        recv_t = time.monotonic()
        q = self._cloud_queues[sensor_index]
        # Ignore duplicate incoming timestamp.
        if q and abs(q[-1][0] - stamp_s) < 1e-12:
            return
        q.append((stamp_s, pts_w, recv_t, msg.header.stamp))

    def _select_unprocessed_pair(self, now):
        """Return [(sensor_index, pts, stamp_msg), ...] with at most one cloud/sensor.

        If both sensors have fresh unprocessed frames, choose the pair with minimum
        |t1-t2|. If only one sensor has an unprocessed fresh frame, return it after
        it has waited approximately sync_tolerance so a matching frame can arrive.
        """
        fresh = []
        for i, q in enumerate(self._cloud_queues):
            candidates = []
            for stamp_s, pts, recv_t, stamp_msg in q:
                stamp_ns = int(round(stamp_s * 1e9))
                if (self._last_processed_stamp_ns[i] is not None and
                        stamp_ns <= self._last_processed_stamp_ns[i]):
                    continue
                if now - recv_t > self.buffer_stale_timeout:
                    continue
                candidates.append((stamp_s, pts, recv_t, stamp_msg))
            fresh.append(candidates)

        if fresh[0] and fresh[1]:
            best = None
            for a in fresh[0]:
                for b in fresh[1]:
                    d = abs(a[0] - b[0])
                    if best is None or d < best[0]:
                        best = (d, a, b)
            dt, a, b = best
            # Mark selected frames processed even if the dt is larger than the
            # fusion tolerance; they can still be processed independently.
            self._last_processed_stamp_ns[0] = int(round(a[0] * 1e9))
            self._last_processed_stamp_ns[1] = int(round(b[0] * 1e9))
            self._record_sync_sample(dt)
            return [(0, a[1], a[3]), (1, b[1], b[3])]

        # One-sensor fallback. Prefer newest fresh unprocessed frame.
        for i in (0, 1):
            if fresh[i]:
                a = fresh[i][-1]
                # Briefly wait for its partner unless it is old enough already.
                if now - a[2] < max(self.sync_tolerance, 0.002):
                    return []
                self._last_processed_stamp_ns[i] = int(round(a[0] * 1e9))
                return [(i, a[1], a[3])]
        return []

    def _record_sync_sample(self, dt_sec):
        ms = float(dt_sec) * 1000.0
        self._sync_samples_ms.append(ms)
        now = time.monotonic()
        if now - self._sync_last_log >= 5.0 and self._sync_samples_ms:
            arr = np.asarray(self._sync_samples_ms, dtype=np.float64)
            self.get_logger().info(
                f'PTP/scan timestamp delta: median={np.median(arr):.3f} ms, '
                f'p95={np.percentile(arr,95):.3f} ms, max={np.max(arr):.3f} ms '
                f'(limit={self.sync_tolerance*1000.0:.1f} ms)')
            self._sync_last_log = now

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------
    def _process(self):
        now = time.monotonic()
        selected = self._select_unprocessed_pair(now)
        if not selected:
            self._maybe_log_stats(now)
            return

        dt = max(1e-3, min(0.5, now - self._last_process_monotonic))
        self._last_process_monotonic = now

        if self._kf.initialized:
            self._kf.predict(dt)
            predicted = self._kf.position
        else:
            predicted = None

        detections = []
        for sensor_index, pts_w, stamp in selected:
            self._stat_sensor_hits[sensor_index] += 1
            self._stat_sensor_pts[sensor_index] += len(pts_w)
            det = self._process_sensor_cloud(sensor_index, pts_w, stamp, predicted)
            if det is not None:
                detections.append(det)

        if self.learn_background:
            self._finalize_background_learning()
            self._maybe_log_stats(now)
            return

        self._stat_frames += 1
        (measurement, cov, out_stamp, fused_points, fused_cands, sensor_count,
         sync_dt, used) = self._fuse_detections(detections, predicted)

        header = Header()
        header.frame_id = self.world_frame
        header.stamp = out_stamp if out_stamp is not None else self.get_clock().now().to_msg()

        frame = FrameInfo(
            n_clouds=len(selected), n_cand=fused_cands,
            sel_pts=fused_points, sync_dt=sync_dt)

        if measurement is not None:
            if not self._kf.initialized:
                self._kf.init(measurement, cov)
                self._log_ekf_diag(header, 'init', [], frame)
            else:
                self._apply_measurement(used, measurement, cov, header, frame)
            self._coast_count = 0
            self._stat_detect += 1
            self._stat_cluster_pts += fused_points
            self._stat_cand += fused_cands

            if self._run_active and self._raw_csv_w is not None:
                self._raw_csv_w.writerow([
                    f'{self._stamp_seconds(header):.9f}',
                    f'{measurement[0]:.6f}', f'{measurement[1]:.6f}', f'{measurement[2]:.6f}',
                    f'{cov[0,0]:.9f}', f'{cov[1,1]:.9f}', f'{cov[2,2]:.9f}',
                    int(fused_points), int(fused_cands), int(sensor_count),
                    '' if sync_dt is None else f'{sync_dt*1000.0:.3f}',
                ])
                self._raw_csv_f.flush()

            if self._run_active and self.enable_waypoint_tracking:
                self._update_waypoint_tracking(measurement)
            self._publish_pose(header)
        else:
            self._coast_count += 1
            if self._kf.initialized:
                if self._coast_count <= self.coast_max:
                    self._publish_pose(header)
                elif self._coast_count > self.reset_after:
                    self.get_logger().warn(
                        f'Tracking lost for {self._coast_count} frames -> KF reset')
                    self._kf.initialized = False
                    self._coast_count = 0
                    if self._run_active:
                        self._finalize_waypoint_episode()

        self._maybe_log_stats(now)

    # ------------------------------------------------------------------
    # Per-sensor processing
    # ------------------------------------------------------------------
    def _process_sensor_cloud(self, sensor_index, pts_w, stamp, predicted):
        cfg = self._sensors[sensor_index]
        pts = np.asarray(pts_w, dtype=np.float64)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < self.min_pts:
            return None

        header = Header()
        header.stamp = stamp
        header.frame_id = self.world_frame

        # Publish full transformed cloud.
        try:
            self._pub_world[sensor_index].publish(
                self._numpy_to_pc2(pts.astype(np.float32), header))
        except Exception as exc:
            self.get_logger().warn(
                f'{cfg.name} world-cloud publish failed: {exc}',
                throttle_duration_sec=5.0)

        # ROI.
        mask = (
            (pts[:, 0] >= self.roi_x_min) & (pts[:, 0] <= self.roi_x_max) &
            (pts[:, 1] >= self.roi_y_min) & (pts[:, 1] <= self.roi_y_max) &
            (pts[:, 2] >= self.roi_z_min) & (pts[:, 2] <= self.roi_z_max)
        )
        pts = pts[mask]
        if len(pts) < self.min_pts:
            return None

        # Optional rejection of points close to the ROI boundary planes.
        if self.remove_boundary_surfaces:
            pts = self._remove_boundary_surfaces(pts)
            if len(pts) < self.min_pts:
                return None

        # Background learning occurs after transform/ROI, separately per sensor.
        if self.learn_background:
            self._accumulate_background(sensor_index, pts)
            try:
                self._pub_filtered[sensor_index].publish(
                    self._numpy_to_pc2(pts.astype(np.float32), header))
            except Exception:
                pass
            return None

        pts = self._subtract_background(sensor_index, pts)
        if len(pts) < self.min_pts:
            return None

        if self.voxel_leaf_size > 0.0:
            pts = self._voxel_downsample(pts, self.voxel_leaf_size)
        if len(pts) < self.min_pts:
            return None

        if self.sor_k > 1 and len(pts) > max(self.sor_k * 3, 30):
            pts = self._remove_statistical_outliers(pts)
        if len(pts) < self.min_pts:
            return None

        try:
            self._pub_filtered[sensor_index].publish(
                self._numpy_to_pc2(pts.astype(np.float32), header))
        except Exception as exc:
            self.get_logger().warn(
                f'{cfg.name} filtered-cloud publish failed: {exc}',
                throttle_duration_sec=5.0)

        try:
            labels = DBSCAN(
                eps=self.eps,
                min_samples=self.min_pts,
                metric='euclidean',
                n_jobs=-1,
            ).fit_predict(pts)
        except Exception as exc:
            self.get_logger().error(
                f'{cfg.name} DBSCAN error: {exc}', throttle_duration_sec=5.0)
            return None

        det = self._select_drone_cluster(sensor_index, pts, labels, stamp, predicted)
        if det is not None:
            self._publish_sensor_pose(det)
        return det

    def _remove_boundary_surfaces(self, points):
        m = max(0.0, self.boundary_surface_margin)
        keep = (
            (points[:, 0] > self.roi_x_min + m) &
            (points[:, 0] < self.roi_x_max - m) &
            (points[:, 1] > self.roi_y_min + m) &
            (points[:, 1] < self.roi_y_max - m) &
            (points[:, 2] > self.roi_z_min + m) &
            (points[:, 2] < self.roi_z_max - m)
        )
        return points[keep]

    @staticmethod
    def _voxel_downsample(points, voxel_size):
        if len(points) == 0 or voxel_size <= 0.0:
            return points
        idx = np.floor(points / voxel_size).astype(np.int64)
        _, inverse = np.unique(idx, axis=0, return_inverse=True)
        n = int(inverse.max()) + 1
        sums = np.zeros((n, 3), dtype=np.float64)
        np.add.at(sums, inverse, points)
        counts = np.bincount(inverse, minlength=n)
        return (sums / counts[:, None]).astype(np.float32)

    def _remove_statistical_outliers(self, points):
        if self.sor_k <= 1 or len(points) <= self.sor_k:
            return points
        k = min(self.sor_k + 1, len(points))
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=k)
        mean_dist = np.mean(distances[:, 1:], axis=1)
        threshold = mean_dist.mean() + self.sor_std * mean_dist.std()
        return points[mean_dist <= threshold]

    # ------------------------------------------------------------------
    # Cluster selection and covariance estimation
    # ------------------------------------------------------------------
    def _select_drone_cluster(self, sensor_index, pts, labels, stamp, predicted):
        cfg = self._sensors[sensor_index]
        unique = [label for label in np.unique(labels) if label >= 0]
        if not unique:
            return None

        candidates = []
        marker_msgs = MarkerArray()
        marker_id = sensor_index * 1000

        for label in unique:
            cluster = pts[labels == label]
            n = len(cluster)
            if n < self.min_pts or n > self.max_pts:
                continue
            mins = np.min(cluster, axis=0)
            maxs = np.max(cluster, axis=0)
            extent = maxs - mins
            if np.max(extent) > self.cluster_max_extent:
                continue
            if np.max(extent) < self.cluster_min_extent:
                continue

            centroid = np.median(cluster, axis=0)
            radial = np.linalg.norm(cluster - centroid, axis=1)
            spread = float(np.median(radial)) if len(radial) else 1.0
            sensor_range = float(np.linalg.norm(centroid - cfg.t_s2w))
            cov = self._estimate_detection_covariance(
                cfg, cluster, centroid, spread, sensor_range)

            # A diagnostic quality score only; fusion itself uses covariance.
            quality = 1.0 / max(1e-9, float(np.trace(cov)))
            assoc_dist = 0.0 if predicted is None else float(
                np.linalg.norm(centroid - predicted))
            candidates.append((centroid, cov, n, spread, quality, sensor_range, assoc_dist))

            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = stamp
            m.ns = f'{cfg.name}_clusters'
            m.id = marker_id
            marker_id += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(centroid[0])
            m.pose.position.y = float(centroid[1])
            m.pose.position.z = float(centroid[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.20
            if sensor_index == 0:
                m.color.r, m.color.g, m.color.b = 0.1, 1.0, 0.2
            else:
                m.color.r, m.color.g, m.color.b = 0.2, 0.4, 1.0
            m.color.a = 0.8
            m.lifetime.sec = 1
            marker_msgs.markers.append(m)

        if marker_msgs.markers:
            self._pub_markers.publish(marker_msgs)
        if not candidates:
            return None

        if predicted is not None:
            candidates.sort(key=lambda c: c[6])
            best = candidates[0]
            sigma = float(np.sqrt(max(1e-9, np.trace(self._kf.cov_pos) / 3.0)))
            gate = min(
                self.max_assoc_dist + self.gate_sigma_k * sigma,
                self.max_assoc_dist_cap)
            if best[6] > gate:
                return None
        else:
            candidates.sort(key=lambda c: c[4], reverse=True)
            best = candidates[0]

        return Detection(
            sensor_index=sensor_index,
            sensor_name=cfg.name,
            position=np.asarray(best[0], dtype=np.float64),
            covariance=np.asarray(best[1], dtype=np.float64),
            cluster_points=int(best[2]),
            spread=float(best[3]),
            quality=float(best[4]),
            sensor_range=float(best[5]),
            stamp=stamp,
            candidate_count=len(candidates),
        )

    def _estimate_detection_covariance(self, cfg, cluster, centroid, spread, sensor_range):
        """Estimate a conservative 3x3 world-frame centroid covariance.

        Components:
          * empirical cluster covariance / N  -> centroid uncertainty
          * centroid noise floor
          * range-dependent term
          * extrinsic calibration RMSE

        This is not a complete OS1 beam model, but it is statistically meaningful
        and much better than treating point count as a fusion weight.
        """
        n = max(1, len(cluster))
        if n >= 3:
            empirical = np.cov(np.asarray(cluster, dtype=np.float64).T)
            if empirical.shape != (3, 3) or not np.isfinite(empirical).all():
                empirical = np.eye(3) * max(spread, self.centroid_noise_floor) ** 2
        else:
            empirical = np.eye(3) * max(spread, self.centroid_noise_floor) ** 2

        centroid_cov = empirical / float(n)
        base_std = self.centroid_noise_floor
        range_std = self.range_noise_per_meter * max(0.0, sensor_range)
        extr_std = cfg.extrinsic_rmse
        isotropic_var = base_std**2 + range_std**2 + extr_std**2
        cov = centroid_cov + np.eye(3) * isotropic_var

        # Clamp eigenvalues to avoid extreme confidence or instability.
        cov = 0.5 * (cov + cov.T)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(
            vals,
            self.min_detection_std**2,
            self.max_detection_std**2)
        return vecs @ np.diag(vals) @ vecs.T

    # ------------------------------------------------------------------
    # Asynchronous scans
    # ------------------------------------------------------------------
    def _time_align_detections(self, detections):
        """Bring detections taken at different instants onto one time base.

        The two LiDARs free-run, so without PTP phase locking their scans are
        offset by up to half a period. That offset is not measurement error --
        the drone really was somewhere else when the older scan was taken --
        so it is corrected, not tolerated: each older detection is carried
        forward to the newest scan's time along the filter's current velocity.

        The extrapolation is only as good as that velocity, so the detection's
        covariance grows by what the carry could have got wrong: the velocity
        estimate's own uncertainty over the gap, plus whatever acceleration
        the constant-velocity model did not see. While hovering -- which is
        when waypoints are scored -- the velocity is ~0 and both terms vanish.

        Returns the reference time and detections with `time_offset` set.
        Nothing happens before the filter has a velocity to extrapolate with.
        """
        if len(detections) < 2 or not self._kf.initialized:
            return detections

        t_ref = max(self._stamp_to_seconds(d.stamp) for d in detections)
        vel = self._kf.velocity
        vel_std = float(np.sqrt(max(1e-12, np.trace(self._kf.P[3:, 3:]) / 3.0)))

        aligned = []
        for d in detections:
            offset = t_ref - self._stamp_to_seconds(d.stamp)
            if offset <= 1e-6:
                d.time_offset = 0.0
                d.extrapolation_std = 0.0
                aligned.append(d)
                continue
            extra = vel_std * offset + 0.5 * self.accel_noise_std * offset**2
            aligned.append(Detection(
                sensor_index=d.sensor_index,
                sensor_name=d.sensor_name,
                position=d.position + vel * offset,
                covariance=d.covariance + np.eye(3) * extra**2,
                cluster_points=d.cluster_points,
                spread=d.spread,
                quality=d.quality,
                sensor_range=d.sensor_range,
                stamp=d.stamp,
                candidate_count=d.candidate_count,
                time_offset=offset,
                extrapolation_std=extra))
        return aligned

    # ------------------------------------------------------------------
    # Covariance-based position-level fusion
    # ------------------------------------------------------------------
    def _fuse_detections(self, detections, predicted):
        """Combine per-sensor detections into one measurement.

        Returns the fused position/covariance plus the list of detections that
        actually survived the timing and agreement checks. The polar EKF path
        updates from that list directly; the fused position is still needed for
        the linear fallback, for filter seeding, and for the raw CSV.
        """
        if not detections:
            return None, None, None, 0, 0, 0, None, []

        if len(detections) == 1:
            d = detections[0]
            return (
                d.position.copy(), d.covariance.copy(), d.stamp,
                d.cluster_points, d.candidate_count, 1, None, [d])

        # Exactly one detection per sensor is expected.
        d0, d1 = self._time_align_detections(detections[:2])
        t0 = self._stamp_to_seconds(d0.stamp)
        t1 = self._stamp_to_seconds(d1.stamp)
        dt = abs(t0 - t1)
        disagreement = float(np.linalg.norm(d0.position - d1.position))

        # How far apart the two scans may be and still both be used. The polar
        # EKF folds them in one at a time, so it only needs the older one to
        # survive being carried forward (_time_align_detections). The fused
        # path averages the two positions outright and so still demands they
        # be simultaneous.
        pair_tolerance = (self.async_tolerance if self.use_polar_ekf
                          else self.sync_tolerance)

        if dt <= pair_tolerance and disagreement <= self.sensor_agreement_dist:
            C0 = self._regularize_cov3(d0.covariance)
            C1 = self._regularize_cov3(d1.covariance)
            I0 = np.linalg.inv(C0)
            I1 = np.linalg.inv(C1)
            Cf = np.linalg.inv(I0 + I1)
            zf = Cf @ (I0 @ d0.position + I1 @ d1.position)
            stamp = d0.stamp if t0 >= t1 else d1.stamp
            return (
                zf, Cf, stamp,
                d0.cluster_points + d1.cluster_points,
                d0.candidate_count + d1.candidate_count,
                2, dt, [d0, d1])

        # Too far apart to bridge -> one sensor is stale, not merely offset.
        if dt > pair_tolerance:
            chosen = d0 if t0 > t1 else d1
            self.get_logger().warn(
                f'LiDAR timestamp mismatch {dt*1000.0:.2f} ms > '
                f'{pair_tolerance*1000.0:.2f} ms; using {chosen.sensor_name}',
                throttle_duration_sec=2.0)
            return (
                chosen.position.copy(), chosen.covariance.copy(), chosen.stamp,
                chosen.cluster_points, chosen.candidate_count, 1, dt, [chosen])

        # Synchronized but spatially inconsistent. Prefer prediction-consistent
        # measurement; at cold start prefer lower covariance trace.
        if predicted is not None:
            e0 = float(np.linalg.norm(d0.position - predicted))
            e1 = float(np.linalg.norm(d1.position - predicted))
            chosen = d0 if e0 <= e1 else d1
        else:
            chosen = d0 if np.trace(d0.covariance) <= np.trace(d1.covariance) else d1
        self.get_logger().warn(
            f'LiDAR position disagreement {disagreement:.3f} m > '
            f'{self.sensor_agreement_dist:.3f} m; using {chosen.sensor_name}. '
            'Check the relative extrinsics between the two LiDARs.',
            throttle_duration_sec=2.0)
        return (
            chosen.position.copy(), chosen.covariance.copy(), chosen.stamp,
            chosen.cluster_points, chosen.candidate_count, 1, dt, [chosen])

    # ------------------------------------------------------------------
    # Filter update: per-sensor polar EKF, or fused linear KF
    # ------------------------------------------------------------------
    def _apply_measurement(self, used, measurement, cov, header, frame):
        """Fold the surviving detections into the single tracking filter.

        Default path updates the filter once per sensor, each in that sensor's
        own polar frame (range, azimuth, elevation). For a linear/Gaussian
        filter, sequential updates are exactly equivalent to one stacked
        update, so nothing is lost by not pre-fusing; what is gained is an R
        that is honest about direction. A LiDAR only sees the face of the
        airframe pointing at it, so its centroid sits a little in front of the
        true centre along its own line of sight -- an error that does NOT
        shrink with point count. Declaring that as sigma_range in polar, and
        letting the Jacobian rotate it into the world, gives each sensor an
        ellipsoid stretched along its own view. In the ns layout the two views
        are widely separated, so one sensor's weak axis is the other's strong
        axis and the near-face bias largely cancels by geometry. See ekf.py.

        Falls back to the covariance-fused linear update when use_polar_ekf is
        off, when no per-sensor detection survived, or when every polar update
        hit the azimuth singularity.
        """
        if not self.use_polar_ekf or not used:
            if self.use_polar_ekf:
                self._stat_ekf_fallback += 1
            self._kf.update(measurement, cov)
            self._log_ekf_diag(
                header, 'fused' if self.use_polar_ekf else 'linear', [], frame)
            return

        applied = []
        for det in used:
            cfg = self._sensors[det.sensor_index]
            # Extrinsic calibration error is isotropic in the world, so it goes
            # into both polar axes. It does not average down with point count.
            # An extrapolated detection is less certain in every direction,
            # so its uncertainty joins both polar axes. It is zero for the
            # newest scan and near zero while hovering.
            iso = math.hypot(cfg.extrinsic_rmse, det.extrapolation_std)
            s_ang = EKF6D.sigma_ang(
                self.ekf_beam_res, det.sensor_range, det.cluster_points,
                self.ekf_sigma_body, iso)
            s_range = math.hypot(self.ekf_sigma_range, iso)
            res = self._kf.update_polar(
                det.position, cfg.R_s2w, cfg.t_s2w, s_range, s_ang)
            nis = None
            if res is not None:
                nis = res[0]
                self._stat_nis.append(nis)
            applied.append((det.sensor_index, det.cluster_points,
                            det.sensor_range, s_ang, nis))

        n_ok = sum(1 for a in applied if a[4] is not None)
        if n_ok == 0:
            # Drone directly above/below every sensor: azimuth undefined.
            # Do not drop the frame; use the world-frame path instead.
            self._stat_ekf_fallback += 1
            self._kf.update(measurement, cov)
            self._log_ekf_diag(header, 'fused', [], frame)
            return

        self._stat_ekf_frames += 1
        self._stat_ekf_sensors += n_ok
        self._log_ekf_diag(header, 'per_sensor', applied, frame)

    def _log_ekf_diag(self, header, mode, applied, frame):
        """Record how each sensor contributed to this frame's update.

        /drone/estimated_pose carries only the position and its covariance, so
        nothing in it says whether both LiDARs were really folded in. Without
        this log, a run that silently tracked on one sensor looks exactly like
        a healthy one until the waypoint errors come out wrong.
        """
        if self._ekf_csv_w is None:
            return
        by = {i: (n, s, nis) for i, n, _, s, nis in applied}
        pos = self._kf.position
        # Count only sensors whose update actually went through, so the fusion
        # rate in check_ekf_fusion.py cannot be flattered by skipped updates.
        n_used = sum(1 for a in applied if a[4] is not None)
        # n_clouds and sync_dt_ms are what turn "a sensor is missing" into a
        # cause: no cloud at all, a cloud with no cluster in it, or a cloud
        # whose scan time was too far from the other sensor's.
        row = [f'{self._stamp_seconds(header):.3f}', mode, n_used,
               int(frame.n_clouds), int(frame.n_cand), int(frame.sel_pts),
               '' if frame.sync_dt is None else f'{frame.sync_dt * 1000.0:.3f}']
        for i, cfg in enumerate(self._sensors):
            n, s, nis = by.get(i, (0, float('nan'), None))
            # Distance is written whether or not the sensor contributed: a
            # missing sensor is only diagnosable if you know how far away the
            # drone was at the time.
            dist = float(np.linalg.norm(cfg.t_s2w - pos))
            row += [n, f'{dist:.3f}', f'{math.degrees(s):.4f}',
                    '' if nis is None else f'{nis:.4f}']
        sigma = float(np.sqrt(max(1e-12, np.trace(self._kf.cov_pos) / 3.0)))
        row.append(f'{sigma:.4f}')
        try:
            self._ekf_csv_w.writerow(row)
        except Exception as exc:
            self.get_logger().warn(
                f'EKF diagnostics write failed: {exc}', throttle_duration_sec=5.0)

    @staticmethod
    def _regularize_cov3(cov):
        cov = np.asarray(cov, dtype=np.float64).reshape(3, 3)
        cov = 0.5 * (cov + cov.T)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-6, 100.0)
        return vecs @ np.diag(vals) @ vecs.T

    # ------------------------------------------------------------------
    # Pose publishers
    # ------------------------------------------------------------------
    def _publish_sensor_pose(self, det: Detection):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = det.stamp
        msg.header.frame_id = self.world_frame
        msg.pose.pose.position.x = float(det.position[0])
        msg.pose.pose.position.y = float(det.position[1])
        msg.pose.pose.position.z = float(det.position[2])
        msg.pose.pose.orientation.w = 1.0
        cov = [0.0] * 36
        for r in range(3):
            for c in range(3):
                cov[r * 6 + c] = float(det.covariance[r, c])
        msg.pose.covariance = cov
        self._pub_sensor_pose[det.sensor_index].publish(msg)

    def _publish_pose(self, header):
        pos = self._kf.position
        vel = self._kf.velocity
        cov_pos = self._kf.cov_pos
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = self.world_frame
        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        msg.pose.pose.orientation.w = 1.0
        cov = [0.0] * 36
        for r in range(3):
            for c in range(3):
                cov[r * 6 + c] = float(cov_pos[r, c])
        msg.pose.covariance = cov
        self._pub_pose.publish(msg)

        if self._run_active and self._traj_csv_w is not None:
            self._traj_csv_w.writerow([
                f'{self._stamp_seconds(header):.9f}',
                f'{pos[0]:.6f}', f'{pos[1]:.6f}', f'{pos[2]:.6f}',
                f'{vel[0]:.6f}', f'{vel[1]:.6f}', f'{vel[2]:.6f}',
                f'{cov_pos[0,0]:.9f}', f'{cov_pos[1,1]:.9f}', f'{cov_pos[2,2]:.9f}',
            ])
            self._traj_csv_f.flush()

    # ------------------------------------------------------------------
    # Waypoint hover estimation
    # ------------------------------------------------------------------
    def _update_waypoint_tracking(self, measurement):
        speed = float(np.linalg.norm(self._kf.velocity[:2]))
        slow = speed <= self.waypoint_speed_threshold

        if self._wp_active_id is not None:
            mx, my = self._marker_xy[self._wp_active_id]
            d = math.hypot(measurement[0] - mx, measurement[1] - my)
            if d <= self.waypoint_gate_radius and slow:
                self._wp_samples.append(measurement.copy())
                self._wp_break_count = 0
                return
            self._wp_break_count += 1
            if self._wp_break_count <= self.waypoint_grace_frames:
                return
            self._finalize_waypoint_episode()

        best_id = None
        best_d = None
        for marker_id, (mx, my) in self._marker_xy.items():
            d = math.hypot(measurement[0] - mx, measurement[1] - my)
            if best_d is None or d < best_d:
                best_id, best_d = marker_id, d
        if best_d is not None and best_d <= self.waypoint_gate_radius and slow:
            self._wp_active_id = best_id
            self._wp_samples = [measurement.copy()]
            self._wp_break_count = 0

    def _finalize_waypoint_episode(self):
        marker_id = self._wp_active_id
        samples = self._wp_samples
        self._wp_active_id = None
        self._wp_samples = []
        self._wp_break_count = 0
        if marker_id is None or len(samples) < self.waypoint_min_frames:
            return
        arr = np.asarray(samples)
        med = np.median(arr, axis=0)
        n = len(samples)
        duration = n / self.update_rate
        self._wp_visit_counts[marker_id] = self._wp_visit_counts.get(marker_id, 0) + 1
        visit = self._wp_visit_counts[marker_id]
        self.get_logger().info(
            f'Waypoint {marker_id} visit {visit}: '
            f'({med[0]:.3f},{med[1]:.3f},{med[2]:.3f}), n={n}, '
            f'duration={duration:.1f}s')
        if self._run_active and self._wp_csv_w is not None:
            self._wp_csv_w.writerow([
                marker_id, visit,
                f'{med[0]:.4f}', f'{med[1]:.4f}', f'{med[2]:.4f}',
                n, f'{duration:.2f}'])
            self._wp_csv_f.flush()
        self._publish_waypoint_marker(marker_id, visit, med)

    def _publish_waypoint_marker(self, marker_id, visit, pos):
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'waypoint_estimates'
        m.id = marker_id * 100 + visit
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
        m.pose.position.z = float(pos[2])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.30
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 1.0, 0.9
        self._wp_markers.markers.append(m)
        # close() finalizes a last episode from inside the shutdown path, where
        # the context is already torn down. Publishing there raises and would
        # skip _close_run_logs(), losing whatever the raw/trajectory CSV
        # buffers still hold.
        if not rclpy.ok():
            return
        self._pub_waypoints.publish(self._wp_markers)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _reset_stats(self):
        self._stat_frames = 0
        self._stat_detect = 0
        self._stat_cluster_pts = 0
        self._stat_cand = 0
        self._stat_sensor_pts = [0, 0]
        self._stat_sensor_hits = [0, 0]
        self._stat_ekf_frames = 0      # frames updated per-sensor (polar EKF)
        self._stat_ekf_sensors = 0     # sum of sensors used in those frames
        self._stat_ekf_fallback = 0    # frames that fell back to fused update
        self._stat_nis = []            # normalised innovation squared

    def _maybe_log_stats(self, now):
        if now - self._stat_t0 < self._stat_period:
            return
        if self.learn_background:
            status = ', '.join(
                f'{cfg.name}:{min(self._bg_frames_seen[i], self.background_frames)}/'
                f'{self.background_frames}'
                for i, cfg in enumerate(self._sensors))
            self.get_logger().info(f'Background learning [{status}]')
            self._stat_t0 = now
            return

        f = self._stat_frames
        if f == 0:
            self.get_logger().warn(
                'No NEW LiDAR frame pairs/frames processed in the last '
                f'{self._stat_period:.0f}s; check topics and timestamps.')
        else:
            det = self._stat_detect
            rate = 100.0 * det / f
            avg_pts = self._stat_cluster_pts / det if det else 0.0
            avg_cand = self._stat_cand / f
            sensors = []
            for i, cfg in enumerate(self._sensors):
                hits = self._stat_sensor_hits[i]
                avg = self._stat_sensor_pts[i] / hits if hits else 0.0
                sensors.append(f'{cfg.name}:{avg:.0f}pts/{hits}frames')
            self.get_logger().info(
                f'Detection {det}/{f} ({rate:.0f}%), '
                f'cluster={avg_pts:.1f}pts, candidates={avg_cand:.1f}, '
                f'sensors=[{" + ".join(sensors)}]')
            if self.use_polar_ekf and det:
                ekf_f = self._stat_ekf_frames
                per_frame = self._stat_ekf_sensors / ekf_f if ekf_f else 0.0
                # NIS follows chi-square with 3 dof, so the mean should sit
                # near 3.0. Far above means the filter is overconfident
                # (sigma too small); far below means it is ignoring good data.
                nis = (f'{np.mean(self._stat_nis):.2f}'
                       if self._stat_nis else 'n/a')
                self.get_logger().info(
                    f'Polar EKF {ekf_f}/{det} frames, '
                    f'{per_frame:.2f} sensors/frame, '
                    f'fused fallback {self._stat_ekf_fallback}, '
                    f'NIS mean {nis} (ideal 3.0)')
        self._stat_t0 = now
        self._reset_stats()

    # ------------------------------------------------------------------
    # PointCloud2 / timestamps
    # ------------------------------------------------------------------
    def _pc2_to_numpy(self, msg):
        try:
            arr = pc2_util.read_points(
                msg, field_names=('x', 'y', 'z'), skip_nans=True)
            if arr is None:
                return None
            if isinstance(arr, np.ndarray):
                if arr.size == 0:
                    return None
                if arr.dtype.names is not None:
                    pts = np.stack([arr['x'], arr['y'], arr['z']], axis=-1)
                else:
                    pts = np.asarray(arr)[..., :3]
            else:
                pts = np.array([[p[0], p[1], p[2]] for p in arr], dtype=np.float32)
            pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
            pts = pts[np.isfinite(pts).all(axis=1)]
            return pts if len(pts) else None
        except Exception as exc:
            self.get_logger().error(
                f'PointCloud2 read error: {exc}', throttle_duration_sec=5.0)
            return None

    def _numpy_to_pc2(self, pts, ref_header):
        h = Header()
        h.stamp = ref_header.stamp
        h.frame_id = self.world_frame
        return pc2_util.create_cloud_xyz32(h, pts.tolist())

    @staticmethod
    def _stamp_to_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _stamp_seconds(header):
        return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def close(self):
        if self.learn_background and self._background_learning_complete():
            self._finalize_background_learning()
        if self._run_active:
            self._finalize_waypoint_episode()
        self._run_active = False
        self._close_run_logs()
        self.get_logger().info('LiDAR tracker closed; run CSV files flushed.')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LidarDroneTracker()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()