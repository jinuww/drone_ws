#!/usr/bin/env python3

import csv
import math
import threading
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
)

from sensor_msgs.msg import PointCloud2

try:
    from sensor_msgs_py import point_cloud2
except ImportError:
    from sensor_msgs import point_cloud2

from scipy.spatial.transform import Rotation as SciPyRotation


class ExtrinsicCalibrator(Node):

    def __init__(self):
        super().__init__('extrinsic_calibrator')

        # ============================================================
        # Parameters
        # ============================================================

        self.declare_parameter(
            'lidar1_topic',
            '/lidar1/points'
        )

        self.declare_parameter(
            'lidar2_topic',
            '/lidar2/points'
        )

        self.declare_parameter(
            'capture_frames',
            20
        )

        self.declare_parameter(
            'minimum_points',
            10
        )

        self.lidar1_topic = (
            self.get_parameter(
                'lidar1_topic'
            ).value
        )

        self.lidar2_topic = (
            self.get_parameter(
                'lidar2_topic'
            ).value
        )

        self.capture_frames = int(
            self.get_parameter(
                'capture_frames'
            ).value
        )

        self.minimum_points = int(
            self.get_parameter(
                'minimum_points'
            ).value
        )

        # ============================================================
        # Latest raw clouds
        # ============================================================

        self.latest_cloud = {
            1: None,
            2: None,
        }

        # ============================================================
        # Capture state
        # ============================================================

        self.capture_request = {
            1: None,
            2: None,
        }

        # Correspondence data:
        #
        # sensor_points[1] =
        # [
        #   [x_L1, y_L1, z_L1],
        #   ...
        # ]
        #
        # world_points[1] =
        # [
        #   [x_W, y_W, z_W],
        #   ...
        # ]

        self.sensor_points = {
            1: [],
            2: [],
        }

        self.world_points = {
            1: [],
            2: [],
        }

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ============================================================
        # Subscribers
        # ============================================================

        self.create_subscription(
            PointCloud2,
            self.lidar1_topic,
            lambda msg: self.cloud_callback(
                msg,
                1
            ),
            sensor_qos,
        )

        self.create_subscription(
            PointCloud2,
            self.lidar2_topic,
            lambda msg: self.cloud_callback(
                msg,
                2
            ),
            sensor_qos,
        )

        self.get_logger().info(
            'Dual OS1 extrinsic calibration node started.'
        )

        self.get_logger().info(
            f'LiDAR 1: {self.lidar1_topic}'
        )

        self.get_logger().info(
            f'LiDAR 2: {self.lidar2_topic}'
        )

        # Interactive command interface
        self.command_thread = threading.Thread(
            target=self.command_loop,
            daemon=True
        )

        self.command_thread.start()

    # ================================================================
    # PointCloud2 -> numpy
    # ================================================================

    def pointcloud_to_numpy(
        self,
        msg
    ):

        try:

            cloud = point_cloud2.read_points(
                msg,
                field_names=(
                    'x',
                    'y',
                    'z'
                ),
                skip_nans=True
            )

            if isinstance(
                cloud,
                np.ndarray
            ):

                if cloud.size == 0:
                    return None

                if cloud.dtype.names:

                    points = np.stack(
                        [
                            cloud['x'],
                            cloud['y'],
                            cloud['z']
                        ],
                        axis=-1
                    )

                else:

                    points = np.asarray(
                        cloud
                    )[..., :3]

            else:

                points = np.array(
                    [
                        [
                            p[0],
                            p[1],
                            p[2]
                        ]
                        for p in cloud
                    ],
                    dtype=np.float64
                )

            points = np.asarray(
                points,
                dtype=np.float64
            ).reshape(
                -1,
                3
            )

            finite = np.isfinite(
                points
            ).all(
                axis=1
            )

            points = points[
                finite
            ]

            if len(points) == 0:
                return None

            return points

        except Exception as exc:

            self.get_logger().error(
                f'PointCloud2 conversion failed: {exc}'
            )

            return None

    # ================================================================
    # LiDAR callback
    # ================================================================

    def cloud_callback(
        self,
        msg,
        sensor_id
    ):

        points = (
            self.pointcloud_to_numpy(
                msg
            )
        )

        if points is None:
            return

        self.latest_cloud[
            sensor_id
        ] = points

        request = self.capture_request[
            sensor_id
        ]

        if request is None:
            return

        self.process_capture_frame(
            sensor_id,
            points
        )

    # ================================================================
    # Capture one calibration target
    # ================================================================

    def process_capture_frame(
        self,
        sensor_id,
        points
    ):

        request = self.capture_request[
            sensor_id
        ]

        if request is None:
            return

        (
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax
        ) = request['roi']

        mask = (
            (points[:, 0] >= xmin)
            &
            (points[:, 0] <= xmax)
            &
            (points[:, 1] >= ymin)
            &
            (points[:, 1] <= ymax)
            &
            (points[:, 2] >= zmin)
            &
            (points[:, 2] <= zmax)
        )

        target_points = (
            points[
                mask
            ]
        )

        if (
            len(target_points)
            <
            self.minimum_points
        ):
            return

        # ------------------------------------------------------------
        # Median rather than mean:
        #
        # Less sensitive to isolated points.
        # ------------------------------------------------------------

        frame_centroid = np.median(
            target_points,
            axis=0
        )

        request[
            'centroids'
        ].append(
            frame_centroid
        )

        n = len(
            request[
                'centroids'
            ]
        )

        if n < self.capture_frames:
            return

        # ------------------------------------------------------------
        # Final target location in raw sensor frame
        #
        # Median of per-frame medians.
        # ------------------------------------------------------------

        sensor_centroid = np.median(
            np.asarray(
                request[
                    'centroids'
                ]
            ),
            axis=0
        )

        world_point = request[
            'world'
        ]

        self.sensor_points[
            sensor_id
        ].append(
            sensor_centroid.copy()
        )

        self.world_points[
            sensor_id
        ].append(
            world_point.copy()
        )

        target_number = len(
            self.sensor_points[
                sensor_id
            ]
        )

        self.get_logger().info(
            '\n'
            f'LiDAR {sensor_id}: '
            f'Target {target_number} captured\n'
            f'  Raw sensor centroid = '
            f'({sensor_centroid[0]:.6f}, '
            f'{sensor_centroid[1]:.6f}, '
            f'{sensor_centroid[2]:.6f})\n'
            f'  Known world point    = '
            f'({world_point[0]:.6f}, '
            f'{world_point[1]:.6f}, '
            f'{world_point[2]:.6f})'
        )

        self.capture_request[
            sensor_id
        ] = None

    # ================================================================
    # Kabsch / SVD rigid registration
    # ================================================================

    @staticmethod
    def solve_kabsch(
        sensor_points,
        world_points
    ):

        A = np.asarray(
            sensor_points,
            dtype=np.float64
        )

        B = np.asarray(
            world_points,
            dtype=np.float64
        )

        if len(A) != len(B):

            raise ValueError(
                'Sensor/world point counts differ.'
            )

        if len(A) < 3:

            raise ValueError(
                'At least 3 point correspondences '
                'are required.'
            )

        # ------------------------------------------------------------
        # Check geometry
        #
        # Points should not all lie on one line.
        # ------------------------------------------------------------

        rank = np.linalg.matrix_rank(
            A - np.mean(
                A,
                axis=0
            )
        )

        if rank < 2:

            raise ValueError(
                'Calibration targets are nearly '
                'collinear. Use targets distributed '
                'throughout the calibration volume.'
            )

        # ------------------------------------------------------------
        # Centroids
        # ------------------------------------------------------------

        centroid_A = np.mean(
            A,
            axis=0
        )

        centroid_B = np.mean(
            B,
            axis=0
        )

        # ------------------------------------------------------------
        # Remove centroids
        # ------------------------------------------------------------

        AA = (
            A
            -
            centroid_A
        )

        BB = (
            B
            -
            centroid_B
        )

        # ------------------------------------------------------------
        # Cross covariance
        # ------------------------------------------------------------

        H = (
            AA.T
            @
            BB
        )

        # ------------------------------------------------------------
        # SVD
        # ------------------------------------------------------------

        U, S, Vt = (
            np.linalg.svd(
                H
            )
        )

        # ------------------------------------------------------------
        # Rotation
        #
        # We require:
        #
        # det(R) = +1
        #
        # not a reflection.
        # ------------------------------------------------------------

        R_s2w = (
            Vt.T
            @
            U.T
        )

        if (
            np.linalg.det(
                R_s2w
            )
            <
            0.0
        ):

            Vt[-1, :] *= -1.0

            R_s2w = (
                Vt.T
                @
                U.T
            )

        # ------------------------------------------------------------
        # Translation
        #
        # world = R * sensor + t
        # ------------------------------------------------------------

        t_s2w = (
            centroid_B
            -
            R_s2w
            @
            centroid_A
        )

        # ------------------------------------------------------------
        # Compute transformed calibration points
        # ------------------------------------------------------------

        A_world = (
            R_s2w
            @
            A.T
        ).T + t_s2w

        errors = np.linalg.norm(
            A_world
            -
            B,
            axis=1
        )

        rmse = math.sqrt(
            np.mean(
                errors ** 2
            )
        )

        max_error = float(
            np.max(
                errors
            )
        )

        # ------------------------------------------------------------
        # Euler angles
        #
        # Same convention as SensorCfg:
        #
        # R.from_euler('xyz',
        #              [roll,pitch,yaw])
        # ------------------------------------------------------------

        rotation = (
            SciPyRotation.from_matrix(
                R_s2w
            )
        )

        roll, pitch, yaw = (
            rotation.as_euler(
                'xyz',
                degrees=False
            )
        )

        return {
            'R': R_s2w,
            't': t_s2w,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'errors': errors,
            'rmse': rmse,
            'max_error': max_error,
            'singular_values': S,
        }

    # ================================================================
    # Print calibration result
    # ================================================================

    def solve_sensor(
        self,
        sensor_id
    ):

        sensor_points = (
            self.sensor_points[
                sensor_id
            ]
        )

        world_points = (
            self.world_points[
                sensor_id
            ]
        )

        n = len(
            sensor_points
        )

        if n < 3:

            self.get_logger().error(
                f'LiDAR {sensor_id}: only '
                f'{n} targets captured. '
                'Need at least 3; '
                '6 or more recommended.'
            )

            return

        try:

            result = self.solve_kabsch(
                sensor_points,
                world_points
            )

        except Exception as exc:

            self.get_logger().error(
                f'Calibration failed: {exc}'
            )

            return

        R_s2w = result[
            'R'
        ]

        t = result[
            't'
        ]

        roll = result[
            'roll'
        ]

        pitch = result[
            'pitch'
        ]

        yaw = result[
            'yaw'
        ]

        print()
        print(
            '=' * 70
        )

        print(
            f'LIDAR {sensor_id} '
            f'EXTRINSIC CALIBRATION'
        )

        print(
            '=' * 70
        )

        print(
            '\nRotation matrix '
            'R_sensor_to_world:\n'
        )

        print(
            np.array2string(
                R_s2w,
                precision=9,
                suppress_small=True
            )
        )

        print(
            '\nTranslation '
            't_sensor_to_world:'
        )

        print(
            f'[{t[0]:.9f}, '
            f'{t[1]:.9f}, '
            f'{t[2]:.9f}]'
        )

        print(
            '\nEuler angles '
            '(radians):'
        )

        print(
            f'roll  = {roll:.9f}'
        )

        print(
            f'pitch = {pitch:.9f}'
        )

        print(
            f'yaw   = {yaw:.9f}'
        )

        print(
            '\nEuler angles '
            '(degrees):'
        )

        print(
            f'roll  = '
            f'{math.degrees(roll):.6f}'
        )

        print(
            f'pitch = '
            f'{math.degrees(pitch):.6f}'
        )

        print(
            f'yaw   = '
            f'{math.degrees(yaw):.6f}'
        )

        print(
            '\nCalibration errors:'
        )

        for i, error in enumerate(
            result[
                'errors'
            ],
            start=1
        ):

            print(
                f'  Target {i}: '
                f'{error:.4f} m'
            )

        print(
            f'\nRMSE      = '
            f'{result["rmse"]:.4f} m'
        )

        print(
            f'Max error = '
            f'{result["max_error"]:.4f} m'
        )

        print(
            '\nYAML parameters:\n'
        )

        print(
            f'lidar{sensor_id}_x: '
            f'{t[0]:.9f}'
        )

        print(
            f'lidar{sensor_id}_y: '
            f'{t[1]:.9f}'
        )

        print(
            f'lidar{sensor_id}_z: '
            f'{t[2]:.9f}'
        )

        print(
            f'lidar{sensor_id}_roll: '
            f'{roll:.9f}'
        )

        print(
            f'lidar{sensor_id}_pitch: '
            f'{pitch:.9f}'
        )

        print(
            f'lidar{sensor_id}_yaw: '
            f'{yaw:.9f}'
        )

        print(
            '=' * 70
        )

        print()

    # ================================================================
    # Save correspondences
    # ================================================================

    def save_correspondences(
        self,
        sensor_id,
        filename
    ):

        path = Path(
            filename
        ).expanduser()

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            path,
            'w',
            newline=''
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow([
                'target',
                'sensor_x',
                'sensor_y',
                'sensor_z',
                'world_x',
                'world_y',
                'world_z',
            ])

            for i, (
                sensor_point,
                world_point
            ) in enumerate(
                zip(
                    self.sensor_points[
                        sensor_id
                    ],
                    self.world_points[
                        sensor_id
                    ]
                ),
                start=1
            ):

                writer.writerow([
                    i,
                    *sensor_point.tolist(),
                    *world_point.tolist(),
                ])

        self.get_logger().info(
            f'LiDAR {sensor_id} '
            f'correspondences saved: {path}'
        )

    # ================================================================
    # Command interface
    # ================================================================

    def command_loop(self):

        print()
        print(
            '=============================================================='
        )
        print(
            'OS1-128 EXTRINSIC CALIBRATION'
        )
        print(
            '=============================================================='
        )
        print()
        print(
            'Commands:'
        )
        print()
        print(
            'capture <lidar> '
            '<world_x> <world_y> <world_z> '
            '<xmin> <xmax> '
            '<ymin> <ymax> '
            '<zmin> <zmax>'
        )
        print()
        print(
            'Example:'
        )
        print()
        print(
            'capture 1 '
            '2.0 2.0 1.5 '
            '1.8 2.2 '
            '-0.3 0.3 '
            '-0.3 0.3'
        )
        print()
        print(
            'solve 1'
        )
        print(
            'solve 2'
        )
        print()
        print(
            'list 1'
        )
        print(
            'list 2'
        )
        print()
        print(
            'delete 1'
        )
        print()
        print(
            'save 1 ~/lidar1_calibration.csv'
        )
        print()
        print(
            '=============================================================='
        )
        print()

        while rclpy.ok():

            try:

                command = input(
                    'calibration> '
                ).strip()

            except EOFError:

                return

            if not command:
                continue

            parts = command.split()

            try:

                # ----------------------------------------------------
                # CAPTURE
                # ----------------------------------------------------

                if (
                    parts[0].lower()
                    ==
                    'capture'
                ):

                    if len(parts) != 11:

                        print(
                            'Expected:\n'
                            'capture lidar '
                            'WX WY WZ '
                            'xmin xmax '
                            'ymin ymax '
                            'zmin zmax'
                        )

                        continue

                    sensor_id = int(
                        parts[1]
                    )

                    if sensor_id not in (
                        1,
                        2
                    ):

                        print(
                            'LiDAR must be 1 or 2.'
                        )

                        continue

                    world = np.array(
                        [
                            float(
                                parts[2]
                            ),
                            float(
                                parts[3]
                            ),
                            float(
                                parts[4]
                            ),
                        ],
                        dtype=np.float64
                    )

                    roi = tuple(
                        float(x)
                        for x in parts[
                            5:11
                        ]
                    )

                    self.capture_request[
                        sensor_id
                    ] = {
                        'world': world,
                        'roi': roi,
                        'centroids': [],
                    }

                    print(
                        f'Capturing LiDAR '
                        f'{sensor_id} target...'
                    )

                    print(
                        f'Known world point: '
                        f'{world}'
                    )

                    print(
                        f'Raw LiDAR ROI: '
                        f'{roi}'
                    )

                    print(
                        f'Waiting for '
                        f'{self.capture_frames} '
                        f'valid scans...'
                    )

                # ----------------------------------------------------
                # SOLVE
                # ----------------------------------------------------

                elif (
                    parts[0].lower()
                    ==
                    'solve'
                ):

                    sensor_id = int(
                        parts[1]
                    )

                    self.solve_sensor(
                        sensor_id
                    )

                # ----------------------------------------------------
                # LIST
                # ----------------------------------------------------

                elif (
                    parts[0].lower()
                    ==
                    'list'
                ):

                    sensor_id = int(
                        parts[1]
                    )

                    for i, (
                        sensor,
                        world
                    ) in enumerate(
                        zip(
                            self.sensor_points[
                                sensor_id
                            ],
                            self.world_points[
                                sensor_id
                            ]
                        ),
                        start=1
                    ):

                        print(
                            f'{i}: '
                            f'sensor={sensor} '
                            f'world={world}'
                        )

                # ----------------------------------------------------
                # DELETE LAST
                # ----------------------------------------------------

                elif (
                    parts[0].lower()
                    ==
                    'delete'
                ):

                    sensor_id = int(
                        parts[1]
                    )

                    if self.sensor_points[
                        sensor_id
                    ]:

                        self.sensor_points[
                            sensor_id
                        ].pop()

                        self.world_points[
                            sensor_id
                        ].pop()

                        print(
                            f'Last LiDAR '
                            f'{sensor_id} '
                            f'correspondence deleted.'
                        )

                # ----------------------------------------------------
                # SAVE
                # ----------------------------------------------------

                elif (
                    parts[0].lower()
                    ==
                    'save'
                ):

                    sensor_id = int(
                        parts[1]
                    )

                    filename = parts[
                        2
                    ]

                    self.save_correspondences(
                        sensor_id,
                        filename
                    )

                elif (
                    parts[0].lower()
                    in (
                        'quit',
                        'exit'
                    )
                ):

                    return

                else:

                    print(
                        'Unknown command.'
                    )

            except Exception as exc:

                print(
                    f'Command error: {exc}'
                )


def main(
    args=None
):

    rclpy.init(
        args=args
    )

    node = (
        ExtrinsicCalibrator()
    )

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == '__main__':
    main()