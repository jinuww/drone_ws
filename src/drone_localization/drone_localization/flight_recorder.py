#!/usr/bin/env python3
"""
Flight recorder: logs the actual flown path and the LiDAR-estimated path
to CSV during a flight, so you can plot them afterwards with plot_flight.

Subscribes:
  /fmu/out/vehicle_local_position_v1  (PX4 ground-truth, NED)  → actual path
  /drone/estimated_pose               (LiDAR localization, world ENU) → estimate

Both are written in the Gazebo world frame (ENU) so they overlay directly
on the planned trajectory. PX4 local NED → world ENU:
  world_x = ned_y (east),  world_y = ned_x (north),  world_z = -ned_z (up)
(inverse of competition_mission's world_to_ned).

Rows are flushed as they arrive, so Ctrl+C (or a crash) still leaves a
complete CSV. Files are overwritten at each run.

Usage:
  ros2 run drone_localization flight_recorder
  # ... fly the mission (competition_mission) ...
  # Ctrl+C when done, then:
  ros2 run drone_localization plot_flight
"""

import csv
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
)
from geometry_msgs.msg import PoseWithCovarianceStamped

try:
    from px4_msgs.msg import VehicleLocalPosition
    HAS_PX4 = True
except ImportError:
    HAS_PX4 = False


class FlightRecorder(Node):
    def __init__(self):
        super().__init__('flight_recorder')

        default_dir = os.path.expanduser('~/drone_project/flight_logs')
        self.declare_parameter('output_dir', default_dir)
        self.declare_parameter(
            'actual_topic', '/fmu/out/vehicle_local_position_v1')
        self.declare_parameter('estimate_topic', '/drone/estimated_pose')
        self.out_dir = self.get_parameter('output_dir').value
        os.makedirs(self.out_dir, exist_ok=True)

        self._actual_path = os.path.join(self.out_dir, 'actual_path.csv')
        self._est_path = os.path.join(self.out_dir, 'estimated_path.csv')

        # Open both files and write headers immediately
        self._actual_f = open(self._actual_path, 'w', newline='')
        self._actual_w = csv.writer(self._actual_f)
        self._actual_w.writerow(['t_sec', 'world_x', 'world_y', 'world_z'])

        self._est_f = open(self._est_path, 'w', newline='')
        self._est_w = csv.writer(self._est_f)
        self._est_w.writerow(['t_sec', 'world_x', 'world_y', 'world_z'])

        self._n_actual = 0
        self._n_est = 0
        # Single shared wall clock for BOTH streams. Using each message's own
        # embedded timestamp is wrong here: PX4 uses boot-microseconds and the
        # estimate uses ROS time — different clocks. Stamping both at reception
        # with one monotonic clock keeps them directly comparable (the <0.1s
        # LiDAR-processing latency is negligible for this analysis).
        self._t0 = time.monotonic()

        # PX4 publishes best-effort; estimate is reliable
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        est_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        if HAS_PX4:
            self.create_subscription(
                VehicleLocalPosition,
                self.get_parameter('actual_topic').value,
                self._actual_cb, px4_qos)
        else:
            self.get_logger().warn(
                'px4_msgs not found — actual (PX4) path will NOT be recorded. '
                'Only /drone/estimated_pose is logged.')

        self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter('estimate_topic').value,
            self._est_cb, est_qos)

        self.get_logger().info(
            f'Recording flight to:\n  {self._actual_path}\n  {self._est_path}\n'
            'Fly the mission, then Ctrl+C and run: ros2 run drone_localization plot_flight')

    def _actual_cb(self, msg: 'VehicleLocalPosition'):
        if not (msg.xy_valid and msg.z_valid):
            return
        t = time.monotonic() - self._t0        # shared clock
        # NED → world ENU
        world_x = msg.y
        world_y = msg.x
        world_z = -msg.z
        self._actual_w.writerow(
            [f'{t:.4f}', f'{world_x:.4f}', f'{world_y:.4f}', f'{world_z:.4f}'])
        self._actual_f.flush()
        self._n_actual += 1

    def _est_cb(self, msg: PoseWithCovarianceStamped):
        t = time.monotonic() - self._t0        # same shared clock
        p = msg.pose.pose.position
        self._est_w.writerow(
            [f'{t:.4f}', f'{p.x:.4f}', f'{p.y:.4f}', f'{p.z:.4f}'])
        self._est_f.flush()
        self._n_est += 1

    def close(self):
        try:
            self._actual_f.close()
            self._est_f.close()
        except Exception:
            pass
        self.get_logger().info(
            f'Saved {self._n_actual} actual + {self._n_est} estimated samples.')


def main():
    rclpy.init()
    node = FlightRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()
