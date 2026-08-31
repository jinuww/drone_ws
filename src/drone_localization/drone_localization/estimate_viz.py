#!/usr/bin/env python3
"""EKF가 낸 위치 하나만 RViz2로 보여준다.

센서별 클라우드(/lidarN/points_world)는 쳐다보지 않는다. 이 노드가 그리는
것은 두 LiDAR가 하나로 합쳐진 결과 — `/drone/estimated_pose` 뿐이다.

  /drone/estimate_cloud    PointCloud2  추정 위치의 궤적. 점 하나가 프레임
                           하나이고, intensity 채널에 런 시작으로부터의 초를
                           넣어 두었다. RViz에서 Intensity로 색을 입히면
                           시간에 따라 색이 흐른다 — 어디서 멈췄고(점이 뭉침)
                           어디서 빨랐는지(점이 성김)가 그대로 보인다.
  /drone/estimate_markers  MarkerArray  현재 위치 구(球) + 좌표 글자.
                           구의 지름은 필터 공분산의 2σ다. 즉 **구가 커지면
                           필터가 자신 없어진 것**이고, 관측이 끊겨 코스팅
                           중일 때 눈에 띄게 부푼다.

트래커와 따로 도는 순수 시각화 노드다. 트래커에 그리기 부담을 지우지 않고,
돌아가는 시스템에 붙였다 뗐다 할 수 있다. GUI가 START를 쏘면 궤적을 비운다.

  ros2 launch drone_localization estimate_view.launch.py
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray


class EstimateVisualizer(Node):

    def __init__(self):
        super().__init__('estimate_visualizer')

        self.declare_parameter('pose_topic', '/drone/estimated_pose')
        # 10 Hz 기준 2000점 = 200초. 대회 임무 한 판이 다 들어간다.
        self.declare_parameter('trail_max_points', 2000)
        # 공분산이 작을 때도 구가 보이도록 하는 최소 지름.
        self.declare_parameter('marker_min_diameter', 0.20)
        self.declare_parameter('sigma_k', 2.0)
        self.declare_parameter('show_text', True)

        gp = lambda n: self.get_parameter(n).value
        self.pose_topic = str(gp('pose_topic'))
        self.trail_max = max(2, int(gp('trail_max_points')))
        self.min_diameter = float(gp('marker_min_diameter'))
        self.sigma_k = float(gp('sigma_k'))
        self.show_text = bool(gp('show_text'))

        self._trail = []          # [(x, y, z, seconds_since_start), ...]
        self._t0 = None
        self._frame = 'world'

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            PoseWithCovarianceStamped, self.pose_topic,
            self._pose_callback, reliable_qos)
        self.create_subscription(
            String, '/competition/mission_event', self._event_callback, 20)

        self._pub_cloud = self.create_publisher(
            PointCloud2, '/drone/estimate_cloud', reliable_qos)
        self._pub_markers = self.create_publisher(
            MarkerArray, '/drone/estimate_markers', reliable_qos)

        self.get_logger().info(
            f'Visualising {self.pose_topic} only (no per-sensor clouds); '
            f'trail={self.trail_max} points')

    # ------------------------------------------------------------------

    def _event_callback(self, msg):
        """새 런이 시작되면 궤적을 비운다. 지난 런이 겹쳐 보이면 안 된다."""
        event = msg.data.strip().upper()
        if event == 'START':
            self._trail.clear()
            self._t0 = None
            self.get_logger().info('Trail cleared on START')

    def _pose_callback(self, msg):
        stamp_s = float(msg.header.stamp.sec) + msg.header.stamp.nanosec * 1e-9
        if self._t0 is None:
            self._t0 = stamp_s
        if msg.header.frame_id:
            self._frame = msg.header.frame_id

        p = msg.pose.pose.position
        self._trail.append((p.x, p.y, p.z, stamp_s - self._t0))
        if len(self._trail) > self.trail_max:
            del self._trail[:len(self._trail) - self.trail_max]

        cov = np.asarray(msg.pose.covariance, dtype=np.float64).reshape(6, 6)
        sigma = np.sqrt(np.clip(np.diag(cov)[:3], 0.0, None))

        self._publish_cloud(msg.header)
        self._publish_markers(msg.header, np.array([p.x, p.y, p.z]), sigma)

    # ------------------------------------------------------------------

    def _publish_cloud(self, header):
        pts = np.asarray(self._trail, dtype=np.float32)
        h = Header()
        h.stamp = header.stamp
        h.frame_id = self._frame

        msg = PointCloud2()
        msg.header = h
        msg.height = 1
        msg.width = len(pts)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12,
                       datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * len(pts)
        msg.is_dense = True
        # 파이썬 루프로 채우면 궤적이 길어질수록 느려진다. 바이트로 한 번에.
        msg.data = np.ascontiguousarray(pts).tobytes()
        self._pub_cloud.publish(msg)

    def _publish_markers(self, header, pos, sigma):
        markers = MarkerArray()

        sphere = Marker()
        sphere.header.stamp = header.stamp
        sphere.header.frame_id = self._frame
        sphere.ns = 'ekf_estimate'
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(pos[0])
        sphere.pose.position.y = float(pos[1])
        sphere.pose.position.z = float(pos[2])
        sphere.pose.orientation.w = 1.0
        # 축마다 그 축의 2σ. 공분산이 방향에 따라 다르면 구가 아니라 타원체로
        # 보이는데, 그게 극좌표 EKF가 실제로 들고 있는 불확실성의 모양이다.
        sphere.scale.x = max(self.min_diameter, 2.0 * self.sigma_k * sigma[0])
        sphere.scale.y = max(self.min_diameter, 2.0 * self.sigma_k * sigma[1])
        sphere.scale.z = max(self.min_diameter, 2.0 * self.sigma_k * sigma[2])
        sphere.color.r, sphere.color.g, sphere.color.b = 1.0, 0.25, 0.1
        sphere.color.a = 0.55
        sphere.lifetime.sec = 1
        markers.markers.append(sphere)

        if self.show_text:
            text = Marker()
            text.header = sphere.header
            text.ns = 'ekf_estimate_text'
            text.id = 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(pos[0])
            text.pose.position.y = float(pos[1])
            text.pose.position.z = float(pos[2]) + 0.45
            text.pose.orientation.w = 1.0
            text.scale.z = 0.22
            text.color.r = text.color.g = text.color.b = 1.0
            text.color.a = 0.9
            rms = float(np.sqrt(np.mean(np.square(sigma))))
            text.text = (f'{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f} m\n'
                         f'sigma {rms * 100.0:.1f} cm')
            text.lifetime.sec = 1
            markers.markers.append(text)

        self._pub_markers.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = EstimateVisualizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
