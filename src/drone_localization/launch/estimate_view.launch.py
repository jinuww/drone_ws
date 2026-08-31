#!/usr/bin/env python3

"""EKF 추정 위치 하나만 RViz2로 띄운다.

측위 시스템과 **따로** 도는 시각화 전용 런치다. 트래커가 이미 돌고 있는
상태(GUI로 띄웠든 lidar_localization.launch.py로 띄웠든)에 붙였다 뗐다 할 수
있다. 센서별 클라우드는 구독하지 않으므로 트래커에 부담을 주지 않는다.

    ros2 launch drone_localization estimate_view.launch.py

RViz 없이 토픽만 만들고 싶으면(원격에서 rviz를 따로 띄울 때):

    ros2 launch drone_localization estimate_view.launch.py rviz:=false

프레임에 대하여: 트래커는 TF를 쓰지 않고 좌표변환을 코드 안에서 끝낸 뒤
`world` 프레임으로 발행한다. RViz는 Fixed Frame이 TF 트리에 있어야 그리므로,
map -> world 항등 변환 하나를 같이 띄운다. 이미 다른 곳에서 TF를 내고 있다면
static_tf:=false 로 끄면 된다.
"""

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():

    share = get_package_share_directory('drone_localization')

    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='RViz2를 같이 띄울지')

    config_arg = DeclareLaunchArgument(
        'rviz_config', default_value='estimate_view.rviz',
        description='drone_localization/rviz/ 아래의 설정 파일')

    static_tf_arg = DeclareLaunchArgument(
        'static_tf', default_value='true',
        description='map -> world 항등 변환을 발행할지 (RViz Fixed Frame용)')

    pose_topic_arg = DeclareLaunchArgument(
        'pose_topic', default_value='/drone/estimated_pose',
        description='시각화할 EKF 추정 토픽')

    trail_arg = DeclareLaunchArgument(
        'trail_max_points', default_value='2000',
        description='궤적으로 남길 최대 점 개수 (10Hz에서 2000 = 200초)')

    rviz_config_path = PathJoinSubstitution([
        share, 'rviz', LaunchConfiguration('rviz_config'),
    ])

    visualizer = Node(
        package='drone_localization',
        executable='estimate_viz',
        name='estimate_visualizer',
        output='screen',
        parameters=[{
            'pose_topic': LaunchConfiguration('pose_topic'),
            'trail_max_points': LaunchConfiguration('trail_max_points'),
        }],
    )

    # 트래커가 TF를 안 내므로 world 프레임이 트리에 존재하지 않는다.
    # 이게 없으면 RViz가 "Fixed Frame [world] does not exist" 만 띄우고
    # 아무것도 안 그린다.
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_world',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'map', '--child-frame-id', 'world'],
        condition=IfCondition(LaunchConfiguration('static_tf')),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_estimate_view',
        arguments=['-d', rviz_config_path],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        rviz_arg,
        config_arg,
        static_tf_arg,
        pose_topic_arg,
        trail_arg,
        static_tf,
        visualizer,
        rviz,
    ])
