#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node


def generate_launch_description():

    # ============================================================
    # Package paths
    # ============================================================

    drone_localization_share = get_package_share_directory(
        'drone_localization'
    )

    dual_ouster_share = get_package_share_directory(
        'dual_ouster_driver'
    )

    # ============================================================
    # Launch arguments
    # ============================================================

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value='lidar_localization_os1.yaml',
        description=(
            'Parameter file under drone_localization/params/.'
        ),
    )

    learn_background_arg = DeclareLaunchArgument(
        'learn_background',
        default_value='false',
        description=(
            'true: collect and save static background. '
            'false: normal drone tracking.'
        ),
    )

    use_background_arg = DeclareLaunchArgument(
        'use_background',
        default_value='true',
        description=(
            'true: use saved background during normal tracking.'
        ),
    )

    # Optional override for the saved .npz file.
    background_file_arg = DeclareLaunchArgument(
        'background_file',
        default_value=(
            '/home/drone/competition_results/'
            'calibration/dual_ouster_background.npz'
        ),
        description='Path of the saved static-background model.',
    )

    # ============================================================
    # Tracker YAML
    # ============================================================

    params_path = PathJoinSubstitution([
        drone_localization_share,
        'params',
        LaunchConfiguration('params_file'),
    ])

    # ============================================================
    # Physical Ouster OS1-128 drivers
    #
    # Expected outputs:
    #
    # /lidar1/points
    # /lidar2/points
    # ============================================================

    dual_ouster_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                dual_ouster_share,
                'launch',
                'dual_os1.launch.py',
            ])
        )
    )

    # ============================================================
    # Drone localization / background-learning node
    # ============================================================

    tracker = Node(
        package='drone_localization',
        executable='lidar_drone_tracker',
        name='lidar_drone_tracker',
        output='screen',

        # First load all normal settings from YAML.
        # Then override the three parameters controlled from
        # the launch command.
        parameters=[
            params_path,
            {
                'learn_background': LaunchConfiguration(
                    'learn_background'
                ),

                'use_background': LaunchConfiguration(
                    'use_background'
                ),

                'background_file': LaunchConfiguration(
                    'background_file'
                ),
            },
        ],

        respawn=False,
    )

    # ============================================================
    # Launch description
    # ============================================================

    return LaunchDescription([

        params_file_arg,

        learn_background_arg,
        use_background_arg,
        background_file_arg,

        dual_ouster_driver,

        tracker,
    ])