#!/usr/bin/env python3

import csv
import json
import math
from pathlib import Path

import rclpy

from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import String


class OfflineScorer(Node):

    def __init__(self):

        super().__init__('competition_score')

        # ============================================================
        # Parameters
        # ============================================================

        self.declare_parameter('pose_topic', '/drone/estimated_pose')

        self.declare_parameter('event_topic', '/competition/mission_event')

        # ------------------------------------------------------------
        # Competition field grid
        # ------------------------------------------------------------

        self.declare_parameter('grid_x', [0.0, 1.0, 2.0, 3.0, 4.0,])

        self.declare_parameter('grid_y', [0.0, 1.0, 2.0, 3.0, 4.0,])

        # ------------------------------------------------------------
        # Home
        # ------------------------------------------------------------

        self.declare_parameter('home', [0.0, 0.0])

        # ------------------------------------------------------------
        # Marker locations
        # ------------------------------------------------------------

        self.declare_parameter('marker_1', [1.0, 1.0])

        self.declare_parameter('marker_2', [1.0, 3.0])

        self.declare_parameter('marker_3', [2.0, 3.0])

        self.declare_parameter('marker_4', [3.0, 3.0])

        # ------------------------------------------------------------
        # Competition timing
        # ------------------------------------------------------------

        self.declare_parameter('target_time', 180.0)

        self.declare_parameter('maximum_time', 600.0)

        # ------------------------------------------------------------
        # Scoring parameters
        # ------------------------------------------------------------

        self.declare_parameter('target_altitude', 1.5)

        self.declare_parameter('required_hover_time', 4.0)

        # Hover scoring:
        #   RMSE <= hover_full_score_rmse  -> 100 points
        #   RMSE >= hover_zero_score_rmse  -> 0 points
        #   Between them                   -> linear interpolation
        #
        # A hover is considered valid when the required hover duration
        # is satisfied and its RMSE does not exceed hover_zero_score_rmse.
        self.declare_parameter('hover_full_score_rmse', 0.20)
        self.declare_parameter('hover_zero_score_rmse', 0.50)

        self.declare_parameter('grid_error_limit', 0.75)

        self.declare_parameter('segment_error_limit', 0.75)

        # Altitude scoring:
        # The competition target is 2.0 m +/- 0.20 m. Therefore an
        # altitude RMSE up to 0.20 m receives full altitude credit.
        # The score then decreases linearly to zero at 0.40 m RMSE.
        self.declare_parameter('altitude_full_score_rmse', 0.20)
        self.declare_parameter('altitude_zero_score_rmse', 0.40)

        # ============================================================
        # Read parameters
        # ============================================================

        self.grid_x = list(self.get_parameter('grid_x').value)

        self.grid_y = list(self.get_parameter('grid_y').value)

        self.home = list(self.get_parameter('home').value)

        self.markers = {
            i: list(
                self.get_parameter(
                    f'marker_{i}'
                ).value
            )
            for i in range(1, 5)
        }

        self.target_time = float(self.get_parameter('target_time').value)

        self.maximum_time = float(self.get_parameter('maximum_time').value)

        self.target_altitude = float(self.get_parameter('target_altitude').value)

        self.required_hover_time = float(
            self.get_parameter('required_hover_time').value
        )

        self.hover_full_score_rmse = float(
            self.get_parameter('hover_full_score_rmse').value
        )

        self.hover_zero_score_rmse = float(
            self.get_parameter('hover_zero_score_rmse').value
        )

        self.grid_error_limit = float(
            self.get_parameter('grid_error_limit').value
        )

        self.segment_error_limit = float(
            self.get_parameter('segment_error_limit').value
        )

        self.altitude_full_score_rmse = float(
            self.get_parameter('altitude_full_score_rmse').value
        )

        self.altitude_zero_score_rmse = float(
            self.get_parameter('altitude_zero_score_rmse').value
        )

        # ============================================================
        # Run information
        #
        # These are supplied by competition_gui.
        # ============================================================

        self.team = None
        self.team_id = None
        self.run_id = None

        self.run_output_dir = None

        self.run_configured = False

        # ============================================================
        # Initialize scoring state
        # ============================================================

        self.reset_run_data()

        # ============================================================
        # QoS for run configuration
        #
        # TRANSIENT_LOCAL means that if the run configuration was
        # published just before this subscriber fully joined, it can
        # still receive the latest configuration.
        # ============================================================

        config_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # ============================================================
        # Subscriptions
        # ============================================================

        self.create_subscription(String,'/competition/run_config',
            self.run_config_callback, config_qos)

        self.create_subscription(PoseWithCovarianceStamped,
            self.get_parameter('pose_topic').value,
            self.pose_callback, qos_profile_sensor_data
        )

        self.create_subscription(String,
            self.get_parameter('event_topic').value,
            self.event_callback, 20
        )

        self.get_logger().info('Competition scorer ready.')

        self.get_logger().info('Waiting for /competition/run_config ...')

        self.get_logger().info('Pose input: /drone/estimated_pose')

    # ================================================================
    # Reset run
    # ================================================================

    def reset_run_data(self):

        self.state = 'IDLE'

        self.marker = 0
        self.segment = None

        self.started = False
        self.finished = False

        self.start_time = None

        self.rows = []

        self.grid_errors = []
        self.altitude_errors = []

        self.hover_errors = {
            i: []
            for i in range(1, 5)
        }

        self.segment_errors = {
            '4-3': [],
            '3-2': [],
            '2-1': [],
            '1-0': [],
        }

        self.detected = set()
        self.valid_hover = set()

        self.hover_start = {}

    # ================================================================
    # Run configuration from GUI
    # ================================================================

    def run_config_callback(self, msg):

        try:

            config = json.loads(msg.data)

            team_name = str(config['team_name'])

            team_id = str(config['team_id'])

            run_id = int(config['run_id'])

            output_dir = Path(config['output_dir']).expanduser()

        except (
            KeyError,
            ValueError,
            TypeError,
            json.JSONDecodeError
        ) as e:

            self.get_logger().error(
                f'Invalid run configuration: {e}'
            )

            return

        # ------------------------------------------------------------
        # Do not change run while current run is active
        # ------------------------------------------------------------

        if self.started:

            self.get_logger().warn(
                'Received new run configuration '
                'while scoring is active. Ignoring it.'
            )

            return

        try:

            output_dir.mkdir(parents=True, exist_ok=True)

        except Exception as e:

            self.get_logger().error(
                f'Cannot create output directory '
                f'{output_dir}: {e}'
            )

            return

        self.team = team_name
        self.team_id = team_id
        self.run_id = run_id

        self.run_output_dir = (output_dir)

        self.run_configured = True

        self.get_logger().info(
            f'Scorer configured: '
            f'{self.team} '
            f'({self.team_id}), '
            f'run {self.run_id}'
        )

        self.get_logger().info(
            f'Scorer output directory: '
            f'{self.run_output_dir}'
        )

    # ================================================================
    # Utility functions
    # ================================================================

    @staticmethod
    def rmse(values):

        if not values:
            return None

        return math.sqrt(
            sum(
                value * value
                for value in values
            )
            /
            len(values)
        )

    @staticmethod
    def score(error, limit):

        if error is None:
            return 0.0

        if not math.isfinite(error):
            return 0.0

        if limit <= 0.0:
            return 0.0

        return max(0.0, min(100.0,100.0*(1.0 - error / limit)))

    @staticmethod
    def tolerance_score(error, full_limit, zero_limit):
        """Score an error using an acceptable/full-credit tolerance band.

        100 points are awarded at or below full_limit.
        The score falls linearly from 100 to 0 between full_limit and
        zero_limit, and is 0 at or above zero_limit.
        """

        if error is None:
            return 0.0

        if not math.isfinite(error):
            return 0.0

        if full_limit < 0.0:
            return 0.0

        if zero_limit <= full_limit:
            return 0.0

        if error <= full_limit:
            return 100.0

        if error >= zero_limit:
            return 0.0

        return 100.0 * (
            zero_limit - error
        ) / (
            zero_limit - full_limit
        )

    @staticmethod
    def point_segment_distance(p, a, b):

        dx = (b[0] - a[0])

        dy = (b[1] - a[1])

        d2 = (dx * dx  +  dy * dy)

        if d2 <= 1e-12:

            return math.hypot(p[0] - a[0], p[1] - a[1])

        u = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / d2

        u = max(0.0, min(1.0, u))

        qx = (a[0] + u * dx)

        qy = (a[1] + u * dy)

        return math.hypot(p[0] - qx, p[1] - qy)

    def clock_time(self):

        return (self.get_clock().now().nanoseconds * 1e-9)

    def elapsed(self):

        if self.start_time is None:
            return 0.0

        return (self.clock_time() - self.start_time)

    # ================================================================
    # Pose callback
    # ================================================================

    def pose_callback(self, msg):

        if not self.started:
            return

        if self.finished:
            return

        x = float(msg.pose.pose.position.x)

        y = float(msg.pose.pose.position.y)

        z = float(msg.pose.pose.position.z)

        t = self.elapsed()

        grid_error = ''
        hover_error = ''
        segment_error = ''

        # ------------------------------------------------------------
        # Altitude error
        #
        # Score altitude only while the UAV is performing the mission
        # at the commanded competition altitude. TAKEOFF, IDLE and
        # post-mission landing transients are intentionally excluded.
        # ------------------------------------------------------------

        altitude_error = ''

        if self.state in ('SEARCH', 'HOVER', 'TRACE', 'RETURN'):

            altitude_error = abs(
                z - self.target_altitude
            )

            self.altitude_errors.append(
                altitude_error
            )

        # ------------------------------------------------------------
        # Grid-search accuracy
        # ------------------------------------------------------------

        if self.state == 'SEARCH':

            error_x = min(abs(x - gx)
                for gx in self.grid_x
            )

            error_y = min(abs(y - gy)
                for gy in self.grid_y
            )

            grid_error = min(error_x, error_y)

            self.grid_errors.append(grid_error)

        # ------------------------------------------------------------
        # Hover accuracy
        # ------------------------------------------------------------

        elif (self.state == 'HOVER' and self.marker in self.markers):

            mx, my = (self.markers[self.marker])

            hover_error = (math.hypot(x - mx, y - my))

            self.hover_errors[self.marker].append(hover_error)

        # ------------------------------------------------------------
        # Segment tracing / return home
        # ------------------------------------------------------------

        elif (self.state in ('TRACE', 'RETURN') and self.segment is not None):

            a_id, b_id = (self.segment)

            a = (self.markers[a_id])

            if b_id == 0:

                b = self.home

            else:

                b = (self.markers[b_id])

            segment_error = (
                self.point_segment_distance((x, y), a, b)
            )

            key = (
                f'{a_id}-{b_id}'
            )

            if (
                key
                in
                self.segment_errors
            ):

                self.segment_errors[
                    key
                ].append(
                    segment_error
                )

        # ------------------------------------------------------------
        # Save scoring trajectory in memory
        # ------------------------------------------------------------

        self.rows.append([t, x, y, z,
            self.state,
            self.marker,
            grid_error,
            hover_error,
            segment_error,
            altitude_error,
        ])

    # ================================================================
    # Competition mission events
    # ================================================================

    def event_callback(self, msg):

        event = (msg.data.strip())

        now = (self.clock_time())

        self.get_logger().info(
            f'Mission event: {event}'
        )

        # ------------------------------------------------------------
        # START
        # ------------------------------------------------------------

        if event == 'START':

            if not self.run_configured:

                self.get_logger().error(
                    'START received before '
                    '/competition/run_config. '
                    'Cannot begin scoring.'
                )

                return

            # Clear any data remaining
            # from a previous run.
            self.reset_run_data()

            self.started = True
            self.finished = False

            self.start_time = now

            self.state = 'TAKEOFF'

            self.get_logger().info(
                f'Scoring STARTED: '
                f'{self.team} '
                f'({self.team_id}), '
                f'run {self.run_id}'
            )

            return

        # ------------------------------------------------------------
        # Ignore mission events until START
        # ------------------------------------------------------------

        if not self.started:
            return

        if self.finished:
            return

        parts = (
            event.split(':')
        )

        # ------------------------------------------------------------
        # Search
        # ------------------------------------------------------------

        if event in ('SEARCH_START', 'SEARCH_RESUME'):

            self.state = 'SEARCH'
            self.marker = 0
            self.segment = None

        # ------------------------------------------------------------
        # Marker detected
        # ------------------------------------------------------------

        elif (len(parts) == 2 and parts[0] == 'MARKER_DETECTED'):

            try:

                marker = int(parts[1])

                if marker in self.markers:

                    self.detected.add(marker)

                    self.get_logger().info(
                        f'Marker {marker} detected.'
                    )

            except ValueError:

                self.get_logger().warn(
                    f'Invalid marker event: '
                    f'{event}'
                )

        # ------------------------------------------------------------
        # Hover start
        # ------------------------------------------------------------

        elif (len(parts) == 2 and parts[0] == 'HOVER_START'):

            try:

                marker = int(parts[1])

            except ValueError:

                self.get_logger().warn(
                    f'Invalid HOVER_START: '
                    f'{event}'
                )

                return

            if marker not in self.markers:

                self.get_logger().warn(
                    f'Unknown marker '
                    f'{marker}'
                )

                return

            self.marker = marker

            self.hover_start[marker] = now

            self.hover_errors[marker].clear()

            self.state = 'HOVER'

        # ------------------------------------------------------------
        # Hover end
        # ------------------------------------------------------------

        elif (len(parts) == 2 and parts[0] == 'HOVER_END'):

            try:

                marker = int(parts[1])

            except ValueError:

                self.get_logger().warn(
                    f'Invalid HOVER_END: '
                    f'{event}'
                )

                return

            if marker not in self.markers:
                return

            start = (self.hover_start.get(marker))

            if start is None:

                self.get_logger().warn(
                    f'HOVER_END:{marker} '
                    f'received without '
                    f'HOVER_START.'
                )

            else:

                duration = (now - start)

                hover_rmse = (self.rmse(self.hover_errors[marker]))

                # A hover is valid if the required duration is met and
                # its RMSE remains within the maximum scoring tolerance.
                #
                # Accuracy is NOT converted abruptly to zero at a smaller
                # threshold. Instead, the final hover score uses the
                # continuous tolerance_score() function.
                if (
                    duration >= self.required_hover_time
                    and hover_rmse is not None
                    and hover_rmse <= self.hover_zero_score_rmse
                ):

                    self.valid_hover.add(marker)

                    self.get_logger().info(
                        f'Marker {marker}: '
                        f'valid hover, '
                        f'duration={duration:.2f}s, '
                        f'RMSE={hover_rmse:.3f}m'
                    )

                else:

                    self.get_logger().warn(
                        f'Marker {marker}: invalid hover, '
                        f'duration={duration:.2f}s, '
                        f'RMSE='
                        f'{hover_rmse if hover_rmse is not None else "None"}'
                    )

            self.state = 'SEARCH'

            self.marker = 0

        # ------------------------------------------------------------
        # Trace segment
        # ------------------------------------------------------------

        elif (len(parts) == 3 and parts[0] == 'TRACE_START'):

            try:

                a = int(parts[1])

                b = int(parts[2])

            except ValueError:

                self.get_logger().warn(
                    f'Invalid TRACE_START: '
                    f'{event}'
                )

                return

            key = (
                f'{a}-{b}'
            )

            if (
                key
                not in
                self.segment_errors
            ):

                self.get_logger().warn(
                    f'Unsupported segment: '
                    f'{key}'
                )

                return

            self.segment = (a, b)

            self.state = 'TRACE'

        # ------------------------------------------------------------
        # Return home
        # ------------------------------------------------------------

        elif event == 'RETURN_HOME':

            self.segment = (1, 0)

            self.state = 'RETURN'

        # ------------------------------------------------------------
        # FINISH
        # ------------------------------------------------------------

        elif event == 'FINISH':

            self.finish(aborted=False)

        # ------------------------------------------------------------
        # ABORT
        # ------------------------------------------------------------

        elif event == 'ABORT':

            self.finish(aborted=True)

    # ================================================================
    # Finish scoring
    # ================================================================

    def finish(self, aborted=False):

        if not self.started:
            return

        if self.finished:
            return

        self.finished = True

        completion_time = (self.elapsed())

        # ------------------------------------------------------------
        # RMSE calculations
        # ------------------------------------------------------------

        grid_rmse = (self.rmse(self.grid_errors))

        altitude_rmse = (self.rmse(self.altitude_errors))

        all_segment_errors = [
            value
            for values
            in self.segment_errors.values()
            for value
            in values
        ]

        segment_rmse = (self.rmse(all_segment_errors))

        # ------------------------------------------------------------
        # Hover score
        # ------------------------------------------------------------

        hover_rmse_by_marker = {}

        hover_scores = []

        for i in range(1, 5):

            hover_rmse = (self.rmse(self.hover_errors[i]))

            hover_rmse_by_marker[str(i)] = hover_rmse

            if i in self.valid_hover:

                hover_scores.append(
                    self.tolerance_score(
                        hover_rmse,
                        self.hover_full_score_rmse,
                        self.hover_zero_score_rmse
                    )
                )

            else:

                hover_scores.append(0.0)

        # ------------------------------------------------------------
        # Time score
        # ------------------------------------------------------------

        if (completion_time <= self.target_time):

            time_score = 100.0

        elif (completion_time >= self.maximum_time):

            time_score = 0.0

        else:

            time_score = (100.0 * (self.maximum_time - completion_time)/
                (self.maximum_time - self.target_time))

        # ------------------------------------------------------------
        # Component scores
        # ------------------------------------------------------------

        scores = {

            'grid':
                self.score(
                    grid_rmse,
                    self.grid_error_limit
                ),

            'hover':
                sum(
                    hover_scores
                ) / 4.0,

            'segments':
                self.score(
                    segment_rmse,
                    self.segment_error_limit
                ),

            'altitude':
                self.tolerance_score(
                    altitude_rmse,
                    self.altitude_full_score_rmse,
                    self.altitude_zero_score_rmse
                ),

            'time':
                time_score,
        }

        # ------------------------------------------------------------
        # Penalties
        # ------------------------------------------------------------

        missing_markers = (4 - len(self.detected))

        invalid_hovers = len(self.detected - self.valid_hover)

        penalty = (10.0 * missing_markers)

        penalty += (5.0 * invalid_hovers)

        # ------------------------------------------------------------
        # Final score
        # ------------------------------------------------------------

        final_score = (0.20 * scores['grid'] + 0.25 * scores['hover']

            + 0.15 * scores['segments'] + 0.25 * scores['altitude']

            + 0.15 * scores['time'] - penalty)

        if aborted:

            final_score = 0.0

        final_score = max(0.0, min(100.0, final_score))

        # ------------------------------------------------------------
        # Individual segment RMSE
        # ------------------------------------------------------------

        segment_rmse_by_segment = {

            key:
                self.rmse(values)

            for (key, values) in self.segment_errors.items()
        }

        # ------------------------------------------------------------
        # Result dictionary
        # ------------------------------------------------------------

        result = {

            'team_name': self.team,

            'team_id': self.team_id,

            'run_id': self.run_id,

            'status':
                (
                    'ABORTED'
                    if aborted
                    else
                    'FINISHED'
                ),

            'completion_time_s': completion_time,

            'detected_markers': sorted(self.detected),

            'valid_hover_markers': sorted(self.valid_hover),

            'grid_rmse_m': grid_rmse,

            'segment_rmse_m': segment_rmse,

            'altitude_rmse_m': altitude_rmse,

            'hover_rmse_m': hover_rmse_by_marker,

            'segment_rmse_individual_m': segment_rmse_by_segment,

            'scoring_thresholds': {
                'hover_full_score_rmse_m':
                    self.hover_full_score_rmse,
                'hover_zero_score_rmse_m':
                    self.hover_zero_score_rmse,
                'altitude_full_score_rmse_m':
                    self.altitude_full_score_rmse,
                'altitude_zero_score_rmse_m':
                    self.altitude_zero_score_rmse,
            },

            'scores': scores,

            'penalty': penalty,

            'final_score': final_score,
        }

        # ------------------------------------------------------------
        # Output directory
        # ------------------------------------------------------------

        if (self.run_output_dir is None):

            self.get_logger().error(
                'No run output directory '
                'was supplied by GUI. '
                'Result cannot be saved.'
            )

            self.started = False

            return

        folder = (self.run_output_dir)

        try:

            folder.mkdir(parents=True, exist_ok=True)

        except Exception as e:

            self.get_logger().error(
                f'Cannot create output '
                f'directory {folder}: {e}'
            )

            self.started = False

            return

        # ------------------------------------------------------------
        # Scoring trajectory
        # ------------------------------------------------------------

        trajectory_file = (folder / 'scoring_trajectory.csv')

        try:

            with open(trajectory_file, 'w', newline='') as file:

                writer = (csv.writer(file))

                writer.writerow([
                    'time_s',
                    'x_m',
                    'y_m',
                    'z_m',
                    'state',
                    'marker',
                    'grid_error_m',
                    'hover_error_m',
                    'segment_error_m',
                    'altitude_error_m',
                ])

                writer.writerows(self.rows)

        except Exception as e:

            self.get_logger().error(
                f'Failed to save '
                f'scoring trajectory: {e}'
            )

        # ------------------------------------------------------------
        # JSON result
        # ------------------------------------------------------------

        result_file = (folder / 'result.json')

        try:

            with open(result_file, 'w') as file:

                json.dump(result, file, indent=2, allow_nan=False)

        except ValueError:

            # JSON does not accept NaN/Infinity when allow_nan=False.
            # rmse() already returns None for empty data, so this should
            # normally never happen.
            self.get_logger().error(
                'Invalid numeric value '
                'while writing result.json'
            )

        except Exception as e:

            self.get_logger().error(
                f'Failed to save '
                f'result.json: {e}'
            )

            self.started = False

            return

        # ------------------------------------------------------------
        # Terminal summary
        # ------------------------------------------------------------

        self.get_logger().info(
            '========================================'
        )

        self.get_logger().info(
            f'Team: '
            f'{self.team} '
            f'({self.team_id})'
        )

        self.get_logger().info(
            f'Run: '
            f'{self.run_id}'
        )

        self.get_logger().info(
            f'Completion time: '
            f'{completion_time:.2f} s'
        )

        self.get_logger().info(
            f'Final score: '
            f'{final_score:.2f} / 100'
        )

        self.get_logger().info(
            f'Result saved: '
            f'{result_file}'
        )

        self.get_logger().info(
            '========================================'
        )

        self.started = False

        self.state = ('ABORTED'
            if aborted
            else
            'FINISHED'
        )


# ====================================================================
# Main
# ====================================================================

def main(args=None):

    rclpy.init(args=args)

    node = (OfflineScorer())

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == '__main__':
    main()