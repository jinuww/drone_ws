#!/usr/bin/env python3

import json
import os
import re
import signal
import subprocess
import threading
from pathlib import Path

import tkinter as tk
from tkinter import messagebox

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
)

from std_msgs.msg import String


class CompetitionGUI(Node):

    def __init__(self):

        super().__init__('competition_gui')

        # ======================================================
        # Commands controlled by GUI
        # ======================================================

        # LiDAR localization system
        self.tracker_command = [
            'ros2',
            'launch',
            'drone_localization',
            'lidar_localization.launch.py',
            'params_file:=lidar_localization_os1.yaml'
        ]

        # Offline scoring node
        self.scorer_command = [
            'ros2',
            'run',
            'competition_score',
            'compet_score'
        ]

        # Expected ROS node names.
        # Used to make sure the processes are ready before START.
        self.tracker_node_name = 'lidar_drone_tracker'
        self.scorer_node_name = 'competition_score'

        # ======================================================
        # Process handles
        # ======================================================

        self.tracker_process = None
        self.scorer_process = None

        self.start_pending = False
        self.shutdown_pending = False

        # ======================================================
        # ROS publishers
        # ======================================================

        # Keep the most recently published run configuration.
        config_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.config_pub = self.create_publisher(
            String,
            '/competition/run_config',
            config_qos
        )

        self.event_pub = self.create_publisher(
            String,
            '/competition/mission_event',
            10
        )

        # ======================================================
        # Competition data
        # ======================================================

        self.base_dir = (
            Path.home() /
            'competition_results'
        )

        self.base_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.active = False

        self.current_folder = None
        self.current_team = None
        self.current_team_id = None
        self.current_run_id = None

        # ======================================================
        # GUI
        # ======================================================

        self.build_gui()


    # ==========================================================
    # GUI construction
    # ==========================================================

    def build_gui(self):

        self.root = tk.Tk()

        self.root.title(
            'UAV Competition Management System'
        )

        self.root.geometry(
            '620x520'
        )

        # ------------------------------------------------------
        # Title
        # ------------------------------------------------------

        title = tk.Label(
            self.root,
            text='Autonomous UAV Competition',
            font=('Arial', 20, 'bold')
        )

        title.pack(
            pady=18
        )

        # ------------------------------------------------------
        # Team information
        # ------------------------------------------------------

        form = tk.Frame(
            self.root
        )

        form.pack(
            pady=10
        )

        tk.Label(
            form,
            text='Team Name:',
            width=16,
            anchor='e'
        ).grid(
            row=0,
            column=0,
            padx=5,
            pady=8
        )

        self.team_name_entry = tk.Entry(
            form,
            width=28
        )

        self.team_name_entry.grid(
            row=0,
            column=1
        )

        tk.Label(
            form,
            text='Team ID:',
            width=16,
            anchor='e'
        ).grid(
            row=1,
            column=0,
            padx=5,
            pady=8
        )

        self.team_id_entry = tk.Entry(
            form,
            width=28
        )

        self.team_id_entry.grid(
            row=1,
            column=1
        )

        tk.Label(
            form,
            text='Run Number:',
            width=16,
            anchor='e'
        ).grid(
            row=2,
            column=0,
            padx=5,
            pady=8
        )

        self.run_entry = tk.Entry(
            form,
            width=28
        )

        self.run_entry.insert(
            0,
            '1'
        )

        self.run_entry.grid(
            row=2,
            column=1
        )

        # ------------------------------------------------------
        # Buttons
        # ------------------------------------------------------

        buttons = tk.Frame(
            self.root
        )

        buttons.pack(
            pady=20
        )

        self.start_button = tk.Button(
            buttons,
            text='START RUN',
            width=17,
            height=2,
            command=self.start_run
        )

        self.start_button.grid(
            row=0,
            column=0,
            padx=8
        )

        self.finish_button = tk.Button(
            buttons,
            text='FINISH RUN',
            width=17,
            height=2,
            state=tk.DISABLED,
            command=self.finish_run
        )

        self.finish_button.grid(
            row=0,
            column=1,
            padx=8
        )

        self.abort_button = tk.Button(
            buttons,
            text='ABORT RUN',
            width=17,
            height=2,
            state=tk.DISABLED,
            command=self.abort_run
        )

        self.abort_button.grid(
            row=0,
            column=2,
            padx=8
        )

        # ------------------------------------------------------
        # Status
        # ------------------------------------------------------

        tk.Label(
            self.root,
            text='System Status',
            font=('Arial', 13, 'bold')
        ).pack(
            pady=(15, 5)
        )

        self.status = tk.Label(
            self.root,
            text='Waiting for team registration',
            font=('Arial', 12),
            justify='left',
            wraplength=560
        )

        self.status.pack(
            pady=10
        )

        # ------------------------------------------------------
        # Closing GUI
        # ------------------------------------------------------

        self.root.protocol(
            'WM_DELETE_WINDOW',
            self.on_gui_close
        )


    # ==========================================================
    # Utility
    # ==========================================================

    @staticmethod
    def safe_name(name):

        name = name.strip()

        return re.sub(
            r'[^A-Za-z0-9_-]+',
            '_',
            name
        )


    # ==========================================================
    # START RUN
    # ==========================================================

    def start_run(self):

        if self.active or self.start_pending:

            messagebox.showwarning(
                'Run active',
                'A competition run is already active.'
            )

            return

        # ------------------------------------------------------
        # Team information
        # ------------------------------------------------------

        team_name = (
            self.team_name_entry
            .get()
            .strip()
        )

        team_id = (
            self.team_id_entry
            .get()
            .strip()
        )

        run_text = (
            self.run_entry
            .get()
            .strip()
        )

        if not team_name:

            messagebox.showerror(
                'Missing information',
                'Enter the team name.'
            )

            return

        if not team_id:

            messagebox.showerror(
                'Missing information',
                'Enter the team ID.'
            )

            return

        try:

            run_id = int(
                run_text
            )

        except ValueError:

            messagebox.showerror(
                'Invalid run',
                'Run number must be an integer.'
            )

            return

        # ------------------------------------------------------
        # Directory
        # ------------------------------------------------------

        safe_team = self.safe_name(
            team_name
        )

        safe_id = self.safe_name(
            team_id
        )

        team_folder = (
            self.base_dir /
            f'{safe_team}_{safe_id}'
        )

        run_folder = (
            team_folder /
            f'run_{run_id:02d}'
        )

        if run_folder.exists():

            answer = messagebox.askyesno(
                'Run already exists',
                (
                    f'{run_folder}\n\n'
                    'This run already exists.\n'
                    'Continue and overwrite its data?'
                )
            )

            if not answer:
                return

        run_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        self.current_folder = run_folder

        self.current_team = team_name
        self.current_team_id = team_id
        self.current_run_id = run_id

        # ------------------------------------------------------
        # Launch nodes
        # ------------------------------------------------------

        self.status.config(
            text=(
                f'STARTING SYSTEM...\n\n'
                f'Team: {team_name}\n'
                f'Team ID: {team_id}\n'
                f'Run: {run_id}\n\n'
                'Launching LiDAR localization...\n'
                'Launching performance scorer...'
            )
        )

        self.start_button.config(
            state=tk.DISABLED
        )

        try:

            self.launch_competition_nodes()

        except Exception as e:

            self.get_logger().error(
                f'Failed to launch nodes: {e}'
            )

            messagebox.showerror(
                'Launch error',
                str(e)
            )

            self.stop_competition_nodes()

            self.start_button.config(
                state=tk.NORMAL
            )

            return

        self.start_pending = True

        # ------------------------------------------------------
        # Wait until ROS nodes appear
        # ------------------------------------------------------

        self.root.after(
            300,
            self.check_nodes_ready
        )


    # ==========================================================
    # Launch ROS processes
    # ==========================================================

    def launch_competition_nodes(self):

        # Prevent old handles
        self.stop_competition_nodes()

        # ------------------------------------------------------
        # LiDAR tracker / launch system
        # ------------------------------------------------------

        self.get_logger().info(
            'Launching LiDAR localization...'
        )

        self.tracker_process = subprocess.Popen(
            self.tracker_command,

            # Creates a process group so the entire ros2 launch
            # tree can be stopped later.
            start_new_session=True
        )

        # ------------------------------------------------------
        # Scorer
        # ------------------------------------------------------

        self.get_logger().info(
            'Launching offline scorer...'
        )

        self.scorer_process = subprocess.Popen(
            self.scorer_command,
            start_new_session=True
        )


    # ==========================================================
    # Wait for nodes
    # ==========================================================

    def check_nodes_ready(self):

        if not self.start_pending:
            return

        # Check for early process failure
        if (
            self.tracker_process is None or
            self.tracker_process.poll() is not None
        ):

            self.start_failed(
                'LiDAR localization process stopped unexpectedly.'
            )

            return

        if (
            self.scorer_process is None or
            self.scorer_process.poll() is not None
        ):

            self.start_failed(
                'Offline scorer process stopped unexpectedly.'
            )

            return

        # ------------------------------------------------------
        # Read ROS graph
        # ------------------------------------------------------

        nodes = set(
            self.get_node_names()
        )

        tracker_ready = (
            self.tracker_node_name
            in nodes
        )

        scorer_ready = (
            self.scorer_node_name
            in nodes
        )

        if (
            tracker_ready
            and
            scorer_ready
        ):

            self.get_logger().info(
                'Tracker and scorer are ready.'
            )

            self.publish_run_start()

            return

        self.status.config(
            text=(
                'STARTING SYSTEM...\n\n'
                f'LiDAR tracker: '
                f'{"READY" if tracker_ready else "starting"}\n'
                f'Scorer: '
                f'{"READY" if scorer_ready else "starting"}'
            )
        )

        # Check again
        self.root.after(
            300,
            self.check_nodes_ready
        )


    def start_failed(self, reason):

        self.start_pending = False

        self.stop_competition_nodes()

        self.start_button.config(
            state=tk.NORMAL
        )

        self.status.config(
            text='START FAILED'
        )

        messagebox.showerror(
            'Start failed',
            reason
        )


    # ==========================================================
    # Publish configuration + START
    # ==========================================================

    def publish_run_start(self):

        # ------------------------------------------------------
        # Run configuration
        # ------------------------------------------------------

        config = {

            'team_name':
                self.current_team,

            'team_id':
                self.current_team_id,

            'run_id':
                self.current_run_id,

            'output_dir':
                str(
                    self.current_folder
                )
        }

        msg = String()

        msg.data = json.dumps(
            config
        )

        self.config_pub.publish(
            msg
        )

        self.get_logger().info(
            f'Published run configuration: {msg.data}'
        )

        # ------------------------------------------------------
        # START
        # ------------------------------------------------------

        event = String()

        event.data = 'START'

        self.event_pub.publish(
            event
        )

        self.get_logger().info(
            'Published START'
        )

        # ------------------------------------------------------
        # State
        # ------------------------------------------------------

        self.start_pending = False
        self.active = True

        self.finish_button.config(
            state=tk.NORMAL
        )

        self.abort_button.config(
            state=tk.NORMAL
        )

        self.status.config(
            text=(
                'RUNNING\n\n'
                f'Team: {self.current_team}\n'
                f'Team ID: {self.current_team_id}\n'
                f'Run: {self.current_run_id}\n\n'
                f'Data directory:\n'
                f'{self.current_folder}'
            )
        )


    # ==========================================================
    # FINISH
    # ==========================================================

    def finish_run(self):

        if not self.active:
            return

        self.active = False
        self.shutdown_pending = True

        self.finish_button.config(
            state=tk.DISABLED
        )

        self.abort_button.config(
            state=tk.DISABLED
        )

        event = String()
        event.data = 'FINISH'

        self.event_pub.publish(
            event
        )

        self.get_logger().info(
            'Published FINISH'
        )

        self.status.config(
            text=(
                'FINISHING RUN...\n\n'
                'Saving localization data and '
                'calculating final score...'
            )
        )

        # Wait for scorer output before stopping nodes
        self.root.after(
            200,
            self.check_result_saved
        )


    # ==========================================================
    # Wait until result.json is saved
    # ==========================================================

    def check_result_saved(self):

        if not self.shutdown_pending:
            return

        result_file = (
            self.current_folder /
            'result.json'
        )

        if result_file.exists():

            self.get_logger().info(
                f'Score result saved: {result_file}'
            )

            self.finish_shutdown()

            return

        # Check if scorer crashed
        if (
            self.scorer_process is not None
            and
            self.scorer_process.poll()
            is not None
        ):

            self.get_logger().warn(
                'Scorer exited before result.json appeared.'
            )

            self.finish_shutdown()

            return

        self.root.after(
            200,
            self.check_result_saved
        )


    # ==========================================================
    # ABORT
    # ==========================================================

    def abort_run(self):

        if (
            not self.active
            and not self.start_pending
        ):

            return

        event = String()
        event.data = 'ABORT'

        self.event_pub.publish(
            event
        )

        self.get_logger().warn(
            'Published ABORT'
        )

        self.active = False
        self.start_pending = False

        # Give callbacks an opportunity to process ABORT.
        self.root.after(
            500,
            self.abort_shutdown
        )


    def abort_shutdown(self):

        self.stop_competition_nodes()

        self.start_button.config(
            state=tk.NORMAL
        )

        self.finish_button.config(
            state=tk.DISABLED
        )

        self.abort_button.config(
            state=tk.DISABLED
        )

        self.status.config(
            text=(
                'RUN ABORTED\n\n'
                f'Team: {self.current_team}\n'
                f'Run: {self.current_run_id}'
            )
        )


    # ==========================================================
    # Finish process shutdown
    # ==========================================================

    def finish_shutdown(self):

        self.shutdown_pending = False

        self.stop_competition_nodes()

        self.start_button.config(
            state=tk.NORMAL
        )

        self.finish_button.config(
            state=tk.DISABLED
        )

        self.abort_button.config(
            state=tk.DISABLED
        )

        self.status.config(
            text=(
                'RUN FINISHED\n\n'
                f'Team: {self.current_team}\n'
                f'Team ID: {self.current_team_id}\n'
                f'Run: {self.current_run_id}\n\n'
                f'Data saved in:\n'
                f'{self.current_folder}'
            )
        )


    # ==========================================================
    # Graceful ROS process termination
    # ==========================================================

    def stop_process(self, process, name):

        if process is None:
            return

        # Already stopped
        if process.poll() is not None:
            return

        try:

            self.get_logger().info(
                f'Stopping {name}...'
            )

            # Equivalent to Ctrl+C
            os.killpg(
                os.getpgid(process.pid),
                signal.SIGINT
            )

            try:

                process.wait(
                    timeout=3.0
                )

            except subprocess.TimeoutExpired:

                self.get_logger().warn(
                    f'{name} did not stop after SIGINT; '
                    'sending SIGTERM.'
                )

                os.killpg(
                    os.getpgid(process.pid),
                    signal.SIGTERM
                )

        except ProcessLookupError:

            pass

        except Exception as e:

            self.get_logger().warn(
                f'Error stopping {name}: {e}'
            )


    def stop_competition_nodes(self):

        self.stop_process(
            self.scorer_process,
            'offline scorer'
        )

        self.stop_process(
            self.tracker_process,
            'LiDAR localization'
        )

        self.scorer_process = None
        self.tracker_process = None


    # ==========================================================
    # GUI shutdown
    # ==========================================================

    def on_gui_close(self):

        if self.active:

            answer = messagebox.askyesno(
                'Competition running',
                (
                    'A run is currently active.\n\n'
                    'Abort the run and close the program?'
                )
            )

            if not answer:
                return

            event = String()
            event.data = 'ABORT'

            self.event_pub.publish(
                event
            )

        self.stop_competition_nodes()

        self.root.destroy()


    def run_gui(self):

        self.root.mainloop()


# ==============================================================
# Main
# ==============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = CompetitionGUI()

    # rclpy must spin while Tkinter owns the main thread.
    ros_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )

    ros_thread.start()

    try:

        node.run_gui()

    finally:

        node.stop_competition_nodes()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()