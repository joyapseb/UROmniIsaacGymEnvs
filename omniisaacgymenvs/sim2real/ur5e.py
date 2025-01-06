
import time
import numpy as np
import asyncio
import omni 
from omni.isaac.core.utils.extensions import enable_extension
from rclpy.duration import Duration
enable_extension("omni.isaac.ros2_bridge")
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Int32
from rclpy.qos import QoSProfile, ReliabilityPolicy
import asyncio

class UR5eTraj(Node):

    joint_name_to_idx = {
        'elbow_joint': 2,
        'shoulder_lift_joint': 1,
        'shoulder_pan_joint': 0,
        'wrist_1_joint': 3,
        'wrist_2_joint': 4,
        'wrist_3_joint': 5
    }
    def __init__(self):

        super().__init__('ur5e_traj')
    
        print("connecting...ROS2")
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.traj_publisher = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory',qos_profile)
        self.gripper_publisher = self.create_publisher(Int32,"/gripper",qos_profile)
        self.subs = self.create_subscription(JointState, '/joint_states', self.joint_callback ,qos_profile)
        print("Subscription to /gripper created")

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ]

        # Data to store the latest trajectory
        self.traj_data = None
        self.grp_state = None
        self.call = None
        self.current_pos = None
        self.grp = Int32()

        def custom_exception_handler(loop, context):
            print(context)

        asyncio.get_event_loop().set_exception_handler(custom_exception_handler)
        asyncio.ensure_future(self.publish_trajectory())

    async def publish_trajectory(self):
        # Publish the latest trajectory if available
        while rclpy.ok():
            await asyncio.sleep(1/120)  # ~120 Hz
            if self.traj_data is None or self.current_pos is None:
                continue

            dur = []
            point = JointTrajectoryPoint()
            moving_avg = 1.0

            for idx, name in enumerate(self.traj_data.joint_names):
                pos = self.current_pos[name]
                print(f"R-UR5e Position of {name}: {pos}")
                point_position = self.traj_data.points[0].positions[idx]
                print(f"S-UR5e Position of {name}: {point_position}")

                cmd = pos * (1 - moving_avg) + point_position * moving_avg

                duration = abs(cmd - pos) 
                # print(f"Duration for {name}: {duration:.6f}")
                dur.append(max(duration, 0))
                # print(f"Durations: {dur}")
                point.positions.append(cmd)

            max_dur = max(dur)
            point.time_from_start = Duration(seconds=max_dur).to_msg()

            self.traj_data.points = [point]  

            if self.grp_state:
                self.grp.data = 0
            else:
                self.grp.data = 150

            self.traj_publisher.publish(self.traj_data)
            self.gripper_publisher.publish(self.grp)
            self.get_logger().info("Published trajectory for UR5e")

    def joint_callback(self, msg):
        self.current_pos = {name: pos for name, pos in zip(msg.name, msg.position)}

    def send_joint_pos(self, joint_pos, grp_open):
        joint_pos = joint_pos[0].cpu().numpy().astype('float64').tolist()
        targ_point = JointTrajectoryPoint()
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        targ_point.positions = joint_pos[:6]
        traj_msg.points.append(targ_point)
        self.traj_data = traj_msg
        self.grp_state = grp_open
