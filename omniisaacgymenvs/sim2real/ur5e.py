# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# import time
# import numpy as np

# #WORKS IN SIMULATION ISAACSIM -> MOVEIT/GAZEBO

# class UR5eTraj(Node):
#     def __init__(self):
#         super().__init__('ur5e_traj')

#         self.publish_topic = '/scaled_joint_trajectory_controller/joint_trajectory'
#         self.traj_publisher = self.create_publisher(JointTrajectory, 
#                                                    self.publish_topic,10)
        
#         # Subscriber to joint_states
#         self.joint_state_sub = self.create_subscription(
#             JointState, 'joint_states_ur5e', self.sub_callback, 10)

#         self.joint_names = [
#             'shoulder_pan_joint',
#             'shoulder_lift_joint',
#             'elbow_joint',
#             'wrist_1_joint',
#             'wrist_2_joint',
#             'wrist_3_joint'
#         ]

#         # Data to store the latest trajectory
#         self.data = None

#         # Create a timer to publish at a regular interval
#         self.create_timer(1.0, self.publish_trajectory)  # Publish every 1 second

#         time.sleep(1)

#     def sub_callback(self, msg):
#         # Update the trajectory based on the latest joint states
#         traj_msg = JointTrajectory()
#         traj_msg.joint_names = self.joint_names

#         targ_point = JointTrajectoryPoint()
#         targ_point.positions = msg.position
#         targ_point.time_from_start.sec = 0
#         targ_point.time_from_start.nanosec = 800 * 1000000  # 800 ms

#         traj_msg.points.append(targ_point)
#         self.data = traj_msg
#         self.get_logger().info("Updated trajectory for UR5e")

#     def publish_trajectory(self):
#         # Publish the latest trajectory if available
#         if self.data is not None:
#             self.traj_publisher.publish(self.data)
#             self.get_logger().info("Published trajectory for UR5e")

# def main(args=None):
#     print("STARTING NODE")
#     rclpy.init(args=args)
#     ur5e_mover = UR5eTraj()
#     try:
#         rclpy.spin(ur5e_mover)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         ur5e_mover.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
