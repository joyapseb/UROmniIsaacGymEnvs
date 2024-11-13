# #This code is based on the following repository
# # https://github.com/j3soon/OmniIsaacGymEnvs-UR10Reacher/tree/main
# #Copyright (C) 2022-2023, Johnson Sun

# #sim2real for ROS2 // to do: setup for ro2 and test with simulations
# import asyncio
# import math
# import numpy as np

# import rclpy
# from rclpy.node import Node
# from control_msg.msg import JointTrajectoryControllerState
# from trajectory_msg.mg import JointTrajectory, JointTrajectoryPoint

# class RealUR5e(Node):
#     sim_dof_angle_limits = [
#         (-360, 360, False),
#         (-360, 360, False),
#         (-360, 360, False),
#         (-360, 360, False),
#         (-360, 360, False),
#         (-360, 360, False),
#         #not using gripper
#         # (-360, 360, False),
#         # (-360, 360, False),
#     ]

#     pi = math.pi #ros2

#     servo_angle_limits = [
#         (-2 * pi, 2 * pi),
#         (-2 * pi, 2 * pi),
#         (-2 * pi, 2 * pi),
#         (-2 * pi, 2 * pi),
#         (-2 * pi, 2 * pi),
#         (-2 * pi, 2 * pi),
#     ]

#     state_topic = '/scaled_pos_joint_traj_controller/state'
#     cmd_topic = '/scaled_pos_joint_traj_controller/command'

#     joint_names = [
#         'shoulder_pan_joint',
#         'shoulder_lift_joint',
#         'elbow_joint',
#         'wrist_1_joint',
#         'wrist_2_joint',
#         'wrist_3_joint'
#     ]
#     joint_names_index = {
#         'shoulder_pan_joint': 0,
#         'shoulder_lift_joint': 1,
#         'elbow_joint': 2,
#         'wrist_1_joint': 3,
#         'wrist_2_joint': 4,
#         'wrist_3_joint': 5
#     }

#     def __init__(self):
#         super().__init__("ur5e_sim2real")
#         self.pub_freq = 10

#         self.current_pos = None
#         self.target_pos = None

#         self.sub = self.create_subscription(JointTrajectoryControllerState,
#                                             self.state_topic, self.sub_callback,1)
        
#         self.pub = self.create_publisher(JointTrajectory,
#                                          self.cmd_topic, 1)
        
#         self.min_traj_dur = 0
#         #timer for rospy.spin and asyncio loop
#         self.create_timer(1.0 / self.pub_freq, self.pub_task)

#     def sub_callback(self, msg):
#         actual_pos = {}
#         for i in range(len(msg.joint_names)):
#             joint_name = msg.joint_names[i]
#             joint_pos = msg.actual_positions[i]
#             actual_pos[joint_name] = joint_pos
#         self.current_pos = actual_pos

#     def pub_task(self):
#         if self.current_pos is None or self.target_pos is None:
#             return
        
#         traj = JointTrajectory()
#         traj.joint_names = self.joint_names
#         point = JointTrajectoryPoint()
#         moving_average = 1 #1, immediate position shift 
#         dur =[]

#         for name in traj.joint_names:
#             pos = self.current_pos[name]
#             cmd = pos * (1 - moving_average) + self.target_pos[self.joint_names_index]
#             max_vel = 3.15 
#             duration = abs(cmd-pos) / max_vel
#             dur.append(max(duration, self.min_traj_dur))

#             point.positions.append(cmd)

#         point.time_from_start = rclpy.time.Duration(seconds = max(dur))
#         traj.points.append(point)
#         self.pub.publish(traj)

#     def send_joint_pos (self, joint_pos):
#         if len(joint_pos) != 6:
#             raise Exception("Lenght does not match")
#         target_pos = [0] *6
#         for i,pos in enumerate(joint_pos):
#             L, U, inversed = self.sim_dof_angle_limits[i] #degrees
#             A, B = self.servo_angle_limits[i] #radians

#             #to degrees as in simulations its radians
#             angle = np.rad2deg(float(pos))

#             if not L <= angle <= U:
#                 print("Simulation angle out of allowed range...CLIP")
#                 angle = np.clip(angle,L,U)
            
#             #linear transformation, scale angle into new range
#             target_pos[i] = (angle - L) * ((B-A) / (U - L)) + A #map [l,u] to [a,b]

#             if not A <= target_pos[i] <= B:
#                 raise Exception("Major ERROR")
#         self.target_pos = target_pos


# def main(args=None):
#     rclpy.init(args=args)
#     print("Run ros2 ur_robot_driver")
#     ur5e = RealUR5e()
#     try:
#         rclpy.spin(ur5e)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         ur5e.destroy_node()
#         rclpy.shutdown()

# if __name__ == "__main__":
#     main()



