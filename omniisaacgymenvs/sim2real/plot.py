import time
import numpy as np
import omni 
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.ros2_bridge")
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench
from geometry_msgs.msg import Vector3
from std_msgs.msg import Int32
from rclpy.qos import QoSProfile
import asyncio

class Plotter(Node):
    sub_topic = '/gripper'
    def __init__(self):
        super().__init__('plotter')

        print("connecting...PLOT...ROS2")
        
        self.data = None
        self.call = None
        qos_profile = QoSProfile(depth=10)
        self.tf_publisher = self.create_publisher(Wrench, '/simulation/tf_data', qos_profile)

        def custom_exception_handler(loop, context):
            print(context)

        asyncio.get_event_loop().set_exception_handler(custom_exception_handler)
        asyncio.ensure_future(self.publish_tf())

    async def publish_tf(self):
        while rclpy.ok():
            await asyncio.sleep(1.0 / 120)
            # if self.call is None:
            #     continue
            if self.data is None:
                continue
            self.tf_publisher.publish(self.data)
            # self.get_logger().info("Published TF of UR5e")


    def get_plot(self, tf):
        tf_data = Wrench()
        force_vec = Vector3()
        torque_vec = Vector3()
        tf = tf.cpu().numpy().astype('float64').tolist()
        force_vec.x, force_vec.y, force_vec.z = tf[0], tf[1], tf[2]
        torque_vec.x, torque_vec.y, torque_vec.z = tf[3], tf[4], tf[5]
        tf_data.force = force_vec
        tf_data.torque = torque_vec
        self.data = tf_data

if __name__ == '__main__':
    rclpy.init(args=None)
    node = Plotter()
    try:
        rclpy.spin(node)
    except:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()