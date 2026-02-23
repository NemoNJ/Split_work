#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32, Int32
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy
import math
import time


class AutoPlantController(Node):
    def __init__(self):
        super().__init__('auto_plant_controller')

        self.n = 5 # number of plants
        self.space = [11.0,15.0,20.0,10.0,10.0]   # cm

        self.WHEEL_RADIUS_CM = 15.0
        self.PULSES_PER_REV = 8192.0

        # ==================================================
        # STATE
        # ==================================================
        self.state = 'IDLE'
        self.current_index = 0
        self.last_avg_ticks = None
        self.total_distance_cm = 0.0
        self.start_distance = 0.0
        self.plant_start_time = None

        # ==================================================
        # QoS
        # ==================================================
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        # ==================================================
        # SUBSCRIBERS
        # ==================================================
        self.create_subscription(
            Float32MultiArray,
            '/encoder_data',
            self.encoder_callback,
            qos
        )

        self.create_subscription(
            Int32,
            '/cal_start',
            self.cal_start_callback,
            10
        )

        # ==================================================
        # PUBLISHERS
        # ==================================================
        self.vel_pub = self.create_publisher(Float32, '/robot_velocity', 10)
        # self.dir_pub = self.create_publisher(Twist, '/robot_direction', 10)
        self.plant_pub = self.create_publisher(Int32, '/auto_plant', 10)

        # ==================================================
        # TIMER
        # ==================================================
        self.timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        self.get_logger().info('✅ Auto Plant Controller Started')

    
    def stop_robot(self):
        self.vel_pub.publish(Float32(data=0.0))

    # ==================================================
    # ENCODER → DISTANCE
    # ==================================================
    def encoder_callback(self, msg):
        if len(msg.data) < 8:
            return

        t1, t2, t3, t4 = msg.data[4:8]
        avg_ticks = (t1 + t2 + t3 + t4) / 4.0

        if self.last_avg_ticks is None:
            self.last_avg_ticks = avg_ticks
            return

        delta_ticks = avg_ticks - self.last_avg_ticks
        self.last_avg_ticks = avg_ticks

        delta_distance = (
            delta_ticks / self.PULSES_PER_REV
        ) * (2.0 * math.pi * self.WHEEL_RADIUS_CM)

        self.total_distance_cm += delta_distance

    # ==================================================
    # START SIGNAL
    # ==================================================
    def cal_start_callback(self, msg):
        if msg.data == 1 and self.state == 'IDLE':
            self.get_logger().info('🚀 CAL START')
            self.current_index = 0
            self.start_distance = self.total_distance_cm
            self.state = 'MOVING'

            # self.set_forward_direction()
            self.vel_pub.publish(Float32(data=0.0))
            self.plant_pub.publish(Int32(data=5))

    # ==================================================
    # MAIN CONTROL LOOP
    # ==================================================
    def control_loop(self):
        if self.state == 'IDLE':
            return

        # ---------------- MOVING ----------------
        if self.state == 'MOVING':
            traveled = self.total_distance_cm - self.start_distance
            target = self.space[self.current_index]


            if traveled >= target:
                self.stop_robot()
                self.plant_pub.publish(Int32(data=2))
                self.plant_pub.publish(Int32(data=1))
                self.plant_start_time = time.time()
                self.state = 'PLANTING'
            else:
                self.vel_pub.publish(Float32(data=60.0))

        # ---------------- PLANTING ----------------
        elif self.state == 'PLANTING':
            if time.time() - self.plant_start_time >= 35.0:
                self.current_index += 1

                if self.current_index >= self.n:
                    self.plant_pub.publish(Int32(data=3))
                    self.stop_robot()
                    self.state = 'FINISHED'
                    self.get_logger().info('🌱 PLANTING FINISHED')
                else:
                    self.start_distance = self.total_distance_cm
                    self.vel_pub.publish(Float32(data=60.0))
                    self.state = 'MOVING'

        # ---------------- FINISHED ----------------
        elif self.state == 'FINISHED':
            pass


def main(args=None):
    rclpy.init(args=args)
    node = AutoPlantController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
