#!/usr/bin/env python3
import cv2
import socket
import struct
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32
from geometry_msgs.msg import Twist
from ultralytics import YOLO
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2

SERVER_IP = "192.168.0.247"
SERVER_PORT = 5001

CLASS_FRONT = "Front_pot"
CLASS_TOP   = "Top_pot"

CAM1_DEV = "/dev/video2"
CAM2_DEV = "/dev/video0"

WIDTH = 320
HEIGHT = 240
FPS = 15

model = YOLO("best-2.pt")

LINE_X1 = 80
LINE_X2 = 240
CENTER_DEADZONE = 16

class PID:
    def __init__(self, kp, ki, kd, out_limit=1.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_limit = out_limit
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def update(self, error):
        now = time.time()
        if self.last_time is None:
            self.last_time = now
            self.last_error = error
            return 0.0

        dt = now - self.last_time
        if dt <= 0.0:
            return 0.0

        self.integral += error * dt
        derivative = (error - self.last_error) / dt

        out = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        out = max(-self.out_limit, min(self.out_limit, out))

        self.last_error = error
        self.last_time = now
        return out

def enhance_image(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.convertScaleAbs(y, alpha=1.2, beta=50)
    return cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)

def open_cam(dev):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def create_udp_socket():
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_frame(sock, cam_id, frame):
    ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return

    data = enc.tobytes()
    packet = bytes([cam_id]) + struct.pack("!Q", len(data)) + data
    sock.sendto(packet, (SERVER_IP, SERVER_PORT))

class AIVisionNode(Node):
    def __init__(self):
        super().__init__("ai_vision_node")

        # ---------- image buffers ----------
        self.cam1_frame = None   # AI cam (เดิม video2)
        self.cam2_frame = None   # Front cam (เดิม video0)

        # ---------- subscribers ----------
        self.sub_cam1 = self.create_subscription(
            CompressedImage,
            "camera/ai/image/compressed",
            self.cam1_cb,
            10
        )

        self.sub_cam2 = self.create_subscription(
            CompressedImage,
            "camera/front/image/compressed",
            self.cam2_cb,
            10
        )
        self.dir_pub  = self.create_publisher(Twist,   "/robot_direction", 10)
        self.vel_pub  = self.create_publisher(Float32, "/robot_velocity",  10)
        self.cam_pub  = self.create_publisher(Int32,   "/move_camera",     10)
        self.cam_moved = False
        self.dx_filt = 0.0
        self.state = "TRACK"
        self.search_start = time.time()
        self.track_start_time = None
        self.state_switch = 0   # ค่าเริ่มต้น = ยังไม่ทำงาน
        self.state_pub = self.create_publisher(Int32,"/state_switch",10)
        self.state_sub = self.create_subscription(Int32,"/state_switch",self.state_switch_cb,10)
        self.turn_pid = PID(1.6, 0.0, 0.01, out_limit=1.0)
        self.udp_sock = create_udp_socket()
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
    def pub_f32(self, pub, value):
        pub.publish(Float32(data=float(value)))
        
    def state_switch_cb(self, msg: Int32):
        self.state_switch = msg.data
    
    def stop(self):
        self.dir_pub.publish(Twist())
        self.pub_f32(self.vel_pub, 0.0)
    def decode_compressed(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def cam1_cb(self, msg):
        self.cam1_frame = self.decode_compressed(msg)

    def cam2_cb(self, msg):
        self.cam2_frame = self.decode_compressed(msg)
    def control_loop(self):
        if self.cam1_frame is None:
            return

        if self.state_switch == 1:
            annotated = run_ai(self.cam1_frame, self)
        else:
            annotated = self.cam1_frame

        # ส่งภาพออก UDP (เหมือนเดิม)
        send_frame(self.udp_sock, 1, annotated)
def run_ai(frame, node: AIVisionNode):
    annotated = frame.copy()
    pre = enhance_image(frame)
    results = model.track(pre, persist=True, conf=0.7)

    h, w = frame.shape[:2]
    cx_frame = w // 2

    object_detected = False
    bbox_h = 0.0
    dx = 0.0
    cx = cx_frame

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss  = results[0].boxes.cls.cpu().numpy().astype(int)
        names = results[0].names

        target_class = CLASS_FRONT if node.state == "TRACK" else CLASS_TOP

        for box, conf, cls_id in zip(boxes, confs, clss):
            if conf < 0.45:
                continue

            cls_name = names[cls_id]
            if cls_name != target_class:
                continue

            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            left_edge  = int(x1)
            right_edge = int(x2)
            cy = int((y1 + y2) / 2)
            dx = cx - cx_frame
            dx_norm = dx / (w / 2)   # อยู่ในช่วง [-1, 1]

            bbox_h = y2 - y1

            object_detected = True

    
            
            cv2.circle(annotated, (cx, cy), 6, (0,255,0), -1)
            cv2.circle(annotated, (cx, cy), 8, (0,0,0), 2)
            cv2.rectangle(annotated,(left_edge, int(y1)),(right_edge, int(y2)),(255, 0, 0), 2)
            crosshair_size = 15
            cv2.line(annotated, (cx-crosshair_size, cy), (cx+crosshair_size, cy), (0,255,0), 2)
            cv2.line(annotated, (cx, cy-crosshair_size), (cx, cy+crosshair_size), (0,255,0), 2)


            text = f"H:{bbox_h} "
            text_x = int(x1)
            text_y = int(y1) - 10
            if text_y < 20:
                text_y = int(y2) + 20
            cv2.putText(annotated, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

            if -CENTER_DEADZONE < dx < CENTER_DEADZONE:
                cv2.circle(annotated, (cx, cy), 25, (0,255,0), 3)
            cv2.circle(annotated, (cx, int((y1+y2)/2)), 6, (0,255,0), -1)
            break
        
        #state search → track → plant → idle

    if node.state == "TRACK":
        if not object_detected:
            node.turn_pid.reset()
            node.dx_filt = 0.0
            cmd = Twist()
            cmd.linear.x = 0.2      # วิ่งตรงช้า ๆ
            cmd.angular.z = 0.0    # ไม่เลี้ยว
            node.dir_pub.publish(cmd)

            return annotated
    
        if abs(dx_norm) < 0.05:
            dx_norm = 0.0
        alpha = 0.25
        node.dx_filt = alpha * dx_norm + (1 - alpha) * node.dx_filt

        turn_cmd = node.turn_pid.update(node.dx_filt)
        
        if bbox_h > 126:
            node.turn_pid.reset()
            node.state = "PLANT"
            if hasattr(node, "plant_start"):
                 del node.plant_start   

            if hasattr(node, "plant_phase"):
                del node.plant_phase  
            node.cam_pub.publish(Int32(data=0))
            return annotated
        elif bbox_h > 104:
            #node.cam_pub.publish(Int32(data=0))  
            if not node.cam_moved:
                node.cam_pub.publish(Int32(data=0))
                node.cam_moved = True
            
        # ---------- DIRECTION ----------
        base_v = 1.0
        v = base_v * (1.0 - min(abs(dx_norm), 1.0))

        t = Twist()
        t.linear.x = v
        t.angular.z = -turn_cmd   # เครื่องหมายเช็กอีกทีตามทิศ
        node.dir_pub.publish(t)


        # ---------- SPEED ----------
        if bbox_h < 30:
            node.pub_f32(node.vel_pub, 90.0)
        elif bbox_h < 40:
            node.pub_f32(node.vel_pub, 80.0)
        else:
            node.pub_f32(node.vel_pub, 40.0)

    elif node.state == "PLANT":

        # ---------------- INIT PLANT STATE ----------------
        if not hasattr(node, 'plant_start'):
            node.plant_start = time.time()
            node.plant_phase = "RUN"     # RUN → STOP → DONE

            node.pub_f32(node.vel_pub, 40.0)   # 🟢 วิ่งทันที 3 วิ
            node.turn_pid.reset()

            return annotated

        elapsed = time.time() - node.plant_start

        # ---------------- PHASE 1: RUN 3 sec ----------------
        if node.plant_phase == "RUN":
            if elapsed >= 3.0:
                node.pub_f32(node.vel_pub, 0.0)   # 🔴 หยุด
                node.plant_phase = "STOP"
                node.plant_start = time.time()    # reset timer

            return annotated

        # ---------------- PHASE 2: STOP 2 sec ----------------
        if node.plant_phase == "STOP":
            if elapsed >= 2.0:
                node.state_switch = 2
                node.state_pub.publish(Int32(data=2))
                node.state = "IDLE"

            return annotated

        
    return annotated

def main():
    rclpy.init()
    node = AIVisionNode()

    udp_sock = create_udp_socket()

    try:
        while rclpy.ok():
            rclpy.spin_once(node)

    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
        udp_sock.close()

if __name__ == "__main__":
    main() 
