#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
from ultralytics import YOLO

class YoloGpuTrackerNode(Node):
    def __init__(self):
        super().__init__("yolo_gpu_tracker_node")

        # 1. GPU 및 모델 설정
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        # Jetson Orin 최적화를 위해 half=True 및 가속 설정 반영
        self.model = YOLO("yolo26n-seg.pt").to(self.device)
        self.get_logger().info(f"🚀 YOLO started on: {self.device}")

        # 2. Topics (설정하신 토픽명 유지) 
        self.color_topic = "/camera_r/camera_r/color/image_rect_raw/compressed"
        self.depth_topic = "/camera_r/camera_r/depth/image_rect_raw"
        self.info_topic = "/camera_r/camera_r/depth/camera_info"
        self.target_3d_topic = "/yolo/target_3d_pose"

        # 3. Subscriptions (QoS 10)
        self.sub_color = self.create_subscription(CompressedImage, self.color_topic, self.color_cb, 10)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.depth_cb, 10)
        self.sub_info = self.create_subscription(CameraInfo, self.info_topic, self.info_cb, 10)

        # 4. Publisher
        self.pub_3d = self.create_publisher(PointStamped, self.target_3d_topic, 10)

        self.bridge = CvBridge()
        self.latest_depth_msg = None
        self.camera_intrinsics = None
        self.imgsz = 640
        
        # 디버깅용 카운터
        self.frame_count = 0

    def info_cb(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self.get_logger().info("✅ Camera Info Received")
            self.camera_intrinsics = msg

    def depth_cb(self, msg: Image):
        self.latest_depth_msg = msg

    def color_cb(self, msg: CompressedImage):
        self.frame_count += 1
        
        # --- 디버깅 단계 1: 데이터 수신 확인 ---
        if self.camera_intrinsics is None:
            if self.frame_count % 30 == 0: # 너무 자주 찍히지 않게 조절
                self.get_logger().warn("⚠️ Waiting for Camera Info...")
            return

        if self.latest_depth_msg is None:
            if self.frame_count % 30 == 0:
                self.get_logger().warn("⚠️ Waiting for Depth Image...")
            return

        # --- 디버깅 단계 2: 이미지 디코딩 ---
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame_rgb = cv2.cvtColor(cv2.imdecode(np_arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f"❌ Decoding Error: {e}")
            return

        # --- 디버깅 단계 3: YOLO 추론 ---
        # classes=[39]는 'bottle' 등 특정 클래스이므로, 테스트 시에는 빼거나 물체를 꼭 두세요. 
        results = self.model.predict(
            source=frame_rgb, imgsz=self.imgsz, conf=0.25,
            classes=[39], verbose=False, device=self.device, half=True
        )
        r = results[0]

        # --- 디버깅 단계 4: 객체 검출 및 좌표 계산 ---
        if r.boxes is not None and len(r.boxes) > 0:
            self.get_logger().info(f"🎯 Detected: {len(r.boxes)} objects (Class 39)")
            
            try:
                boxes = r.boxes.xyxy.cpu().numpy()
                cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, desired_encoding="passthrough")
                fx, cx, fy, cy = self.camera_intrinsics.k[0], self.camera_intrinsics.k[2], self.camera_intrinsics.k[4], self.camera_intrinsics.k[5]

                for box in boxes:
                    u, v = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                    depth_m = self._get_median_depth(cv_depth, self.latest_depth_msg.encoding, u, v)

                    if depth_m:
                        # PointStamped 발행
                        out_pt = PointStamped()
                        out_pt.header = self.latest_depth_msg.header
                        out_pt.point.x = (u - cx) * depth_m / fx
                        out_pt.point.y = (v - cy) * depth_m / fy
                        out_pt.point.z = float(depth_m)
                        
                        self.pub_3d.publish(out_pt)
                        self.get_logger().info(f"📤 Published 3D Pose: Z={depth_m:.2f}m")
                    else:
                        self.get_logger().warn(f"❓ Invalid Depth at center ({u}, {v})")
            except Exception as e:
                self.get_logger().error(f"❌ Processing Error: {e}")
        else:
            if self.frame_count % 30 == 0:
                self.get_logger().info("🔍 Searching for target (Class 39)...")

    def _get_median_depth(self, cv_depth, encoding, u, v):
        r = 3 # 픽셀 주변부 탐색 범위 확대
        try:
            patch = cv_depth[max(0, v-r):v+r+1, max(0, u-r):u+r+1]
            valid = patch[patch > 0]
            # 인코딩이 16UC1(mm)인 경우 1000으로 나눔
            return float(np.median(valid)) / 1000.0 if valid.size > 0 else None
        except:
            return None

def main():
    rclpy.init()
    node = YoloGpuTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()