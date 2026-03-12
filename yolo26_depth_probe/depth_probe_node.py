#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

class DepthProbeNode(Node):
    def __init__(self):
        super().__init__("depth_probe_node")

        # Topics
        self.depth_topic = "/camera/camera/aligned_depth_to_color/image_raw"
        self.info_topic = "/camera/camera/aligned_depth_to_color/camera_info"
        self.uv_topic = "/yolo/center_uv"

        # Subscriptions
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_cb, qos_profile_sensor_data)
        self.info_sub = self.create_subscription(CameraInfo, self.info_topic, self.info_cb, qos_profile_sensor_data)
        self.uv_sub = self.create_subscription(PointStamped, self.uv_topic, self.uv_cb, 10)

        # Publisher
        self.pub = self.create_publisher(PointStamped, "/yolo/depth_uvz", 10)

        self.latest_depth_msg = None
        self.camera_intrinsics = None
        self.bridge = CvBridge()

        self.get_logger().info("Depth Probe Node Initialized (with 3D Projection).")

    def info_cb(self, msg: CameraInfo):
        # Cache camera intrinsics once
        if self.camera_intrinsics is None:
            self.camera_intrinsics = msg
            self.get_logger().info("Camera Intrinsics received.")

    def depth_cb(self, msg: Image):
        self.latest_depth_msg = msg

    def uv_cb(self, uv_msg: PointStamped):
        if self.latest_depth_msg is None or self.camera_intrinsics is None:
            return

        dmsg = self.latest_depth_msg
        u = int(round(uv_msg.point.x))
        v = int(round(uv_msg.point.y))

        if u < 0 or v < 0 or u >= dmsg.width or v >= dmsg.height:
            return

        try:
            cv_depth = self.bridge.imgmsg_to_cv2(dmsg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Depth convert failed: {e}")
            return

        if dmsg.encoding == "16UC1":
            depth_m = self._median_depth(cv_depth, u, v, k=5) / 1000.0
        else:  # 32FC1
            depth_m = float(self._median_depth(cv_depth, u, v, k=5))

        if not np.isfinite(depth_m) or depth_m <= 0.0:
            return

        # --- 2D to 3D Projection Logic ---
        # K matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        fx = self.camera_intrinsics.k[0]
        cx = self.camera_intrinsics.k[2]
        fy = self.camera_intrinsics.k[4]
        cy = self.camera_intrinsics.k[5]

        # Calculate actual 3D coordinates in meters
        x_m = (u - cx) * depth_m / fx
        y_m = (v - cy) * depth_m / fy
        z_m = depth_m

        out = PointStamped()
        out.header = dmsg.header  # Use depth map's header to ensure TF matches
        out.point.x = float(x_m)
        out.point.y = float(y_m)
        out.point.z = float(z_m)
        
        self.pub.publish(out)

    @staticmethod
    def _median_depth(depth_img: np.ndarray, u: int, v: int, k: int = 5) -> float:
        r = k // 2
        y1, y2 = max(0, v - r), min(depth_img.shape[0], v + r + 1)
        x1, x2 = max(0, u - r), min(depth_img.shape[1], u + r + 1)
        patch = depth_img[y1:y2, x1:x2]
        
        if patch.dtype == np.float32:
            patch = patch[np.isfinite(patch)]
            
        patch = patch[patch > 0]
        if patch.size == 0:
            return float("nan") if depth_img.dtype == np.float32 else 0.0
            
        return float(np.median(patch))

def main():
    rclpy.init()
    node = DepthProbeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()