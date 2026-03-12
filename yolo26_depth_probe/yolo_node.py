#!/usr/bin/env python3
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

from ultralytics import YOLO


class YoloAllDepthVizNode(Node):
    def __init__(self):
        super().__init__("yolo_all_depth_viz_node")

        # ---- Topics ---- 실제 로봇 topic 기준
        """
        self.color_topic = "/camera_left/camera_left/color/image_rect_raw"
        self.depth_topic = "/camera_left/camera_left/aligned_depth_to_color/image_raw"
        self.overlay_topic = "/yolo/overlay/image"
        self.uv_topic = "/yolo/center_uv" 
        """
        
        self.color_topic = "/camera/camera/color/image_rect_raw"
        self.depth_topic = "/camera/camera/aligned_depth_to_color/image_raw"
        self.overlay_topic = "/yolo/overlay/image"
        self.uv_topic = "/yolo/center_uv" 
        
        # ---- ROS IO ----
        self.sub_color = self.create_subscription(
            Image, self.color_topic, self.color_cb, qos_profile_sensor_data
        )
        self.sub_depth = self.create_subscription(
            Image, self.depth_topic, self.depth_cb, qos_profile_sensor_data
        )
        self.pub_overlay = self.create_publisher(Image, self.overlay_topic, 10)
        self.pub_uv = self.create_publisher(PointStamped, self.uv_topic, 10) # Added Publisher

        # ---- Model ----
        self.model = YOLO("yolo26n-seg.pt")
        self.bridge = CvBridge()

        # ---- Latest depth cache ----
        self.latest_depth_msg = None
        self.latest_depth_time = 0.0

        # ---- Inference params ----
        self.imgsz = 640
        self.conf = 0.25

        # ---- Optional throttle ----
        self.min_dt = 0.0  
        self._last_run = 0.0

        # ---- Depth sampling ----
        self.depth_patch_k = 5
        self.max_depth_age_s = 0.5

        self.get_logger().info(f"Subscribing color: {self.color_topic}")
        self.get_logger().info(f"Subscribing depth: {self.depth_topic}")
        self.get_logger().info(f"Publishing overlay: {self.overlay_topic}")
        self.get_logger().info(f"Publishing UV: {self.uv_topic}")

    def depth_cb(self, msg: Image):
        self.latest_depth_msg = msg
        self.latest_depth_time = time.time()

    def _depth_at_uv_m(self, u: int, v: int):
        dmsg = self.latest_depth_msg
        if dmsg is None:
            return None
        if (time.time() - self.latest_depth_time) > self.max_depth_age_s:
            return None

        if u < 0 or v < 0 or u >= dmsg.width or v >= dmsg.height:
            return None

        # Safe depth conversion using cv_bridge
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(dmsg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Depth convert failed: {e}")
            return None

        k = int(self.depth_patch_k)
        r = k // 2
        
        y_min, y_max = max(0, v - r), min(cv_depth.shape[0], v + r + 1)
        x_min, x_max = max(0, u - r), min(cv_depth.shape[1], u + r + 1)
        
        patch = cv_depth[y_min:y_max, x_min:x_max]

        if dmsg.encoding == "16UC1":
            patch = patch[patch > 0]
            if patch.size == 0:
                return None
            return float(np.median(patch)) / 1000.0

        elif dmsg.encoding == "32FC1":
            patch = patch[np.isfinite(patch)]
            patch = patch[patch > 0]
            if patch.size == 0:
                return None
            return float(np.median(patch))
            
        return None

    def _rosimg_to_rgb(self, msg: Image) -> np.ndarray:
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def color_cb(self, msg: Image):
        now = time.time()
        if self.min_dt > 0 and (now - self._last_run) < self.min_dt:
            return
        self._last_run = now

        try:
            frame_rgb = self._rosimg_to_rgb(msg)
        except Exception as e:
            self.get_logger().error(f"Color convert failed: {e}")
            return

        try:
            results = self.model.predict(
                source=frame_rgb, 
                imgsz=self.imgsz, 
                conf=self.conf, 
                classes=[63],  # <--- UNCOMMENT this line to detect ONLY laptops
                verbose=False
            )
            r = results[0]
        except Exception as e:
            self.get_logger().error(f"YOLO predict failed: {e}")
            return

        try:
            overlay = r.plot()
        except Exception as e:
            self.get_logger().error(f"r.plot() failed: {e}")
            return

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            cls = r.boxes.cls.detach().cpu().numpy().astype(int)
            confs = r.boxes.conf.detach().cpu().numpy()

            for i in range(xyxy.shape[0]):
                x1, y1, x2, y2 = xyxy[i]
                u = int((x1 + x2) / 2.0)
                v = int((y1 + y2) / 2.0)

                # Publish UV coordinates for DepthProbeNode
                uv_msg = PointStamped()
                uv_msg.header = msg.header
                uv_msg.point.x = float(u)
                uv_msg.point.y = float(v)
                uv_msg.point.z = 0.0
                self.pub_uv.publish(uv_msg)

                cname = str(self.model.names.get(int(cls[i]), cls[i]))
                cconf = float(confs[i])
                z_m = self._depth_at_uv_m(u, v)
                
                ztxt = "z=NA" if z_m is None else f"z={z_m:.3f}m"
                label = f"{cname} {cconf:.2f} {ztxt}"

                cv2.circle(overlay, (u, v), 4, (0, 255, 255), -1)
                tx = int(max(0, x1))
                ty = int(max(15, y1))
                cv2.putText(
                    overlay, label,
                    (tx, ty - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
                )

        try:
            out_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
            out_msg.header = msg.header
            self.pub_overlay.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"Overlay publish failed: {e}")


def main():
    rclpy.init()
    node = YoloAllDepthVizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()