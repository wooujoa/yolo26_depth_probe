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
from rclpy.qos import qos_profile_sensor_data
from collections import deque


class YoloGpuTrackerNode(Node):
    def __init__(self):
        super().__init__("yolo_gpu_tracker_node")

        # 1. GPU 및 모델 설정
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("yolo26n-seg.pt").to(self.device)
        self.get_logger().info(f"🚀 YOLO started on: {self.device}")

        # 2. Topics
        self.color_topic = "/camera_l/camera_l/color/image_rect_raw/compressed"
        self.depth_topic = "/camera_l/camera_l/aligned_depth_to_color/image_raw"
        self.info_topic = "/camera_l/camera_l/aligned_depth_to_color/camera_info"
        self.target_3d_topic = "/yolo/target_3d_pose"

        # 3. Subscriptions
        self.sub_color = self.create_subscription(
            CompressedImage, self.color_topic, self.color_cb, qos_profile_sensor_data
        )
        self.sub_depth = self.create_subscription(
            Image, self.depth_topic, self.depth_cb, qos_profile_sensor_data
        )
        self.sub_info = self.create_subscription(
            CameraInfo, self.info_topic, self.info_cb, qos_profile_sensor_data
        )

        # 4. Publisher
        self.pub_3d = self.create_publisher(PointStamped, self.target_3d_topic, 10)

        self.bridge = CvBridge()
        self.latest_depth_msg = None
        self.camera_intrinsics = None
        self.imgsz = 640
        self.frame_count = 0

        # ===== Z filtering buffer =====
        self.z_history = deque(maxlen=7)   # 최근 7프레임 저장
        self.min_history_size = 3          # 최소 3개 이상 쌓여야 publish
        self.outlier_thresh_m = 0.10       # median 기준 10cm 이상 차이나면 outlier 제거

    def info_cb(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self.get_logger().info("✅ Camera Info Received")
            self.camera_intrinsics = msg

    def depth_cb(self, msg: Image):
        self.latest_depth_msg = msg

    def color_cb(self, msg: CompressedImage):
        self.frame_count += 1

        if self.camera_intrinsics is None:
            if self.frame_count % 30 == 0:
                self.get_logger().warn("⚠️ Waiting for Camera Info...")
            return

        if self.latest_depth_msg is None:
            if self.frame_count % 30 == 0:
                self.get_logger().warn("⚠️ Waiting for Depth Image...")
            return

        # optional: color/depth timestamp 차이 체크
        self._log_time_diff_if_needed(msg, self.latest_depth_msg)

        # 이미지 디코딩
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if decoded is None:
                self.get_logger().error("❌ Decoding Error: imdecode returned None")
                return
            frame_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f"❌ Decoding Error: {e}")
            return

        # YOLO 추론
        results = self.model.predict(
            source=frame_rgb,
            imgsz=self.imgsz,
            conf=0.25,
            verbose=False,
            device=self.device,
            half=True
        )
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            self.get_logger().info(f"🎯 Detected: {len(r.boxes)} objects")

            try:
                boxes = r.boxes.xyxy.cpu().numpy()
                cv_depth = self.bridge.imgmsg_to_cv2(
                    self.latest_depth_msg, desired_encoding="passthrough"
                )

                fx = float(self.camera_intrinsics.k[0])
                fy = float(self.camera_intrinsics.k[4])
                cx = float(self.camera_intrinsics.k[2])
                cy = float(self.camera_intrinsics.k[5])

                for box in boxes:
                    u = int((box[0] + box[2]) / 2)
                    v = int((box[1] + box[3]) / 2)

                    depth_m = self._get_median_depth(
                        cv_depth,
                        self.latest_depth_msg.encoding,
                        u,
                        v
                    )

                    if depth_m is not None:
                        # raw depth를 history에 저장
                        self.z_history.append(depth_m)

                        # 충분한 히스토리가 쌓였을 때만 publish
                        if len(self.z_history) < self.min_history_size:
                            self.get_logger().info(
                                f"⏳ Collecting depth history... "
                                f"{len(self.z_history)}/{self.min_history_size} "
                                f"(raw_z={depth_m:.3f}m)"
                            )
                            continue

                        z_filtered = self._get_filtered_depth()

                        if z_filtered is None:
                            self.get_logger().warn(
                                f"❓ Could not compute filtered depth at ({u}, {v})"
                            )
                            continue

                        out_pt = PointStamped()
                        out_pt.header = self.latest_depth_msg.header
                        out_pt.point.x = (u - cx) * z_filtered / fx
                        out_pt.point.y = (v - cy) * z_filtered / fy
                        out_pt.point.z = float(z_filtered)

                        self.pub_3d.publish(out_pt)

                        self.get_logger().info(
                            f"📤 Published 3D Pose | "
                            f"pixel=({u},{v}) "
                            f"raw_z={depth_m:.3f}m "
                            f"filtered_z={z_filtered:.3f}m "
                            f"history={list(np.round(np.array(self.z_history), 3))}"
                        )
                    else:
                        self.get_logger().warn(f"❓ Invalid Depth at center ({u}, {v})")

            except Exception as e:
                self.get_logger().error(f"❌ Processing Error: {e}")

        else:
            if self.frame_count % 30 == 0:
                self.get_logger().info("🔍 Searching for target...")

    def _get_median_depth(self, cv_depth, encoding, u, v):
        r = 2
        try:
            h, w = cv_depth.shape[:2]
            u0 = max(0, u - r)
            u1 = min(w, u + r + 1)
            v0 = max(0, v - r)
            v1 = min(h, v + r + 1)

            patch = cv_depth[v0:v1, u0:u1]
            if patch.size == 0:
                return None

            if "16UC1" in encoding:
                valid = patch[patch > 0]
                return float(np.median(valid)) / 1000.0 if valid.size > 0 else None

            elif "32FC1" in encoding:
                valid = patch[np.isfinite(patch) & (patch > 0)]
                return float(np.median(valid)) if valid.size > 0 else None

            else:
                self.get_logger().error(f"Unsupported depth encoding: {encoding}")
                return None

        except Exception as e:
            self.get_logger().error(f"Depth error: {e}")
            return None

    def _get_filtered_depth(self):
        try:
            if len(self.z_history) == 0:
                return None

            vals = np.array(self.z_history, dtype=np.float32)
            med = np.median(vals)

            # median 기준 outlier 제거
            inliers = vals[np.abs(vals - med) <= self.outlier_thresh_m]

            if inliers.size == 0:
                return float(med)

            # 마지막 대표값도 median으로
            return float(np.median(inliers))

        except Exception as e:
            self.get_logger().error(f"Filtered depth error: {e}")
            return None

    def _log_time_diff_if_needed(self, color_msg, depth_msg):
        try:
            color_t = color_msg.header.stamp.sec + color_msg.header.stamp.nanosec * 1e-9
            depth_t = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9
            diff_ms = abs(color_t - depth_t) * 1000.0

            if diff_ms > 100.0:
                self.get_logger().warn(
                    f"⚠️ Color/Depth timestamp diff is large: {diff_ms:.2f} ms"
                )
        except Exception as e:
            self.get_logger().warn(f"⚠️ Timestamp check failed: {e}")


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