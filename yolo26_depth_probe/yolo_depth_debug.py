#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from rclpy.qos import qos_profile_sensor_data
from collections import deque


class YoloGpuTrackerNode(Node):
    def __init__(self):
        super().__init__("yolo_gpu_tracker_node_left")

        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("yolo26n-seg.pt").to(self.device)
        self.get_logger().info(f"🚀 YOLO started on: {self.device}")

        self.color_topic = "/camera_l/camera_l/color/image_rect_raw/compressed"
        self.depth_topic = "/camera_l/camera_l/aligned_depth_to_color/image_raw"
        self.info_topic = "/camera_l/camera_l/aligned_depth_to_color/camera_info"

        self.target_3d_topic = "/yolo/target_3d_pose_l"
        self.bbox_center_topic = "/yolo/bbox_center_px_l"
        self.bbox_size_topic = "/yolo/bbox_size_px_l"
        self.bbox_size_m_topic = "/yolo/bbox_size_m_l"
        self.debug_image_topic = "/yolo/debug_image_l/compressed"

        self.declare_parameter("debug_image_enable", True)
        self.declare_parameter("debug_publish_period_sec", 3.0)
        self.declare_parameter("debug_jpeg_quality", 40)
        self.declare_parameter("log_every_detection", True)

        self.declare_parameter("skip_if_time_diff_too_large", False)
        self.declare_parameter("max_color_depth_dt_ms", 120.0)
        self.declare_parameter("warn_if_time_diff_ms", 100.0)

        self.debug_image_enable = bool(self.get_parameter("debug_image_enable").value)
        self.debug_publish_period_sec = float(self.get_parameter("debug_publish_period_sec").value)
        self.debug_jpeg_quality = int(self.get_parameter("debug_jpeg_quality").value)
        self.log_every_detection = bool(self.get_parameter("log_every_detection").value)

        self.skip_if_time_diff_too_large = bool(self.get_parameter("skip_if_time_diff_too_large").value)
        self.max_color_depth_dt_ms = float(self.get_parameter("max_color_depth_dt_ms").value)
        self.warn_if_time_diff_ms = float(self.get_parameter("warn_if_time_diff_ms").value)

        self.sub_color = self.create_subscription(
            CompressedImage, self.color_topic, self.color_cb, qos_profile_sensor_data
        )
        self.sub_depth = self.create_subscription(
            Image, self.depth_topic, self.depth_cb, qos_profile_sensor_data
        )
        self.sub_info = self.create_subscription(
            CameraInfo, self.info_topic, self.info_cb, qos_profile_sensor_data
        )

        self.pub_3d = self.create_publisher(PointStamped, self.target_3d_topic, 10)
        self.pub_bbox_center = self.create_publisher(PointStamped, self.bbox_center_topic, 10)
        self.pub_bbox_size = self.create_publisher(Float32MultiArray, self.bbox_size_topic, 10)
        self.pub_bbox_size_m = self.create_publisher(Float32MultiArray, self.bbox_size_m_topic, 10)
        self.pub_debug_img = self.create_publisher(CompressedImage, self.debug_image_topic, 3)

        self.bridge = CvBridge()
        self.latest_depth_msg = None
        self.camera_intrinsics = None
        self.imgsz = 640
        self.frame_count = 0

        self.z_history = deque(maxlen=7)
        self.min_history_size = 3
        self.outlier_thresh_m = 0.10

        self.last_debug_pub_time = None

        self.get_logger().info("========================================")
        self.get_logger().info("YOLO LEFT ARM NODE STARTED")
        self.get_logger().info(f"color_topic                : {self.color_topic}")
        self.get_logger().info(f"depth_topic                : {self.depth_topic}")
        self.get_logger().info(f"info_topic                 : {self.info_topic}")
        self.get_logger().info(f"target_3d_topic            : {self.target_3d_topic}")
        self.get_logger().info(f"bbox_center_topic          : {self.bbox_center_topic}")
        self.get_logger().info(f"bbox_size_topic            : {self.bbox_size_topic}")
        self.get_logger().info(f"bbox_size_m_topic          : {self.bbox_size_m_topic}")
        self.get_logger().info(f"debug_image_topic          : {self.debug_image_topic}")
        self.get_logger().info(f"debug_image_enable         : {self.debug_image_enable}")
        self.get_logger().info(f"debug_publish_period_sec   : {self.debug_publish_period_sec}")
        self.get_logger().info(f"debug_jpeg_quality         : {self.debug_jpeg_quality}")
        self.get_logger().info(f"skip_if_time_diff_too_large: {self.skip_if_time_diff_too_large}")
        self.get_logger().info(f"max_color_depth_dt_ms      : {self.max_color_depth_dt_ms}")
        self.get_logger().info("========================================")

    def info_cb(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self.get_logger().info(
                f"✅ Camera Info Received | frame_id={msg.header.frame_id} "
                f"K=[fx={msg.k[0]:.3f}, fy={msg.k[4]:.3f}, cx={msg.k[2]:.3f}, cy={msg.k[5]:.3f}]"
            )
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

        dt_ms = self._time_diff_ms(msg.header, self.latest_depth_msg.header)
        self._log_time_diff_if_needed(msg.header, self.latest_depth_msg.header)

        if self.skip_if_time_diff_too_large and dt_ms is not None and dt_ms > self.max_color_depth_dt_ms:
            self.get_logger().warn(
                f"⏭️ Skip publish: color-depth dt too large ({dt_ms:.2f} ms > {self.max_color_depth_dt_ms:.2f} ms)"
            )
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if decoded is None:
                self.get_logger().error("❌ Decoding Error: imdecode returned None")
                return
            frame_bgr = decoded
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f"❌ Decoding Error: {e}")
            return

        results = self.model.predict(
            source=frame_rgb,
            imgsz=self.imgsz,
            conf=0.25,
            verbose=False,
            device=self.device,
            half=True
        )
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            if self.frame_count % 30 == 0:
                self.get_logger().info("🔍 Searching for target...")
            if self.debug_image_enable and self._should_publish_debug():
                overlay = frame_bgr.copy()
                cv2.putText(
                    overlay, "NO DETECTION", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
                )
                self._publish_debug_compressed(overlay, msg.header)
            return

        try:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
            clss = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None

            cv_depth = self.bridge.imgmsg_to_cv2(
                self.latest_depth_msg, desired_encoding="passthrough"
            )

            fx = float(self.camera_intrinsics.k[0])
            fy = float(self.camera_intrinsics.k[4])
            cx = float(self.camera_intrinsics.k[2])
            cy = float(self.camera_intrinsics.k[5])

            self.get_logger().info(f"🎯 Detected: {len(boxes)} objects")

            overlay = frame_bgr.copy()
            draw_overlay = self.debug_image_enable and self._should_publish_debug()

            if draw_overlay:
                cv2.putText(
                    overlay,
                    f"Detections: {len(boxes)}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(float, box)
                width_px = x2 - x1
                height_px = y2 - y1
                u = int((x1 + x2) / 2.0)
                v = int((y1 + y2) / 2.0)

                cls_id = int(clss[i]) if clss is not None else -1
                conf_val = float(confs[i]) if confs is not None else -1.0

                class_name = "unknown"
                if cls_id >= 0 and hasattr(self.model, "names"):
                    try:
                        class_name = self.model.names[cls_id]
                    except Exception:
                        class_name = str(cls_id)

                bbox_center_msg = PointStamped()
                bbox_center_msg.header.stamp = msg.header.stamp
                bbox_center_msg.header.frame_id = self.latest_depth_msg.header.frame_id
                bbox_center_msg.point.x = float(u)
                bbox_center_msg.point.y = float(v)
                bbox_center_msg.point.z = 0.0
                self.pub_bbox_center.publish(bbox_center_msg)

                bbox_size_msg = Float32MultiArray()
                bbox_size_msg.data = [float(width_px), float(height_px)]
                self.pub_bbox_size.publish(bbox_size_msg)

                depth_m = self._get_median_depth(
                    cv_depth,
                    self.latest_depth_msg.encoding,
                    u,
                    v
                )

                if depth_m is None:
                    self.get_logger().warn(
                        f"det_id={cls_id:02d} class={class_name} conf={conf_val:.2f} "
                        f"center_px=({u},{v}) bbox_px=({width_px:.1f},{height_px:.1f}) depth=INVALID"
                    )
                    if draw_overlay:
                        self._draw_detection(
                            overlay, x1, y1, x2, y2, u, v,
                            cls_id=cls_id, class_name=class_name, conf_val=conf_val,
                            extra_text="depth=None"
                        )
                    continue

                self.z_history.append(depth_m)

                if len(self.z_history) < self.min_history_size:
                    self.get_logger().info(
                        f"det_id={cls_id:02d} class={class_name} conf={conf_val:.2f} "
                        f"center_px=({u},{v}) bbox_px=({width_px:.1f},{height_px:.1f}) "
                        f"raw_z={depth_m:.3f}m collecting={len(self.z_history)}/{self.min_history_size}"
                    )
                    if draw_overlay:
                        self._draw_detection(
                            overlay, x1, y1, x2, y2, u, v,
                            cls_id=cls_id, class_name=class_name, conf_val=conf_val,
                            extra_text=f"z={depth_m:.2f}"
                        )
                    continue

                z_filtered = self._get_filtered_depth()
                if z_filtered is None:
                    self.get_logger().warn(
                        f"det_id={cls_id:02d} class={class_name} conf={conf_val:.2f} "
                        f"center_px=({u},{v}) filtered_depth=FAILED"
                    )
                    continue

                x_m = (u - cx) * z_filtered / fx
                y_m = (v - cy) * z_filtered / fy

                out_pt = PointStamped()
                out_pt.header.stamp = msg.header.stamp
                out_pt.header.frame_id = self.latest_depth_msg.header.frame_id
                out_pt.point.x = float(x_m)
                out_pt.point.y = float(y_m)
                out_pt.point.z = float(z_filtered)
                self.pub_3d.publish(out_pt)

                real_width_m = float(width_px * z_filtered / fx)
                real_height_m = float(height_px * z_filtered / fy)

                bbox_size_m_msg = Float32MultiArray()
                bbox_size_m_msg.data = [real_width_m, real_height_m]
                self.pub_bbox_size_m.publish(bbox_size_m_msg)

                if self.log_every_detection:
                    self.get_logger().info(
                        f"det_id={cls_id:02d} class={class_name} conf={conf_val:.2f} "
                        f"center_px=({u},{v}) "
                        f"xyz=({x_m:.3f},{y_m:.3f},{z_filtered:.3f})m "
                        f"bbox_px=({width_px:.1f},{height_px:.1f}) "
                        f"bbox_m≈({real_width_m:.3f},{real_height_m:.3f}) | "
                        f"stamp=color, frame={out_pt.header.frame_id}, dt={dt_ms:.2f}ms"
                    )

                if draw_overlay:
                    self._draw_detection(
                        overlay, x1, y1, x2, y2, u, v,
                        cls_id=cls_id, class_name=class_name, conf_val=conf_val,
                        extra_text=f"z={z_filtered:.2f}m"
                    )

            if draw_overlay:
                self._publish_debug_compressed(overlay, msg.header)

        except Exception as e:
            self.get_logger().error(f"❌ Processing Error: {e}")

    def _draw_detection(self, img, x1, y1, x2, y2, u, v,
                        cls_id=-1, class_name="unknown", conf_val=-1.0, extra_text=""):
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(img, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        cv2.circle(img, (u, v), 4, (0, 0, 255), -1)

        label = f"ID:{cls_id} {class_name}"
        if conf_val >= 0.0:
            label += f" C:{conf_val:.2f}"
        if extra_text:
            label += f" {extra_text}"

        text_y = max(20, y1i - 8)
        cv2.putText(
            img,
            label,
            (x1i, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    def _should_publish_debug(self):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.last_debug_pub_time is None:
            self.last_debug_pub_time = now_sec
            return True

        if (now_sec - self.last_debug_pub_time) >= self.debug_publish_period_sec:
            self.last_debug_pub_time = now_sec
            return True

        return False

    def _publish_debug_compressed(self, bgr_img, header):
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.debug_jpeg_quality)]
            ok, enc = cv2.imencode(".jpg", bgr_img, encode_param)
            if not ok:
                self.get_logger().warn("⚠️ Debug image JPEG encode failed")
                return

            msg = CompressedImage()
            msg.header = header
            msg.format = "jpeg"
            msg.data = enc.tobytes()
            self.pub_debug_img.publish(msg)
        except Exception as e:
            self.get_logger().error(f"❌ Debug image publish failed: {e}")

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

            if "32FC1" in encoding:
                valid = patch[np.isfinite(patch) & (patch > 0)]
                return float(np.median(valid)) if valid.size > 0 else None

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
            inliers = vals[np.abs(vals - med) <= self.outlier_thresh_m]

            if inliers.size == 0:
                return float(med)

            return float(np.median(inliers))

        except Exception as e:
            self.get_logger().error(f"Filtered depth error: {e}")
            return None

    def _time_diff_ms(self, color_header, depth_header):
        try:
            color_t = color_header.stamp.sec + color_header.stamp.nanosec * 1e-9
            depth_t = depth_header.stamp.sec + depth_header.stamp.nanosec * 1e-9
            return abs(color_t - depth_t) * 1000.0
        except Exception:
            return None

    def _log_time_diff_if_needed(self, color_header, depth_header):
        try:
            diff_ms = self._time_diff_ms(color_header, depth_header)
            if diff_ms is not None and diff_ms > self.warn_if_time_diff_ms:
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