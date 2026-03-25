#!/usr/bin/env python3
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField, CompressedImage
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge


class FullFramePointCloudNode(Node):
    def __init__(self):
        super().__init__("full_frame_pointcloud_node")

        # =========================
        # Parameters
        # =========================
        self.declare_parameter("use_compressed_color", False)

        self.declare_parameter("color_topic", "/camera/camera/color/image_rect_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("info_topic", "/camera/camera/aligned_depth_to_color/camera_info")

        self.declare_parameter("overlay_topic", "/yolo/overlay/image/compressed")
        self.declare_parameter("pc_topic", "/yolo/target_pc")   # 기존 grasp 노드와 동일

        self.declare_parameter("pc_publish_period_sec", 0.5)
        self.declare_parameter("debug_publish_period_sec", 2.0)
        self.declare_parameter("debug_jpeg_quality", 40)

        # full-frame filtering
        self.declare_parameter("pixel_stride", 4)          # 1이면 전 픽셀, 2~6 정도 추천
        self.declare_parameter("min_depth_m", 0.20)
        self.declare_parameter("max_depth_m", 0.35)

        self.use_compressed_color = bool(self.get_parameter("use_compressed_color").value)

        self.color_topic = str(self.get_parameter("color_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.info_topic = str(self.get_parameter("info_topic").value)

        self.overlay_topic = str(self.get_parameter("overlay_topic").value)
        self.pc_topic = str(self.get_parameter("pc_topic").value)

        self.pc_publish_period_sec = float(self.get_parameter("pc_publish_period_sec").value)
        self.debug_publish_period_sec = float(self.get_parameter("debug_publish_period_sec").value)
        self.debug_jpeg_quality = int(self.get_parameter("debug_jpeg_quality").value)

        self.pixel_stride = int(self.get_parameter("pixel_stride").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        # =========================
        # ROS IO
        # =========================
        self.bridge = CvBridge()

        if self.use_compressed_color:
            self.sub_color = self.create_subscription(
                CompressedImage,
                self.color_topic,
                self.color_cb_compressed,
                qos_profile_sensor_data
            )
        else:
            self.sub_color = self.create_subscription(
                Image,
                self.color_topic,
                self.color_cb_raw,
                qos_profile_sensor_data
            )

        self.sub_depth = self.create_subscription(
            Image, self.depth_topic, self.depth_cb, qos_profile_sensor_data
        )
        self.sub_info = self.create_subscription(
            CameraInfo, self.info_topic, self.info_cb, qos_profile_sensor_data
        )

        self.pub_overlay = self.create_publisher(CompressedImage, self.overlay_topic, 3)
        self.pub_pc = self.create_publisher(PointCloud2, self.pc_topic, 10)

        # =========================
        # Cache / State
        # =========================
        self.latest_depth_msg = None
        self.camera_intrinsics = None
        self.frame_count = 0
        self.last_debug_pub_time = None
        self.last_pc_pub_time = None

        self.get_logger().info("========================================")
        self.get_logger().info(f"use_compressed_color     : {self.use_compressed_color}")
        self.get_logger().info(f"color_topic              : {self.color_topic}")
        self.get_logger().info(f"depth_topic              : {self.depth_topic}")
        self.get_logger().info(f"info_topic               : {self.info_topic}")
        self.get_logger().info(f"overlay_topic            : {self.overlay_topic}")
        self.get_logger().info(f"pc_topic                 : {self.pc_topic}")
        self.get_logger().info(f"pixel_stride             : {self.pixel_stride}")
        self.get_logger().info(f"min_depth_m              : {self.min_depth_m}")
        self.get_logger().info(f"max_depth_m              : {self.max_depth_m}")
        self.get_logger().info(f"pc_publish_period_sec    : {self.pc_publish_period_sec}")
        self.get_logger().info("========================================")

    # =========================
    # Callbacks
    # =========================
    def info_cb(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = msg
            self.get_logger().info(
                f"✅ Camera Info Received | frame_id={msg.header.frame_id} "
                f"K=[fx={msg.k[0]:.3f}, fy={msg.k[4]:.3f}, cx={msg.k[2]:.3f}, cy={msg.k[5]:.3f}]"
            )

    def depth_cb(self, msg: Image):
        self.latest_depth_msg = msg

    def color_cb_raw(self, msg: Image):
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"❌ Raw color decode failed: {e}")
            return
        self._process_color_frame(msg.header, frame_bgr)

    def color_cb_compressed(self, msg: CompressedImage):
        try:
            frame_bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"❌ Compressed color decode failed: {e}")
            return
        self._process_color_frame(msg.header, frame_bgr)

    def _process_color_frame(self, header, frame_bgr):
        self.frame_count += 1

        if self.camera_intrinsics is None:
            if self.frame_count % 30 == 0:
                self.get_logger().warn("⚠️ Waiting for Camera Info...")
            return

        if self.latest_depth_msg is None:
            if self.frame_count % 30 == 0:
                self.get_logger().warn("⚠️ Waiting for Depth Image...")
            return

        self._log_time_diff_if_needed(header, self.latest_depth_msg)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._should_publish_debug():
            try:
                overlay = frame_bgr.copy()
                h, w = overlay.shape[:2]
                text = f"FULL FRAME PC | stride={self.pixel_stride} | depth=[{self.min_depth_m:.2f},{self.max_depth_m:.2f}]m"
                cv2.putText(
                    overlay, text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
                )
                cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 255, 0), 2)
                self._publish_debug_compressed(overlay, header)
            except Exception as e:
                self.get_logger().error(f"❌ Overlay publish failed: {e}")

        self._publish_full_frame_pc(header, frame_rgb)

    # =========================
    # Main processing
    # =========================
    def _publish_full_frame_pc(self, header, frame_rgb):
        dmsg = self.latest_depth_msg

        try:
            cv_depth = self.bridge.imgmsg_to_cv2(dmsg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"❌ Depth convert failed: {e}")
            return

        if dmsg.encoding == "16UC1":
            depth_m = cv_depth.astype(np.float32) / 1000.0
        elif dmsg.encoding == "32FC1":
            depth_m = cv_depth.astype(np.float32)
        else:
            self.get_logger().error(f"❌ Unsupported depth encoding: {dmsg.encoding}")
            return

        build_result = self._build_full_frame_points(depth_m, frame_rgb)
        if build_result is None:
            self.get_logger().warn("[pc] Full-frame point cloud build failed")
            return

        pc_msg = self._create_point_cloud_msg_from_points(header, build_result)
        if pc_msg is None:
            self.get_logger().warn("[pc] PointCloud message not created, publish skipped")
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.last_pc_pub_time is not None:
            dt = now_sec - self.last_pc_pub_time
            if dt < self.pc_publish_period_sec:
                self.get_logger().info(
                    f"[pc] publish throttled: dt={dt:.3f}s < {self.pc_publish_period_sec:.3f}s"
                )
                return

        self.pub_pc.publish(pc_msg)
        self.last_pc_pub_time = now_sec
        self.get_logger().info(f"📦 Published FULL-FRAME point cloud: {pc_msg.width} points")

    def _build_full_frame_points(self, depth_m, rgb_img):
        fx = float(self.camera_intrinsics.k[0])
        fy = float(self.camera_intrinsics.k[4])
        cx = float(self.camera_intrinsics.k[2])
        cy = float(self.camera_intrinsics.k[5])

        h, w = depth_m.shape[:2]
        stride = max(1, self.pixel_stride)

        vv, uu = np.mgrid[0:h:stride, 0:w:stride]
        z = depth_m[0:h:stride, 0:w:stride]

        valid = (
            np.isfinite(z)
            & (z > self.min_depth_m)
            & (z < self.max_depth_m)
        )

        if not np.any(valid):
            self.get_logger().warn("[pc] No valid full-frame depth points")
            return None

        u = uu[valid].astype(np.float32)
        v = vv[valid].astype(np.float32)
        z = z[valid].astype(np.float32)

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        xyz = np.column_stack((x, y, z)).astype(np.float32)

        colors = rgb_img[0:h:stride, 0:w:stride][valid]

        self.get_logger().info(
            f"[pc] full-frame valid points={xyz.shape[0]} "
            f"(stride={stride}, depth_range={self.min_depth_m:.2f}~{self.max_depth_m:.2f}m)"
        )

        return {
            "xyz": xyz,
            "colors": colors,
        }

    def _create_point_cloud_msg_from_points(self, header, cluster_result):
        xyz = cluster_result["xyz"]
        colors = cluster_result["colors"]

        if xyz.shape[0] == 0:
            self.get_logger().warn("[pc] No points after selection")
            return None

        r = colors[:, 0].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        b = colors[:, 2].astype(np.uint32)

        rgb_uint32 = (r << 16) | (g << 8) | b
        rgb_float = rgb_uint32.view(np.float32)

        points = np.column_stack((xyz[:, 0], xyz[:, 1], xyz[:, 2], rgb_float)).tolist()

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        out_header = header
        out_header.frame_id = self.latest_depth_msg.header.frame_id
        return pc2.create_cloud(out_header, fields, points)

    # =========================
    # Debug helpers
    # =========================
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
            self.pub_overlay.publish(msg)
        except Exception as e:
            self.get_logger().error(f"❌ Debug image publish failed: {e}")

    def _log_time_diff_if_needed(self, color_header, depth_msg: Image):
        try:
            color_t = color_header.stamp.sec + color_header.stamp.nanosec * 1e-9
            depth_t = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9
            diff_ms = abs(color_t - depth_t) * 1000.0

            if diff_ms > 100.0:
                self.get_logger().warn(
                    f"⚠️ Color/Depth timestamp diff is large: {diff_ms:.2f} ms"
                )
        except Exception as e:
            self.get_logger().warn(f"⚠️ Timestamp check failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FullFramePointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()