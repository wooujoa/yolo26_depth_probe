#!/usr/bin/env python3
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField, CompressedImage
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge

from ultralytics import YOLO
import torch


class YoloBackprojectionNode(Node):
    def __init__(self):
        super().__init__("yolo_backprojection")

        # =========================
        # GPU / Model
        # =========================
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolo26n-seg.pt").to(self.device)
        self.get_logger().info(f"🚀 YOLO started on: {self.device}")

        # =========================
        # Parameters
        # =========================
        # 네 기존 코드 기준으로 /camera/camera/... 로 맞춤
        self.declare_parameter("use_compressed_color", False)

        self.declare_parameter("color_topic", "/camera_l/camera_l/color/image_rect_raw/compressed")
        self.declare_parameter("depth_topic", "/camera_l/camera_l/aligned_depth_to_color/image_raw")
        self.declare_parameter("info_topic", "/camera_l/camera_l/aligned_depth_to_color/camera_info")

        self.declare_parameter("overlay_topic", "/yolo/overlay/image/compressed")

        # grasp 입력용 ROI 전체 point cloud
        self.declare_parameter("pc_topic", "/yolo/target_pc")

        # 분리된 point cloud
        self.declare_parameter("obj_pc_topic", "/yolo/object_pc")
        self.declare_parameter("bg_pc_topic", "/yolo/background_pc")

        self.declare_parameter("target_class", 39)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)

        # bbox expansion
        self.declare_parameter("bbox_expand_ratio", 0.20)
        self.declare_parameter("bbox_expand_min_px", 20)

        # point cloud generation
        self.declare_parameter("pixel_stride", 2)
        self.declare_parameter("min_depth_m", 0.10)
        self.declare_parameter("max_depth_m", 1.20)

        # depth band around object median depth
        self.declare_parameter("use_depth_band", True)
        self.declare_parameter("depth_band_margin_m", 0.12)

        self.declare_parameter("min_roi_points", 120)
        self.declare_parameter("min_object_points", 80)

        self.declare_parameter("debug_publish_period_sec", 2.0)
        self.declare_parameter("debug_jpeg_quality", 40)
        self.declare_parameter("pc_publish_period_sec", 0.5)

        self.use_compressed_color = bool(self.get_parameter("use_compressed_color").value)

        self.color_topic = str(self.get_parameter("color_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.info_topic = str(self.get_parameter("info_topic").value)

        self.overlay_topic = str(self.get_parameter("overlay_topic").value)
        self.pc_topic = str(self.get_parameter("pc_topic").value)
        self.obj_pc_topic = str(self.get_parameter("obj_pc_topic").value)
        self.bg_pc_topic = str(self.get_parameter("bg_pc_topic").value)

        self.target_class = int(self.get_parameter("target_class").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)

        self.bbox_expand_ratio = float(self.get_parameter("bbox_expand_ratio").value)
        self.bbox_expand_min_px = int(self.get_parameter("bbox_expand_min_px").value)

        self.pixel_stride = int(self.get_parameter("pixel_stride").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        self.use_depth_band = bool(self.get_parameter("use_depth_band").value)
        self.depth_band_margin_m = float(self.get_parameter("depth_band_margin_m").value)

        self.min_roi_points = int(self.get_parameter("min_roi_points").value)
        self.min_object_points = int(self.get_parameter("min_object_points").value)

        self.debug_publish_period_sec = float(self.get_parameter("debug_publish_period_sec").value)
        self.debug_jpeg_quality = int(self.get_parameter("debug_jpeg_quality").value)
        self.pc_publish_period_sec = float(self.get_parameter("pc_publish_period_sec").value)

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
        self.pub_obj_pc = self.create_publisher(PointCloud2, self.obj_pc_topic, 10)
        self.pub_bg_pc = self.create_publisher(PointCloud2, self.bg_pc_topic, 10)

        # =========================
        # Cache / State
        # =========================
        self.latest_depth_msg = None
        self.camera_intrinsics = None
        self.frame_count = 0
        self.last_debug_pub_time = None
        self.last_pc_pub_time = None

        self.get_logger().info("========================================")
        self.get_logger().info(f"node_name               : {self.get_name()}")
        self.get_logger().info(f"use_compressed_color    : {self.use_compressed_color}")
        self.get_logger().info(f"color_topic             : {self.color_topic}")
        self.get_logger().info(f"depth_topic             : {self.depth_topic}")
        self.get_logger().info(f"info_topic              : {self.info_topic}")
        self.get_logger().info(f"overlay_topic           : {self.overlay_topic}")
        self.get_logger().info(f"pc_topic                : {self.pc_topic}")
        self.get_logger().info(f"obj_pc_topic            : {self.obj_pc_topic}")
        self.get_logger().info(f"bg_pc_topic             : {self.bg_pc_topic}")
        self.get_logger().info(f"target_class            : {self.target_class}")
        self.get_logger().info(f"imgsz                   : {self.imgsz}")
        self.get_logger().info(f"conf                    : {self.conf}")
        self.get_logger().info(f"bbox_expand_ratio       : {self.bbox_expand_ratio}")
        self.get_logger().info(f"bbox_expand_min_px      : {self.bbox_expand_min_px}")
        self.get_logger().info(f"pixel_stride            : {self.pixel_stride}")
        self.get_logger().info(f"depth_range             : [{self.min_depth_m:.2f}, {self.max_depth_m:.2f}] m")
        self.get_logger().info(f"use_depth_band          : {self.use_depth_band}")
        self.get_logger().info(f"depth_band_margin_m     : {self.depth_band_margin_m}")
        self.get_logger().info(f"pc_publish_period_sec   : {self.pc_publish_period_sec}")
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

        try:
            results = self.model.predict(
                source=frame_rgb,
                imgsz=self.imgsz,
                conf=self.conf,
                classes=[self.target_class],
                verbose=False,
                device=self.device,
                half=(self.device != "cpu"),
            )
            r = results[0]
        except Exception as e:
            self.get_logger().error(f"❌ YOLO predict failed: {e}")
            return

        overlay = frame_bgr.copy()
        if self._should_publish_debug():
            try:
                overlay = r.plot()
            except Exception:
                pass

        if r.boxes is None or len(r.boxes) == 0:
            if self._should_publish_debug():
                self._publish_debug_compressed(overlay, header)
            if self.frame_count % 30 == 0:
                self.get_logger().info("🔍 Searching for target...")
            return

        self.get_logger().info(f"🎯 Detected: {len(r.boxes)} objects")
        self._process_detection_to_pc(header, frame_rgb, overlay, r)

    # =========================
    # Main processing
    # =========================
    def _process_detection_to_pc(self, header, frame_rgb, overlay_bgr, result):
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

        h, w = frame_rgb.shape[:2]

        try:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
        except Exception as e:
            self.get_logger().error(f"❌ Failed to parse boxes: {e}")
            return

        roi_mask = np.zeros((h, w), dtype=bool)

        for box in xyxy:
            x1, y1, x2, y2 = box
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)

            mx = max(self.bbox_expand_min_px, int(self.bbox_expand_ratio * bw))
            my = max(self.bbox_expand_min_px, int(self.bbox_expand_ratio * bh))

            ex1 = max(0, int(x1) - mx)
            ey1 = max(0, int(y1) - my)
            ex2 = min(w, int(x2) + mx)
            ey2 = min(h, int(y2) + my)

            roi_mask[ey1:ey2, ex1:ex2] = True
            cv2.rectangle(overlay_bgr, (ex1, ey1), (ex2, ey2), (0, 255, 255), 2)

        obj_mask = None
        if result.masks is not None and result.masks.data is not None:
            try:
                masks_np = result.masks.data.detach().cpu().numpy()
                obj_mask = np.zeros((h, w), dtype=bool)
                for m in masks_np:
                    m_bin = (m > 0.5).astype(np.uint8)
                    if m_bin.shape[0] != h or m_bin.shape[1] != w:
                        m_bin = cv2.resize(m_bin, (w, h), interpolation=cv2.INTER_NEAREST)
                    obj_mask |= (m_bin > 0)
            except Exception as e:
                self.get_logger().warn(f"⚠️ Failed to parse segmentation mask: {e}")
                obj_mask = None

        if obj_mask is None:
            obj_mask = np.zeros((h, w), dtype=bool)

        obj_in_roi = obj_mask & roi_mask
        bg_in_roi = (~obj_mask) & roi_mask

        if self.use_depth_band:
            obj_depths = depth_m[obj_in_roi]
            obj_depths = obj_depths[
                np.isfinite(obj_depths)
                & (obj_depths > self.min_depth_m)
                & (obj_depths < self.max_depth_m)
            ]
            if obj_depths.size > 0:
                z_med = float(np.median(obj_depths))
                z_lo = max(self.min_depth_m, z_med - self.depth_band_margin_m)
                z_hi = min(self.max_depth_m, z_med + self.depth_band_margin_m)

                band_mask = np.isfinite(depth_m) & (depth_m >= z_lo) & (depth_m <= z_hi)
                roi_mask = roi_mask & band_mask
                obj_in_roi = obj_in_roi & band_mask
                bg_in_roi = bg_in_roi & band_mask

                cv2.putText(
                    overlay_bgr,
                    f"depth band: [{z_lo:.2f}, {z_hi:.2f}] m",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        roi_cloud = self._build_cloud_from_mask(header, depth_m, frame_rgb, roi_mask)
        obj_cloud = self._build_cloud_from_mask(header, depth_m, frame_rgb, obj_in_roi)
        bg_cloud = self._build_cloud_from_mask(header, depth_m, frame_rgb, bg_in_roi)

        roi_n = 0 if roi_cloud is None else roi_cloud.width
        obj_n = 0 if obj_cloud is None else obj_cloud.width
        bg_n = 0 if bg_cloud is None else bg_cloud.width

        self.get_logger().info(f"[pc] roi={roi_n}, object={obj_n}, background={bg_n}")

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.last_pc_pub_time is not None:
            dt = now_sec - self.last_pc_pub_time
            if dt < self.pc_publish_period_sec:
                self.get_logger().info(
                    f"[pc] publish throttled: dt={dt:.3f}s < {self.pc_publish_period_sec:.3f}s"
                )
                if self._should_publish_debug():
                    self._publish_debug_compressed(overlay_bgr, header)
                return

        if roi_cloud is not None and roi_n >= self.min_roi_points:
            self.pub_pc.publish(roi_cloud)

        if obj_cloud is not None and obj_n >= self.min_object_points:
            self.pub_obj_pc.publish(obj_cloud)

        if bg_cloud is not None and bg_n > 0:
            self.pub_bg_pc.publish(bg_cloud)

        if roi_cloud is not None and roi_n >= self.min_roi_points:
            self.last_pc_pub_time = now_sec
            self.get_logger().info("📦 Published ROI/Object/Background point clouds")
        else:
            self.get_logger().warn("⚠️ ROI point cloud too small, publish skipped")

        if self._should_publish_debug():
            self._publish_debug_compressed(overlay_bgr, header)

    def _build_cloud_from_mask(self, header, depth_m, rgb_img, mask):
        fx = float(self.camera_intrinsics.k[0])
        fy = float(self.camera_intrinsics.k[4])
        cx = float(self.camera_intrinsics.k[2])
        cy = float(self.camera_intrinsics.k[5])

        h, w = depth_m.shape[:2]
        stride = max(1, self.pixel_stride)

        mask_ds = mask[0:h:stride, 0:w:stride]
        depth_ds = depth_m[0:h:stride, 0:w:stride]
        rgb_ds = rgb_img[0:h:stride, 0:w:stride]

        vv, uu = np.mgrid[0:h:stride, 0:w:stride]

        valid = (
            mask_ds
            & np.isfinite(depth_ds)
            & (depth_ds > self.min_depth_m)
            & (depth_ds < self.max_depth_m)
        )

        if not np.any(valid):
            return None

        u = uu[valid].astype(np.float32)
        v = vv[valid].astype(np.float32)
        z = depth_ds[valid].astype(np.float32)

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        colors = rgb_ds[valid]
        r = colors[:, 0].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        b = colors[:, 2].astype(np.uint32)

        rgb_uint32 = (r << 16) | (g << 8) | b
        rgb_float = rgb_uint32.view(np.float32)

        points = np.column_stack((x, y, z, rgb_float)).tolist()

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
    node = YoloBackprojectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()