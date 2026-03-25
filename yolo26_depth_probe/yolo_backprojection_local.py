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


class YoloPointCloudNode(Node):
    def __init__(self):
        super().__init__("yolo_pointcloud_node")

        # =========================
        # GPU / Model
        # =========================
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolo26n-seg.pt").to(self.device)
        self.get_logger().info(f"🚀 YOLO started on: {self.device}")

        # =========================
        # Parameters
        # =========================
        self.declare_parameter("use_compressed_color", False)

        self.declare_parameter("color_topic", "/camera/camera/color/image_rect_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("info_topic", "/camera/camera/aligned_depth_to_color/camera_info")

        self.declare_parameter("overlay_topic", "/yolo/overlay/image/compressed")
        self.declare_parameter("pc_topic", "/yolo/target_pc")

        self.declare_parameter("target_class", 39)   # bottle=39, laptop=63
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)

        # segmentation 후처리
        self.declare_parameter("mask_min_pixels", 300)
        self.declare_parameter("mask_erode_iters", 0)
        self.declare_parameter("mask_dilate_iters", 0)
        self.declare_parameter("min_component_pixels", 200)

        # 3D cluster filtering
        self.declare_parameter("cluster_radius_m", 0.025)
        self.declare_parameter("min_cluster_points", 120)
        self.declare_parameter("front_percentile", 10.0)

        self.declare_parameter("debug_publish_period_sec", 2.0)
        self.declare_parameter("debug_jpeg_quality", 40)
        self.declare_parameter("pc_publish_period_sec", 0.5)

        self.use_compressed_color = bool(self.get_parameter("use_compressed_color").value)

        self.color_topic = str(self.get_parameter("color_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.info_topic = str(self.get_parameter("info_topic").value)

        self.overlay_topic = str(self.get_parameter("overlay_topic").value)
        self.pc_topic = str(self.get_parameter("pc_topic").value)

        self.target_class = int(self.get_parameter("target_class").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)

        self.mask_min_pixels = int(self.get_parameter("mask_min_pixels").value)
        self.mask_erode_iters = int(self.get_parameter("mask_erode_iters").value)
        self.mask_dilate_iters = int(self.get_parameter("mask_dilate_iters").value)
        self.min_component_pixels = int(self.get_parameter("min_component_pixels").value)

        self.cluster_radius_m = float(self.get_parameter("cluster_radius_m").value)
        self.min_cluster_points = int(self.get_parameter("min_cluster_points").value)
        self.front_percentile = float(self.get_parameter("front_percentile").value)

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
        self.get_logger().info(f"target_class             : {self.target_class}")
        self.get_logger().info(f"imgsz                    : {self.imgsz}")
        self.get_logger().info(f"conf                     : {self.conf}")
        self.get_logger().info(f"mask_min_pixels          : {self.mask_min_pixels}")
        self.get_logger().info(f"mask_erode_iters         : {self.mask_erode_iters}")
        self.get_logger().info(f"mask_dilate_iters        : {self.mask_dilate_iters}")
        self.get_logger().info(f"min_component_pixels     : {self.min_component_pixels}")
        self.get_logger().info(f"cluster_radius_m         : {self.cluster_radius_m}")
        self.get_logger().info(f"min_cluster_points       : {self.min_cluster_points}")
        self.get_logger().info(f"front_percentile         : {self.front_percentile}")
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

        if self._should_publish_debug():
            try:
                overlay = r.plot()
                self._publish_debug_compressed(overlay, header)
            except Exception as e:
                self.get_logger().error(f"❌ Overlay publish failed: {e}")

        if r.boxes is None or len(r.boxes) == 0:
            if self.frame_count % 30 == 0:
                self.get_logger().info("🔍 Searching for target...")
            return

        self.get_logger().info(f"🎯 Detected: {len(r.boxes)} objects")
        self._process_segmentation_to_pc(header, frame_rgb, r)

    # =========================
    # Main processing
    # =========================
    def _process_segmentation_to_pc(self, header, frame_rgb, result):
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

        seg_mask = self._build_union_segmentation_mask(result, frame_rgb.shape[:2])
        if seg_mask is None:
            self.get_logger().warn("[pc] Segmentation mask not created, publish skipped")
            return

        mask_pixels = int(np.count_nonzero(seg_mask))
        self.get_logger().info(f"[pc] segmentation mask pixels={mask_pixels}")

        cluster_result = self._build_frontmost_cluster_points(depth_m, frame_rgb, seg_mask)
        if cluster_result is None:
            self.get_logger().warn("[pc] No valid front-most cluster found, publish skipped")
            return

        pc_msg = self._create_point_cloud_msg_from_points(header, cluster_result)
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
        self.get_logger().info(f"📦 Published target point cloud: {pc_msg.width} points")

    def _build_union_segmentation_mask(self, result, image_shape_hw):
        h, w = image_shape_hw

        if result.masks is None or result.masks.data is None:
            self.get_logger().warn("[pc] No segmentation masks in YOLO result")
            return None

        try:
            boxes_cls = result.boxes.cls.detach().cpu().numpy().astype(int)
            masks = result.masks.data.detach().cpu().numpy()
        except Exception as e:
            self.get_logger().error(f"❌ Failed to parse segmentation masks: {e}")
            return None

        if masks.ndim != 3 or masks.shape[0] == 0:
            self.get_logger().warn("[pc] Empty segmentation mask tensor")
            return None

        union_mask = np.zeros((h, w), dtype=np.uint8)
        target_count = 0

        for i in range(len(boxes_cls)):
            if boxes_cls[i] != self.target_class:
                continue

            raw_mask = masks[i]
            if raw_mask.dtype != np.uint8:
                raw_mask = (raw_mask > 0.5).astype(np.uint8)

            resized_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            union_mask = np.maximum(union_mask, resized_mask)
            target_count += 1

        self.get_logger().info(f"[pc] target masks used={target_count}")

        if target_count == 0:
            self.get_logger().warn("[pc] No masks matched target class")
            return None

        if self.mask_erode_iters > 0:
            kernel = np.ones((3, 3), np.uint8)
            union_mask = cv2.erode(union_mask, kernel, iterations=self.mask_erode_iters)

        if self.mask_dilate_iters > 0:
            kernel = np.ones((3, 3), np.uint8)
            union_mask = cv2.dilate(union_mask, kernel, iterations=self.mask_dilate_iters)

        # connected component filtering
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(union_mask, connectivity=8)
        filtered_mask = np.zeros_like(union_mask)

        kept_components = 0
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_component_pixels:
                continue
            filtered_mask[labels == label] = 1
            kept_components += 1

        self.get_logger().info(f"[pc] kept 2D components={kept_components}")

        mask_pixels = int(np.count_nonzero(filtered_mask))
        if mask_pixels < self.mask_min_pixels:
            self.get_logger().warn(
                f"[pc] Segmentation mask too small after component filter: {mask_pixels} < {self.mask_min_pixels}"
            )
            return None

        return filtered_mask.astype(bool)

    def _build_frontmost_cluster_points(self, depth_m, rgb_img, mask):
        fx = float(self.camera_intrinsics.k[0])
        fy = float(self.camera_intrinsics.k[4])
        cx = float(self.camera_intrinsics.k[2])
        cy = float(self.camera_intrinsics.k[5])

        v, u = np.where(mask)
        if len(v) == 0:
            self.get_logger().warn("[pc] Empty segmentation region")
            return None

        z_all = depth_m[v, u]
        self.get_logger().info(f"[pc] candidate depth pixels={len(z_all)}")

        valid = (z_all > 0.0) & np.isfinite(z_all)

        u = u[valid]
        v = v[valid]
        z = z_all[valid]

        self.get_logger().info(f"[pc] valid depth pixels={len(z)}")

        if len(z) == 0:
            self.get_logger().warn("[pc] No valid depth in segmentation region")
            return None

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        xyz = np.column_stack((x, y, z)).astype(np.float32)
        colors = rgb_img[v, u]

        cluster_labels = self._radius_cluster_labels(xyz, self.cluster_radius_m)
        if cluster_labels is None:
            self.get_logger().warn("[pc] 3D clustering failed")
            return None

        unique_labels = np.unique(cluster_labels)
        unique_labels = unique_labels[unique_labels >= 0]

        if len(unique_labels) == 0:
            self.get_logger().warn("[pc] No 3D clusters found")
            return None

        candidates = []
        for label in unique_labels:
            idx = np.where(cluster_labels == label)[0]
            if len(idx) < self.min_cluster_points:
                continue

            cluster_xyz = xyz[idx]
            cluster_z = cluster_xyz[:, 2]

            front_score = float(np.percentile(cluster_z, self.front_percentile))
            mean_z = float(np.mean(cluster_z))

            candidates.append({
                "label": int(label),
                "indices": idx,
                "size": int(len(idx)),
                "front_score": front_score,
                "mean_z": mean_z,
            })

        self.get_logger().info(f"[pc] valid 3D clusters={len(candidates)}")

        if len(candidates) == 0:
            self.get_logger().warn("[pc] No 3D clusters passed min_cluster_points")
            return None

        # front-most meaningful cluster
        candidates.sort(key=lambda c: (c["front_score"], c["mean_z"], -c["size"]))
        best = candidates[0]

        self.get_logger().info(
            f"[pc] selected cluster label={best['label']} size={best['size']} "
            f"front_score={best['front_score']:.4f} mean_z={best['mean_z']:.4f}"
        )

        idx = best["indices"]
        selected_xyz = xyz[idx]
        selected_colors = colors[idx]

        return {
            "xyz": selected_xyz,
            "colors": selected_colors,
        }

    def _radius_cluster_labels(self, xyz, radius):
        n = xyz.shape[0]
        if n == 0:
            return None

        labels = -np.ones(n, dtype=np.int32)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0
        radius2 = radius * radius

        # O(N^2) but okay for current object-only clouds
        for i in range(n):
            if visited[i]:
                continue

            visited[i] = True
            queue = [i]
            labels[i] = cluster_id

            while queue:
                cur = queue.pop()
                diff = xyz - xyz[cur]
                dist2 = np.einsum("ij,ij->i", diff, diff)
                neighbors = np.where((dist2 <= radius2) & (~visited))[0]

                if len(neighbors) > 0:
                    visited[neighbors] = True
                    labels[neighbors] = cluster_id
                    queue.extend(neighbors.tolist())

            cluster_id += 1

        return labels

    def _create_point_cloud_msg_from_points(self, header, cluster_result):
        xyz = cluster_result["xyz"]
        colors = cluster_result["colors"]

        if xyz.shape[0] == 0:
            self.get_logger().warn("[pc] No points after cluster selection")
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
    node = YoloPointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()