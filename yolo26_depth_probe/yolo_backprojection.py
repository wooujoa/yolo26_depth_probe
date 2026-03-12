"""
#!/usr/bin/env python3
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge

from ultralytics import YOLO


class YoloPointCloudNode(Node):
    def __init__(self):
        super().__init__("yolo_pointcloud_node")

        # ---- Topics ---- 
        self.color_topic = "/camera/camera/color/image_rect_raw"
        self.depth_topic = "/camera/camera/aligned_depth_to_color/image_raw"
        self.info_topic = "/camera/camera/aligned_depth_to_color/camera_info"
        self.overlay_topic = "/yolo/overlay/image"
        self.pc_topic = "/yolo/target_pc" 
        
        # ---- ROS IO ----
        self.sub_color = self.create_subscription(
            Image, self.color_topic, self.color_cb, qos_profile_sensor_data
        )
        self.sub_depth = self.create_subscription(
            Image, self.depth_topic, self.depth_cb, qos_profile_sensor_data
        )
        self.sub_info = self.create_subscription(
            CameraInfo, self.info_topic, self.info_cb, qos_profile_sensor_data
        )
        
        self.pub_overlay = self.create_publisher(Image, self.overlay_topic, 10)
        self.pub_pc = self.create_publisher(PointCloud2, self.pc_topic, 10) 

        # ---- Model ----
        self.model = YOLO("yolo26n-seg.pt")
        self.bridge = CvBridge()

        # ---- Caches ----
        self.latest_depth_msg = None
        self.camera_intrinsics = None

        # ---- Inference params ----
        self.imgsz = 640
        self.conf = 0.25

        self.get_logger().info("YOLO PointCloud Node Initialized.")

    def info_cb(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = msg
            self.get_logger().info("Camera Intrinsics received.")

    def depth_cb(self, msg: Image):
        self.latest_depth_msg = msg

    def _rosimg_to_rgb(self, msg: Image) -> np.ndarray:
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def color_cb(self, msg: Image):
        if self.latest_depth_msg is None or self.camera_intrinsics is None:
            return

        # 1) Color -> RGB
        try:
            frame_rgb = self._rosimg_to_rgb(msg)
        except Exception as e:
            self.get_logger().error(f"Color convert failed: {e}")
            return

        # 2) YOLO Inference (Uncomment classes=[63] to detect laptops only)
        try:
            results = self.model.predict(
                source=frame_rgb, 
                imgsz=self.imgsz, 
                conf=self.conf, 
                classes=[63],  # laptop
                verbose=False
            )
            r = results[0]
        except Exception as e:
            self.get_logger().error(f"YOLO predict failed: {e}")
            return

        # 3) Publish 2D Overlay for debugging
        try:
            overlay = r.plot()
            out_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
            out_msg.header = msg.header
            self.pub_overlay.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"Overlay publish failed: {e}")

        # 4) Process Masks & Generate PointCloud
        if r.masks is not None and len(r.masks) > 0:
            self._process_masks_to_pc(msg.header, frame_rgb, r.masks.data)

    def _process_masks_to_pc(self, header, frame_rgb, masks_data):
        # Convert depth image
        dmsg = self.latest_depth_msg
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(dmsg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Depth convert failed: {e}")
            return

        # Convert depth to meters (float32)
        if dmsg.encoding == "16UC1":
            cv_depth = cv_depth.astype(np.float32) / 1000.0
        elif dmsg.encoding == "32FC1":
            cv_depth = cv_depth.astype(np.float32)
        else:
            return

        # Combine all detected masks into a single boolean mask
        h, w = frame_rgb.shape[:2]
        combined_mask = np.zeros((h, w), dtype=bool)
        masks_np = masks_data.detach().cpu().numpy()

        for m in masks_np:
            # Resize mask to match original image resolution
            m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            combined_mask |= (m_resized > 0.5)

        # Generate and publish PointCloud2
        pc_msg = self._create_point_cloud_msg(header, cv_depth, frame_rgb, combined_mask)
        if pc_msg is not None:
            self.pub_pc.publish(pc_msg)

    def _create_point_cloud_msg(self, header, depth_m, rgb_img, mask):
        # Camera Intrinsics
        fx = self.camera_intrinsics.k[0]
        cx = self.camera_intrinsics.k[2]
        fy = self.camera_intrinsics.k[4]
        cy = self.camera_intrinsics.k[5]

        # Fast Vectorized Operations (Numpy)
        v, u = np.where(mask)
        z = depth_m[v, u]

        # Filter out zero or invalid depths
        valid = (z > 0) & np.isfinite(z)
        u = u[valid]
        v = v[valid]
        z = z[valid]

        if len(z) == 0:
            return None

        # Deprojection formulas to get X, Y in meters
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Extract RGB colors for each point
        colors = rgb_img[v, u]
        r = colors[:, 0].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        b = colors[:, 2].astype(np.uint32)
        
        # Pack RGB into a single float32 value for PointCloud2
        rgb_packed = (r << 16) | (g << 8) | b
        rgb_float = rgb_packed.view(np.float32)

        # Stack X, Y, Z, RGB into a single N x 4 array
        points = np.column_stack((x, y, z, rgb_float))

        # Define PointCloud2 fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # Ensure PointCloud2 uses the same coordinate frame as depth camera
        header.frame_id = self.latest_depth_msg.header.frame_id
        
        pc2_msg = pc2.create_cloud(header, fields, points)
        return pc2_msg


def main():
    rclpy.init()
    node = YoloPointCloudNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    
"""
#!/usr/bin/env python3
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge

from ultralytics import YOLO


class YoloPointCloudNode(Node):
    def __init__(self):
        super().__init__("yolo_pointcloud_node")

        # ---- Topics ---- 
        """
        self.color_topic = "/camera/camera/color/image_rect_raw"
        self.depth_topic = "/camera/camera/aligned_depth_to_color/image_raw"
        self.info_topic = "/camera/camera/aligned_depth_to_color/camera_info"
        self.overlay_topic = "/yolo/overlay/image"
        self.pc_topic = "/yolo/target_pc" 
        """
        
        # ---- FFW Topics(left기준) ----
        self.color_topic = "/camera_l/camera_l/color/image_rect_raw"
        self.depth_topic = "/camera_l/camera_l/depth/image_rect_raw"
        self.info_topic = "/camera_l/camera_l/color/camera_info" # YOLO(Color) 기준이므로 color의 info를 사용합니다.
        self.overlay_topic = "/yolo/overlay/image"
        self.pc_topic = "/yolo/target_pc"
        
        
        # ---- ROS Subscriptions ----
        self.sub_color = self.create_subscription(
            Image, self.color_topic, self.color_cb, qos_profile_sensor_data
        )
        self.sub_depth = self.create_subscription(
            Image, self.depth_topic, self.depth_cb, qos_profile_sensor_data
        )
        self.sub_info = self.create_subscription(
            CameraInfo, self.info_topic, self.info_cb, qos_profile_sensor_data
        )
        
        # ---- ROS Publishers ----
        self.pub_overlay = self.create_publisher(Image, self.overlay_topic, 10)
        self.pub_pc = self.create_publisher(PointCloud2, self.pc_topic, 10) 

        # ---- Model ----
        # Note: Since we only use bounding boxes now, you could even use a faster
        # detection model (e.g., "yolov8n.pt") instead of a segmentation model.
        self.model = YOLO("yolo26n-seg.pt")
        self.bridge = CvBridge()

        # ---- Caches ----
        self.latest_depth_msg = None
        self.camera_intrinsics = None

        # ---- Inference params ----
        self.imgsz = 640
        self.conf = 0.25
        
        # ---- Expansion Margin (Pixels) ----
        # How much context to include around the detected object for grasp collision avoidance
        self.bbox_expand_margin = 50 

        self.get_logger().info("YOLO PointCloud Node (Expanded BBox Version) Initialized.")

    def info_cb(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = msg
            self.get_logger().info("Camera Intrinsics received.")

    def depth_cb(self, msg: Image):
        self.latest_depth_msg = msg

    def _rosimg_to_rgb(self, msg: Image) -> np.ndarray:
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def color_cb(self, msg: Image):
        if self.latest_depth_msg is None or self.camera_intrinsics is None:
            return

        # 1) Color -> RGB
        try:
            frame_rgb = self._rosimg_to_rgb(msg)
        except Exception as e:
            self.get_logger().error(f"Color convert failed: {e}")
            return

        # 2) YOLO Inference
        try:
            results = self.model.predict(
                source=frame_rgb, 
                imgsz=self.imgsz, 
                conf=self.conf, 
                classes=[39],  # bottle=39, laptop=63
                verbose=False
            )
            r = results[0]
        except Exception as e:
            self.get_logger().error(f"YOLO predict failed: {e}")
            return

        # 3) Publish 2D Overlay for debugging
        try:
            overlay = r.plot()
            out_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
            out_msg.header = msg.header
            self.pub_overlay.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"Overlay publish failed: {e}")

        # 4) Process Expanded Boxes & Generate PointCloud
        if r.boxes is not None and len(r.boxes) > 0:
            self._process_expanded_boxes_to_pc(msg.header, frame_rgb, r.boxes)

    def _process_expanded_boxes_to_pc(self, header, frame_rgb, boxes):
        # Convert depth image
        dmsg = self.latest_depth_msg
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(dmsg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Depth convert failed: {e}")
            return

        # Convert depth to meters (float32)
        if dmsg.encoding == "16UC1":
            cv_depth = cv_depth.astype(np.float32) / 1000.0
        elif dmsg.encoding == "32FC1":
            cv_depth = cv_depth.astype(np.float32)
        else:
            return

        h, w = frame_rgb.shape[:2]
        
        # Create an empty boolean mask
        combined_mask = np.zeros((h, w), dtype=bool)
        xyxy = boxes.xyxy.detach().cpu().numpy()

        # Iterate through all detected boxes and expand them
        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i]
            
            # Expand coordinates by margin, ensuring they stay within image bounds
            exp_x1 = max(0, int(x1) - self.bbox_expand_margin)
            exp_y1 = max(0, int(y1) - self.bbox_expand_margin)
            exp_x2 = min(w, int(x2) + self.bbox_expand_margin)
            exp_y2 = min(h, int(y2) + self.bbox_expand_margin)

            # Set the expanded rectangular region to True
            combined_mask[exp_y1:exp_y2, exp_x1:exp_x2] = True

        # Generate and publish PointCloud2
        pc_msg = self._create_point_cloud_msg(header, cv_depth, frame_rgb, combined_mask)
        if pc_msg is not None:
            self.pub_pc.publish(pc_msg)

    def _create_point_cloud_msg(self, header, depth_m, rgb_img, mask):
        # Camera Intrinsics
        fx = self.camera_intrinsics.k[0]
        cx = self.camera_intrinsics.k[2]
        fy = self.camera_intrinsics.k[4]
        cy = self.camera_intrinsics.k[5]

        # Fast Vectorized Operations (Numpy)
        v, u = np.where(mask)
        z = depth_m[v, u]

        # Filter out zero or invalid depths
        valid = (z > 0) & np.isfinite(z)
        u = u[valid]
        v = v[valid]
        z = z[valid]

        if len(z) == 0:
            return None

        # Deprojection formulas to get X, Y in meters
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Extract RGB colors for each point
        colors = rgb_img[v, u]
        r = colors[:, 0].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        b = colors[:, 2].astype(np.uint32)
        
        # Pack RGB into a single float32 value for PointCloud2
        rgb_packed = (r << 16) | (g << 8) | b
        rgb_float = rgb_packed.view(np.float32)

        # Stack X, Y, Z, RGB into a single N x 4 array
        points = np.column_stack((x, y, z, rgb_float))

        # Define PointCloud2 fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # Ensure PointCloud2 uses the same coordinate frame as depth camera
        header.frame_id = self.latest_depth_msg.header.frame_id
        
        pc2_msg = pc2.create_cloud(header, fields, points)
        return pc2_msg


def main():
    rclpy.init()
    node = YoloPointCloudNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()