#!/usr/bin/env python3
import os
import time
from typing import List, Tuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from builtin_interfaces.msg import Time as RosTime

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from graspgen_wrapper.srv import PlanGrasp

# ROS TF
import tf2_ros
from tf2_ros.transform_listener import TransformListener

# PointCloud2 → numpy
from sensor_msgs_py import point_cloud2 as pc2

# GraspGen core
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps
from grasp_gen.robot import get_gripper_info

# Math helpers (we rely on trimesh.transformations, already in GraspGen deps)
import trimesh
import trimesh.transformations as tra


def now_stamp(node: Node) -> RosTime:
    # Use node clock (sim / use_sim_time compatible)
    return node.get_clock().now().to_msg()


def pc2_to_xyz(msg: PointCloud2) -> Tuple[np.ndarray, str]:
    """Convert PointCloud2 → (N,3) float32 and return frame_id."""
    gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    pts = np.fromiter(gen, dtype=np.float32).reshape(-1, 3)
    return pts, msg.header.frame_id or ""


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0.0 or points.size == 0:
        return points
    keys = np.floor(points / voxel)
    # Unique rows
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def build_robotiq_85_proxy() -> trimesh.Trimesh:
    """
    Conservative envelope mesh for 2F-85:
    a single box representing the gripper 'swept' volume (palm + fingers).
    This is purposefully conservative to err on removing risky grasps.
    Dimensions (meters): width=0.085, height=0.04, depth=0.12
    """
    box = trimesh.creation.box(extents=[0.085, 0.04, 0.12])
    # Center at origin, approach along +Z (convention typical for GraspGen meshes)
    return box


def pose_from_matrix(T: np.ndarray, frame_id: str, stamp: RosTime) -> PoseStamped:
    ps = PoseStamped()
    ps.header.stamp = stamp
    ps.header.frame_id = frame_id
    ps.pose.position.x = float(T[0, 3])
    ps.pose.position.y = float(T[1, 3])
    ps.pose.position.z = float(T[2, 3])
    q = tra.quaternion_from_matrix(T)  # returns [w, x, y, z]
    ps.pose.orientation.w = float(q[0])
    ps.pose.orientation.x = float(q[1])
    ps.pose.orientation.y = float(q[2])
    ps.pose.orientation.z = float(q[3])
    return ps


def transform_matrix(T_child_parent: np.ndarray, T_pose_in_child: np.ndarray) -> np.ndarray:
    """Compose T_parent_pose = T_parent_child @ T_child_pose."""
    return T_child_parent @ T_pose_in_child


class GraspGenService(Node):
    def __init__(self):
        super().__init__("graspgen_service")

        # ---------- Parameters ----------
        self.declare_parameter("pointcloud_topic", "/wrist_mounted_camera/points")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("models_dir", "/models")
        self.declare_parameter(
            "gripper_config",
            "/models/checkpoints/graspgen_robotiq_2f_140.yml",
        )
        self.declare_parameter("num_grasps", 200)
        self.declare_parameter("default_topk", 10)
        self.declare_parameter("default_threshold", 0.80)
        self.declare_parameter("filter_collisions_default", True)
        self.declare_parameter("collision_threshold_default", 0.02)
        self.declare_parameter("max_scene_points", 60000)
        self.declare_parameter("voxel_size", 0.0075)  # 7.5 mm
        self.declare_parameter("do_clustering_default", False)
        # Collision gripper selection: 'auto' (from YAML), 'robotiq_2f_85', 'robotiq_2f_140'
        self.declare_parameter("collision_gripper", "auto")

        self.pointcloud_topic = self.get_parameter("pointcloud_topic").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.models_dir = self.get_parameter("models_dir").get_parameter_value().string_value
        self.gripper_config = self.get_parameter("gripper_config").get_parameter_value().string_value
        self.num_grasps = self.get_parameter("num_grasps").get_parameter_value().integer_value
        self.default_topk = self.get_parameter("default_topk").get_parameter_value().integer_value
        self.default_threshold = self.get_parameter("default_threshold").get_parameter_value().double_value
        self.filter_collisions_default = self.get_parameter("filter_collisions_default").get_parameter_value().bool_value
        self.collision_threshold_default = self.get_parameter("collision_threshold_default").get_parameter_value().double_value
        self.max_scene_points = self.get_parameter("max_scene_points").get_parameter_value().integer_value
        self.voxel_size = self.get_parameter("voxel_size").get_parameter_value().double_value
        self.do_clustering_default = self.get_parameter("do_clustering_default").get_parameter_value().bool_value
        self.collision_gripper = self.get_parameter("collision_gripper").get_parameter_value().string_value

        # ---------- GraspGen config ----------
        self.get_logger().info(f"Loading GraspGen config: {self.gripper_config}")
        cfg = load_grasp_cfg(self.gripper_config)
        # Make sure checkpoints are resolvable even if paths are relative
        cfg_dir = os.path.dirname(self.gripper_config)
        def _rebase(p):
            if p is None:
                return None
            if os.path.isabs(p):
                return p
            # common case: checkpoints live under <models_dir>/checkpoints
            cand = os.path.join(self.models_dir, p)
            if os.path.exists(cand):
                return cand
            cand = os.path.join(cfg_dir, p)
            return cand

        if "eval" in cfg and "checkpoint" in cfg.eval:
            cfg.eval.checkpoint = _rebase(cfg.eval.checkpoint)
        if "discriminator" in cfg and "checkpoint" in cfg.discriminator:
            cfg.discriminator.checkpoint = _rebase(cfg.discriminator.checkpoint)

        self.gripper_name = str(cfg.data.gripper_name)
        self.get_logger().info(f"Gripper (YAML): {self.gripper_name}")

        self.sampler = GraspGenSampler(cfg)
        self.get_logger().info("GraspGenSampler ready.")

        # Collision mesh
        self.gripper_collision_mesh = None
        if self.collision_gripper.lower() == "robotiq_2f_85":
            self.get_logger().warn("Using conservative 2F-85 proxy collision mesh.")
            self.gripper_collision_mesh = build_robotiq_85_proxy()
        else:
            try:
                info = get_gripper_info(self.gripper_name if self.collision_gripper == "auto"
                                        else self.collision_gripper)
                self.gripper_collision_mesh = info.collision_mesh
            except Exception as e:
                self.get_logger().warn(
                    f"Could not load collision mesh for '{self.gripper_name}'; "
                    f"falling back to 2F-85 proxy. Err: {e}"
                )
                self.gripper_collision_mesh = build_robotiq_85_proxy()

        # ---------- Subscriptions ----------
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.latest_cloud: Optional[PointCloud2] = None
        self.create_subscription(PointCloud2, self.pointcloud_topic, self._cloud_cb, qos)
        self.get_logger().info(f"Subscribed to: {self.pointcloud_topic}")

        # ---------- TF ----------
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------- Service ----------
        self.srv = self.create_service(PlanGrasp, "graspgen/plan", self._plan_cb)
        self.get_logger().info("Service ready: /graspgen/plan")

        # ---------- Debug publishers ----------
        self.pub_best_cam = self.create_publisher(PoseStamped, "graspgen/best_grasp_cam", 1)
        self.pub_best_base = self.create_publisher(PoseStamped, "graspgen/best_grasp_base", 1)

    # ----------------- Callbacks -----------------

    def _cloud_cb(self, msg: PointCloud2):
        self.latest_cloud = msg

    def _plan_cb(self, req: PlanGrasp.Request, res: PlanGrasp.Response) -> PlanGrasp.Response:
        start = time.time()

        if self.latest_cloud is None:
            res.success = False
            res.message = "No point cloud received yet."
            return res

        # Parameters from request (with node defaults)
        topk = req.topk if req.topk > 0 else self.default_topk
        threshold = req.threshold if req.threshold != 0.0 else self.default_threshold
        do_collisions = req.filter_collisions if req.filter_collisions else self.filter_collisions_default
        collision_thresh = req.collision_threshold if req.collision_threshold != 0.0 else self.collision_threshold_default
        do_cluster = req.do_clustering if req.do_clustering else self.do_clustering_default

        # Convert cloud
        scene_xyz, cloud_frame = pc2_to_xyz(self.latest_cloud)
        if scene_xyz.size == 0:
            res.success = False
            res.message = "Empty point cloud."
            return res

        # Keep the scene (downsample for collision filtering)
        scene_xyz_ds = voxel_downsample(scene_xyz, self.voxel_size)
        if self.max_scene_points > 0 and scene_xyz_ds.shape[0] > self.max_scene_points:
            idx = np.random.choice(scene_xyz_ds.shape[0], self.max_scene_points, replace=False)
            scene_xyz_ds = scene_xyz_ds[idx]

        # Object cloud selection
        if do_cluster:
            # Simple DBSCAN clustering (no floor removal) — robust enough in sim
            try:
                from sklearn.cluster import DBSCAN
                eps = max(0.015, self.voxel_size * 3.0)
                db = DBSCAN(eps=eps, min_samples=300).fit(scene_xyz_ds)
                labels = db.labels_
                clusters = []
                for k in set(labels):
                    if k == -1:
                        continue
                    seg = scene_xyz_ds[labels == k]
                    if seg.shape[0] >= 300:
                        clusters.append(seg)
                obj_clouds = clusters if clusters else [scene_xyz_ds]
            except Exception as e:
                self.get_logger().warn(f"Clustering failed ({e}); falling back to whole cloud.")
                obj_clouds = [scene_xyz_ds]
        else:
            obj_clouds = [scene_xyz_ds]

        all_grasps_cam: List[np.ndarray] = []
        all_scores: List[float] = []

        # Generate grasps for each object cloud
        for obj_pc in obj_clouds:
            try:
                grasps_t, scores_t = GraspGenSampler.run_inference(
                    obj_pc,
                    self.sampler,
                    grasp_threshold=threshold,
                    num_grasps=self.num_grasps,
                    topk_num_grasps=topk
                )
                if grasps_t is None or len(grasps_t) == 0:
                    continue
                grasps = grasps_t.cpu().numpy() if hasattr(grasps_t, "cpu") else np.asarray(grasps_t)
                scores = scores_t.cpu().numpy() if hasattr(scores_t, "cpu") else np.asarray(scores_t)

                # Homogeneous guard
                if grasps.ndim == 3 and grasps.shape[1:] == (4, 4):
                    grasps[:, 3, 3] = 1.0

                # Collision filter (scene vs gripper mesh)
                if do_collisions and self.gripper_collision_mesh is not None:
                    mask = filter_colliding_grasps(
                        scene_pc=scene_xyz_ds,
                        grasp_poses=grasps,
                        gripper_collision_mesh=self.gripper_collision_mesh,
                        collision_threshold=collision_thresh,
                    )
                    grasps = grasps[mask]
                    scores = scores[mask]

                # Append
                for g, s in zip(grasps, scores):
                    all_grasps_cam.append(g)
                    all_scores.append(float(s))
            except Exception as e:
                self.get_logger().warn(f"GraspGen inference failed on a cluster: {e}")

        if not all_grasps_cam:
            res.success = False
            res.message = "No grasps found."
            return res

        # Sort by score desc and keep topk across all clusters
        order = np.argsort(-np.asarray(all_scores))
        order = order[:topk]
        all_grasps_cam = [all_grasps_cam[i] for i in order]
        all_scores = [all_scores[i] for i in order]

        stamp = now_stamp(self)
        # Build PoseStamped (camera frame)
        cam_frame = cloud_frame if cloud_frame else "camera"
        grasps_cam_ps = [pose_from_matrix(G, cam_frame, stamp) for G in all_grasps_cam]

        # Transform to base_frame
        grasps_base_ps: List[PoseStamped] = []
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame, cam_frame, rclpy.time.Time()
            )
            # Build T_base_cam
            t = tf_msg.transform.translation
            q = tf_msg.transform.rotation
            T_base_cam = tra.quaternion_matrix([q.w, q.x, q.y, q.z])
            T_base_cam[:3, 3] = [t.x, t.y, t.z]

            for G in all_grasps_cam:
                T_base = transform_matrix(T_base_cam, G)
                grasps_base_ps.append(pose_from_matrix(T_base, self.base_frame, stamp))
        except Exception as e:
            self.get_logger().warn(f"TF transform {cam_frame}→{self.base_frame} failed: {e}")
            # Fallback: return only camera-frame (empty base list)
            grasps_base_ps = []

        # Publish best grasp for convenience
        self.pub_best_cam.publish(grasps_cam_ps[0])
        if grasps_base_ps:
            self.pub_best_base.publish(grasps_base_ps[0])

        # Fill response
        res.success = True
        res.message = f"ok: {len(grasps_cam_ps)} grasps (in {time.time()-start:.2f}s)"
        res.grasps_cam = grasps_cam_ps
        res.grasps_base = grasps_base_ps
        res.scores = [float(s) for s in all_scores]
        return res


def main():
    rclpy.init()
    node = GraspGenService()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
