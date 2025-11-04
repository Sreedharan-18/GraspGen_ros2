from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("pointcloud_topic", default_value="/wrist_mounted_camera/points"),
        DeclareLaunchArgument("base_frame", default_value="base_link"),
        DeclareLaunchArgument("models_dir", default_value="/ros2_ws/src/GraspGen_ros2/GraspGenModels"),
        DeclareLaunchArgument("gripper_config", default_value="/ros2_ws/src/GraspGen_ros2/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml"),
        DeclareLaunchArgument("num_grasps", default_value="200"),
        DeclareLaunchArgument("default_topk", default_value="10"),
        DeclareLaunchArgument("default_threshold", default_value="0.8"),
        DeclareLaunchArgument("filter_collisions_default", default_value="true"),
        DeclareLaunchArgument("collision_threshold_default", default_value="0.02"),
        DeclareLaunchArgument("max_scene_points", default_value="60000"),
        DeclareLaunchArgument("voxel_size", default_value="0.0075"),
        DeclareLaunchArgument("do_clustering_default", default_value="false"),
        DeclareLaunchArgument("collision_gripper", default_value="auto"),
        Node(
            package="graspgen_wrapper",
            executable="graspgen_service",
            name="graspgen_service",
            output="screen",
            parameters=[{
                "pointcloud_topic": LaunchConfiguration("pointcloud_topic"),
                "base_frame": LaunchConfiguration("base_frame"),
                "models_dir": LaunchConfiguration("models_dir"),
                "gripper_config": LaunchConfiguration("gripper_config"),
                "num_grasps": LaunchConfiguration("num_grasps"),
                "default_topk": LaunchConfiguration("default_topk"),
                "default_threshold": LaunchConfiguration("default_threshold"),
                "filter_collisions_default": LaunchConfiguration("filter_collisions_default"),
                "collision_threshold_default": LaunchConfiguration("collision_threshold_default"),
                "max_scene_points": LaunchConfiguration("max_scene_points"),
                "voxel_size": LaunchConfiguration("voxel_size"),
                "do_clustering_default": LaunchConfiguration("do_clustering_default"),
                "collision_gripper": LaunchConfiguration("collision_gripper"),
            }]
        )
    ])
