# Build base image
FROM nvcr.io/nvidia/pytorch:23.07-py3 AS base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash","-lc"]

# tmux is for debugging, osmesa is for rendering. Put all apt-get installs in this line
RUN apt update && apt-get install -y --no-install-recommends \
    tmux libosmesa6-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# ---------- Python deps already in your original file ----------
# (Pin numpy before anything that might try to pull NumPy 2.x)
RUN pip install "numpy==1.26.4"

# Install dependencies
RUN pip install h5py hydra-core matplotlib meshcat scikit-learn scipy tensorboard "trimesh==4.5.3"

# Install scene_synthesizer
RUN pip install "scene-synthesizer[recommend]"

# Torch stack (keep exactly as you had it)
RUN pip install torch==2.1.0 torchvision

# Install torch_cluster
RUN pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# OpenCV note:
#   Your previous build pulled NumPy 2.x via opencv-python and broke other deps.
#   Use headless wheel that stays compatible with NumPy 1.26.
RUN pip install imageio pickle5 opencv-python-headless==4.10.0.84 python-fcl

# Install pyrender (+ compatible PyOpenGL)
RUN pip install "pyrender==0.1.45" "pyglet==2.1.6" \
 && pip install "PyOpenGL==3.1.5"

WORKDIR /ros2_ws/src/GraspGen_ros2/GraspGen
# Install pointnet2 modules  **NO BUILD ISOLATION**  (fixes "No module named 'torch'")
COPY pointnet2_ops ./pointnet2_ops
RUN pip install --no-build-isolation ./pointnet2_ops

# Diffusion dependencies
RUN pip install "diffusers==0.11.1" "timm==1.0.15"
RUN pip install "huggingface-hub==0.25.2"

# PointTransformerV3 dependencies (optional; keep as you had them)
RUN pip install addict "yapf==0.40.1" tensorboardx sharedarray torch-geometric
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
RUN pip install spconv-cu120

# For the analytic model
RUN pip install "qpsolvers[clarabel]"

# This was not installed before
RUN pip install "yourdfpy==0.0.56"

# Not sure why this is needed again
RUN pip install "trimesh==4.5.3" "timm==1.0.15"

# For dataset management and manifold
RUN pip install webdataset \
 && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
 && apt-get update && apt-get install -y --no-install-recommends git-lfs \
 && rm -rf /var/lib/apt/lists/*
RUN pip install "objaverse==0.1.7"
RUN mkdir -p /install/ && cd /install/ && git clone --recursive -j8 https://github.com/hjwdzh/Manifold.git
RUN mkdir -p /install/Manifold/build && cd /install/Manifold/build \
 && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"$(nproc)"
ENV PATH="${PATH}:/install/Manifold/build/"

# ---------- ADD ROS 2 HUMBLE (Jammy) ----------
# Keys & repo
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
  | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu \
  $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  > /etc/apt/sources.list.d/ros2.list

# Core + CycloneDDS + common msgs + tools for building your wrapper later
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-std-msgs \
    ros-humble-std-srvs \
    ros-humble-tf2-ros-py \
    ros-humble-tf2-geometry-msgs \
    ros-humble-moveit-msgs \
    python3-colcon-common-extensions \
    python3-argcomplete \
 && rm -rf /var/lib/apt/lists/*

RUN source /opt/ros/humble/setup.bash
ENV ROS_DISTRO=humble
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# Helpful defaults (you can override at run time)
ENV ROS_DOMAIN_ID=0
ENV ROS_LOCALHOST_ONLY=0

WORKDIR /ros2_ws/src/GraspGen_ros2/GraspGen
RUN pip install -e .
# A tiny entrypoint that sources ROS env for interactive shells
COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]

