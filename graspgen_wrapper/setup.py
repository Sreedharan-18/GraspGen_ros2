from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'graspgen_wrapper'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install srv and launch
        (os.path.join('share', package_name, 'srv'), glob('srv/*.srv')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='ROS2 wrapper to run GraspGen on camera point clouds and return PoseStamped grasps.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'graspgen_service = graspgen_wrapper.graspgen_service_node:main',
        ],
    },
)
