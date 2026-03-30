from setuptools import find_packages, setup

package_name = 'yolo26_depth_probe'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jwg',
    maintainer_email='wjddnrud4487@kw.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'yolo26_node = yolo26_depth_probe.yolo_node:main',
        'depth_probe_node = yolo26_depth_probe.depth_probe_node:main',
        'yolo_backprojection = yolo26_depth_probe.yolo_backprojection:main',
        'yolo_depth_debug = yolo26_depth_probe.yolo_depth_debug:main',
        'yolo_depth_debug_r = yolo26_depth_probe.yolo_depth_debug_r:main',
        'yolo_backprojection_local = yolo26_depth_probe.yolo_backprojection_local:main',
        'yolo_backprojection_robot = yolo26_depth_probe.yolo_backprojection_robot:main',
        'yolo_backprojection_robot_r = yolo26_depth_probe.yolo_backprojection_robot_r:main',
        'fullpc_backprojection = yolo26_depth_probe.fullpc_backprojection:main',
        ],
    },
)
