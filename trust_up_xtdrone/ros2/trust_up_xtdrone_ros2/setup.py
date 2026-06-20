from setuptools import setup

package_name = "trust_up_xtdrone_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/trust_up_ros2.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ysdeng",
    maintainer_email="ysdeng@example.com",
    description="ROS2 adapter for TRUST-UP target pursuit.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "trust_up_ros2_node = trust_up_xtdrone_ros2.trust_up_ros2_node:main",
        ],
    },
)
