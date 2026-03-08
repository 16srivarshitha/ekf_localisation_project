from setuptools import setup

package_name = 'ekf_package'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # Required by ament so colcon can find the package
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='EKF Localization with Landmark Observations',
    license='MIT',
    entry_points={
        'console_scripts': [
            # ros2 run ekf_package ekf_node
            'ekf_node = ekf_package.ekf_node:main',
            # ros2 run ekf_package simulation_publisher
            'simulation_publisher = ekf_package.simulation_publisher:main',
        ],
    },
)