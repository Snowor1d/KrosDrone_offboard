from setuptools import setup

package_name = 'trajectory_planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hoon',
    maintainer_email='tg42008@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'offboard_control_py = trajectory_planning.offboard_control:main',
            'hovering = trajectory_planning.Hovering:main',
            'waypoint1 = trajectory_planning.Waypoint1:main',
            'waypoint2 = trajectory_planning.Waypoint2:main',
        ],
    },
)
