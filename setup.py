from setuptools import setup

package_name = 'offboard'

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
            'offboard_control_py = offboard.offboard_control:main',
            'hovering = offboard.Hovering:main',
            'waypoint1 = offboard.Waypoint1:main',
            'waypoint2 = offboard.Waypoint2:main',
            'waypoint3 = offboard.Waypoint3:main',
            'waypoint4 = offboard.Waypoint4:main'
        ],
    },
)
