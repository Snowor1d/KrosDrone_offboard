from setuptools import find_packages, setup

package_name = 'offboard'

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
    maintainer='snowor1d',
    maintainer_email='437travel@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        	'offboard1 = offboard.offboard_control:main',
        	'waypoint = offboard.offboard_waypoint:main',
        	'waypoint1 = offboard.offboard_waypoint1:main',
        	'delivery = offboard.offboard_delivery:main',
        	'lidar = offboard.offboard_control_lidar_based:main',
        	'hovering = offboard.offboard_hovering:main',
        	'real = offboard.offboard_control_real:main',
        	'real2 = offboard.offboard_control_real2:main',
        	'real3 = offboard.offboard_control_real3:main',
            'real4 = offboard.offboard_control_real4:main',
            'real5 = offboard.offboard_control_real5:main',
            'real6 = offboard.offboard_control_real6:main',
            'sim2 = offboard.offboard_control_sim2:main',
        	'sim3 = offboard.offboard_control_sim3:main',
        	'sim4 = offboard.offboard_control_sim4:main',
        	'sim5 = offboard.offboard_control_sim5:main',
            'sim6 = offboard.offboard_control_sim6:main',
        ],
    },
)
