#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
import math
import numpy as np
from sensor_msgs.msg import NavSatFix


# =========================
# GPS Mapping (Method B)
# =========================

GPS_LATLON = np.array([
    [37.6558868, 128.6757588],  # Point1 (P1)
    [37.6557475, 128.6758443],  # Point2
    [37.6557181, 128.6757395],  # Point3
    [37.6558802, 128.6756559],  # Point4
], dtype=float)

# Candidate zone vertices in Gazebo WORLD ENU (x=E, y=N)
SIM_EN_WORLD = np.array([
    [-5.0, 26.3226159],   # A_world (Point1)  <-- P1 target
    [18.5, 26.3226159],   # B_world (Point2)
    [23.0, 40.9948284],   # C_world (Point3)
    [-5.0, 40.9948284],   # D_world (Point4)
], dtype=float)


def latlon_to_EN(lat: float, lon: float, lat0: float, lon0: float):
    R = 111111.0
    east = (lon - lon0) * R * math.cos(math.radians(lat0))
    north = (lat - lat0) * R
    return east, north


def EN_to_latlon(east: float, north: float, lat0: float, lon0: float):
    R = 111111.0
    lat = lat0 + (north / R)
    lon = lon0 + (east / (R * math.cos(math.radians(lat0))))
    return lat, lon


def fit_affine_EN_to_EN(sim_EN: np.ndarray, gps_latlon: np.ndarray):
    lat0 = float(gps_latlon[0, 0])
    lon0 = float(gps_latlon[0, 1])
    gps_EN = np.array([latlon_to_EN(lat, lon, lat0, lon0) for lat, lon in gps_latlon], dtype=float)

    X = np.hstack([sim_EN, np.ones((sim_EN.shape[0], 1))])
    M, *_ = np.linalg.lstsq(X, gps_EN, rcond=None)
    return M, lat0, lon0


def sim_local_NE_to_latlon(north_m: float, east_m: float, M, lat0: float, lon0: float):
    simE = float(east_m)
    simN = float(north_m)
    gps_EN_pred = np.array([simE, simN, 1.0]) @ M
    return EN_to_latlon(float(gps_EN_pred[0]), float(gps_EN_pred[1]), lat0, lon0)


class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('offboard_control_go_to_p1_hover')
        print("### AMRL's KROS Drone ###")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # [GPS publish]
        self.sim_navsat_pub = self.create_publisher(
            NavSatFix, '/sim/navsat_fix', qos_profile)

        # Subscribers (v1 topics)
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',
            self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1',
            self.vehicle_status_callback, qos_profile)

        # State
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()

        # Flight params
        self.takeoff_height = -10.0  # NED: negative up
        self.wp_reached_xy = 0.8    # meters

        # Target: P1 (GPS Point1) mapped to SIM world vertex A
        # SIM_EN_WORLD uses (E, N). PX4 local uses (N, E).
        p1_e = float(SIM_EN_WORLD[0, 0])  # -5.0
        p1_n = float(SIM_EN_WORLD[0, 1])  # 26.3226159
        self.target_n = p1_n
        self.target_e = p1_e

        # GPS mapping (affine fit)
        self.affine_M, self.lat0, self.lon0 = fit_affine_EN_to_EN(SIM_EN_WORLD, GPS_LATLON)
        self.get_logger().info(f"\n[GPS Mapping] Affine M:\n{self.affine_M}\nlat0={self.lat0}, lon0={self.lon0}")

        # Timer
        self.timer = self.create_timer(0.1, self.timer_callback)

    def vehicle_local_position_callback(self, vehicle_local_position):
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        self.vehicle_status = vehicle_status

    def arm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float):
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 1.57079
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")

    def publish_vehicle_command(self, command, **params) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def publish_sim_navsat_fix(self):
        n = float(self.vehicle_local_position.x)  # North
        e = float(self.vehicle_local_position.y)  # East
        lat, lon = sim_local_NE_to_latlon(n, e, self.affine_M, self.lat0, self.lon0)

        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = 0.0
        self.sim_navsat_pub.publish(msg)

    def _xy_reached(self, target_n: float, target_e: float) -> bool:
        n = float(self.vehicle_local_position.x)
        e = float(self.vehicle_local_position.y)
        dist = math.sqrt((target_n - n) ** 2 + (target_e - e) ** 2)
        return dist <= self.wp_reached_xy

    def timer_callback(self) -> None:
        self.publish_offboard_control_heartbeat_signal()

        # Engage offboard + arm after some setpoints
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        # Publish mapped GPS if local_position is valid
        if getattr(self.vehicle_local_position, "timestamp", 0) != 0:
            self.publish_sim_navsat_fix()

        # -----------------------------
        # State machine:
        # 1) takeoff/hover to -5m at origin
        # 2) go to P1 (Point1) and hover
        # -----------------------------
        if self.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            # keep sending takeoff setpoint until offboard is active
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)
        else:
            # If not yet reached takeoff altitude, keep hovering at origin
            if float(self.vehicle_local_position.z) > self.takeoff_height + 0.3:
                self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)
            else:
                # Go to P1 and hover there
                self.publish_position_setpoint(self.target_n, self.target_e, self.takeoff_height)

                # Optional: log when reached (still keeps publishing same setpoint = hover)
                if self._xy_reached(self.target_n, self.target_e):
                    self.get_logger().info("Reached P1. Hovering...")

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1


def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)





















# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
# from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
# import math
# import numpy as np
# from sensor_msgs.msg import NavSatFix


# class OffboardControl(Node):
#     """Node for controlling a vehicle in offboard mode."""

#     def __init__(self) -> None:
#         super().__init__('offboard_control_takeoff_and_land')
#         print("### AMRL's KROS Drone ###")
#         # Configure QoS profile for publishing and subscribing
#         qos_profile = QoSProfile(
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             durability=DurabilityPolicy.TRANSIENT_LOCAL,
#             history=HistoryPolicy.KEEP_LAST,
#             depth=1
#         )

#         # Create publishers
#         self.offboard_control_mode_publisher = self.create_publisher(
#             OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
#         self.trajectory_setpoint_publisher = self.create_publisher(
#             TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
#         self.vehicle_command_publisher = self.create_publisher(
#             VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

#         # Create subscribers
#         self.vehicle_local_position_subscriber = self.create_subscription(
#             VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
#         self.vehicle_status_subscriber = self.create_subscription(
#             VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)

#         # Initialize variables
#         self.offboard_setpoint_counter = 0
#         self.vehicle_local_position = VehicleLocalPosition()
#         self.vehicle_status = VehicleStatus()
#         self.takeoff_height = -5.0

#         # Create a timer to publish control commands
#         self.timer = self.create_timer(0.1, self.timer_callback)

#     def vehicle_local_position_callback(self, vehicle_local_position):
#         """Callback function for vehicle_local_position topic subscriber."""
#         self.vehicle_local_position = vehicle_local_position

#     def vehicle_status_callback(self, vehicle_status):
#         """Callback function for vehicle_status topic subscriber."""
#         self.vehicle_status = vehicle_status

#     def arm(self):
#         """Send an arm command to the vehicle."""
#         self.publish_vehicle_command(
#             VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
#         self.get_logger().info('Arm command sent')

#     def disarm(self):
#         """Send a disarm command to the vehicle."""
#         self.publish_vehicle_command(
#             VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
#         self.get_logger().info('Disarm command sent')

#     def engage_offboard_mode(self):
#         """Switch to offboard mode."""
#         self.publish_vehicle_command(
#             VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
#         self.get_logger().info("Switching to offboard mode")

#     def land(self):
#         """Switch to land mode."""
#         self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
#         self.get_logger().info("Switching to land mode")

#     def publish_offboard_control_heartbeat_signal(self):
#         """Publish the offboard control mode."""
#         msg = OffboardControlMode()
#         msg.position = True
#         msg.velocity = False
#         msg.acceleration = False
#         msg.attitude = False
#         msg.body_rate = False
#         msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
#         self.offboard_control_mode_publisher.publish(msg)

#     def publish_position_setpoint(self, x: float, y: float, z: float):
#         """Publish the trajectory setpoint."""
#         msg = TrajectorySetpoint()
#         msg.position = [x, y, z]
#         msg.yaw = 1.57079  # (90 degree)
#         msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
#         #self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")
#         self.trajectory_setpoint_publisher.publish(msg)
#         self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")

#     def publish_vehicle_command(self, command, **params) -> None:
#         """Publish a vehicle command."""
#         msg = VehicleCommand()
#         msg.command = command
#         msg.param1 = params.get("param1", 0.0)
#         msg.param2 = params.get("param2", 0.0)
#         msg.param3 = params.get("param3", 0.0)
#         msg.param4 = params.get("param4", 0.0)
#         msg.param5 = params.get("param5", 0.0)
#         msg.param6 = params.get("param6", 0.0)
#         msg.param7 = params.get("param7", 0.0)
#         msg.target_system = 1
#         msg.target_component = 1
#         msg.source_system = 1
#         msg.source_component = 1
#         msg.from_external = True
#         msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
#         self.vehicle_command_publisher.publish(msg)

#     def timer_callback(self) -> None:
#         """Callback function for the timer."""
#         self.publish_offboard_control_heartbeat_signal()

#         if self.offboard_setpoint_counter == 10:
#             self.engage_offboard_mode()
#             self.arm()
        
#         #if self.vehicle_local_position.z > self.takeoff_height and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
#         self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)

#         if self.vehicle_local_position.z <= self.takeoff_height:
#             self.land()
#             exit(0)

#         if self.offboard_setpoint_counter < 11:
#             self.offboard_setpoint_counter += 1


# def main(args=None) -> None:
#     print('Starting offboard control node...')
#     rclpy.init(args=args)
#     offboard_control = OffboardControl()
#     rclpy.spin(offboard_control)
#     offboard_control.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     try:
#         main()
#     except Exception as e:
#         print(e)















