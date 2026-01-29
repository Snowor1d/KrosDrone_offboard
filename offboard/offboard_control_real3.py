#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, VehicleOdometry, VehicleGlobalPosition
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import PointStamped
import math
import numpy as np
import os, time
from datetime import datetime
LINE = "line_2"
VELOCITY = 2
FLOATING_SPEED = 0.5
MOVING_SPEED = 1.5
NAN = float('nan')
TAKEOFF_HEIGHT = -6
MAIN_GPS_low_left  =  {"lat" : 37.293149, "lon" : 126.974744,    "alt" : 50.0}
MAIN_GPS_low_right =  {"lat" : 37.293353, "lon" : 126.974786,    "alt" : 50.0}
MAIN_GPS_high_left =  {"lat" : 37.293196, "lon" : 126.974514,    "alt" : 50.0}
MAIN_GPS_high_right = {"lat" : 37.293394, "lon" : 126.974565,    "alt" : 50.0}

# MAIN_GPS_low_left  =  {"lat" : 37.65612320390357, "lon" : 128.67570247043663,    "alt" : 50.0}
# MAIN_GPS_low_right =  {"lat" : 37.6561236, "lon" : 128.6759682,    "alt" : 50.0}
# MAIN_GPS_high_left =  {"lat" : 37.6562558, "lon" : 128.6757019,    "alt" : 50.0}
# MAIN_GPS_high_right = {"lat" : 37.6562558, "lon" : 128.6760189,    "alt" : 50.0}
# lat: 37.65588681286121
# lon: 128.67575855717305
# ref_lat: 37.655886600647314
# ref_lon: 128.67575873828986

MAIN_POINT_GPS = {
    "low_left" : MAIN_GPS_low_left,
    "low_right" : MAIN_GPS_low_right,
    "high_left" : MAIN_GPS_high_left,
    "high_right" : MAIN_GPS_high_right
}
def wrap_to_pi(a: float) -> float:
    """wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def angle_diff(target: float, current: float) -> float:
    """target - current in [-pi, pi]."""
    return wrap_to_pi(target - current)

def euler_from_quaternion(w, x, y, z): 
    t0 = +2 * (w*x+y*z)
    t1 = +1 - 2*(x*x+y*y)
    t3 = +2*(w*z+x*y)
    t4 = +1-2*(y*y+z*z)
    yaw_z = math.atan2(t3, t4)
    return yaw_z

def yaw_to_next_xy(x: float, y: float, next_x: float, next_y:float, eps:float=1e-6) -> float:
    dx = float(next_x - x)
    dy = float(next_y - y)
    yaw = math.atan2(dy, dx)
    return yaw

# =========================
# GPS -> NED 변환 유틸 (WGS84)
# =========================
_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3

def _deg2rad(d): return d * math.pi / 180.0

def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    lat = _deg2rad(lat_deg)
    lon = _deg2rad(lon_deg)
    s = math.sin(lat)
    c = math.cos(lat)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * s*s)

    x = (N + alt_m) * c * math.cos(lon)
    y = (N + alt_m) * c * math.sin(lon)
    z = (N * (1.0 - _WGS84_E2) + alt_m) * s
    return np.array([x, y, z], dtype=float)

def ecef_to_ned(ecef, ref_lat_deg, ref_lon_deg, ref_alt_m):
    ref_ecef = geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    d = ecef - ref_ecef

    lat = _deg2rad(ref_lat_deg)
    lon = _deg2rad(ref_lon_deg)

    sL = math.sin(lat); cL = math.cos(lat)
    sO = math.sin(lon); cO = math.cos(lon)

    # ECEF->NED 회전
    R = np.array([
        [-sL*cO, -sL*sO,  cL],
        [   -sO,     cO, 0.0],
        [-cL*cO, -cL*sO, -sL]
    ], dtype=float)

    ned = R @ d
    return ned  # [north, east, down]

def lla_to_ned(lat_deg, lon_deg, alt_m, ref_lat_deg, ref_lon_deg, ref_alt_m):
    ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    return ecef_to_ned(ecef, ref_lat_deg, ref_lon_deg, ref_alt_m)


# MAIN_WAYPOINT = {"x" : 1, "y" : 0, "z" : 0}

# #NED 기준. yaw방향은 시작위치 -> MAIN_WAYPOINT 방향이 0.0.
# WAYPOINTS = [
#     {"x": 0,  "y": 0,  "z": TAKEOFF_HEIGHT, "yaw": 0.0,              "speed": 1.0, "stop_seconds": 0.1},
#     {"x": 2,  "y": 0,  "z": TAKEOFF_HEIGHT, "yaw": math.radians(0),  "speed": 0.3, "stop_seconds": 0.1},
#     {"x": 3,  "y": 3,  "z": TAKEOFF_HEIGHT, "yaw": math.radians(45), "speed": 0.8, "stop_seconds": 10},
#     {"x": 0,  "y": 5,  "z": TAKEOFF_HEIGHT, "yaw": math.radians(90), "speed": 1.5, "stop_seconds": 0.1},
#     {"x": -5, "y": 0,  "z": TAKEOFF_HEIGHT, "yaw": math.radians(180),"speed": 2.0, "stop_seconds": 0.1},
#     {"x": -5, "y": 0,  "z": 0,              "yaw": math.radians(180),"speed": 1.0, "stop_seconds": 0.0},
# ]

#WAYPOINTS = [[0, 0, -2, 1, ], [5, 0, -2, 1], [5, 5, -2, 1], [0, 5, -2, 1], [0, 0, -2, 1], [0, 0, -0.1, 1]] #[x, y, z, yaw_mode, deisred_speed]



class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')
        print("### AMRL's KROS Drone ###")
        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.vehicle_odom_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odom_callback, qos_profile)
        

        self.vehicle_global_subscriber = self.create_subscription(
            VehicleGlobalPosition, '/fmu/out/vehicle_global_position',
            self.vehicle_global_callback, qos_profile)
        self.subscription = self.create_subscription(
            PointStamped,
            '/rescue/target_pose_global',  # 구독할 토픽 이름
            self.target_listener_callback,
            10  # QoS Depth
        )
        self.gimbal_subscriber = self.create_subscription(
            Bool,
            '/camera/gimbal_cmd',
            self.gimbal_callback,
            10
        )
        

        self.lidar_height = float('nan')
        self.delivery_open_flag = False
        self.delivery_open_flag_pub = self.create_publisher(
            Bool, 'delivery_open_flag', 10
        )
        
        self.lidar_height_sub = self.create_subscription(
            LaserScan, 'lidar_height', self.lidar_height_callback, 10
        )

        self.delivery_open = False

        # Initialize variable
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.vehicle_odom = VehicleOdometry()
        self.vehicle_global = VehicleGlobalPosition()


        self.takeoff_height = TAKEOFF_HEIGHT
        self.dt = 0.1
        self.is_new_go = 0
        self.is_departed = 0
        self.is_near = 0
        self.wait_in_waypoint = 0
        self.previous_yaw = 0.0
        self.now_yaw = 0
        self.is_land = 0
        self.stop_start_time = None
        self.is_stopping = False
        self.stage_finished = 0
        self.yaw_hold = 0
        self.gimbal = 0
        
        #PX4 local origin(GPS)
        self.ref_lat = None
        self.ref_lon = None
        self.ref_alt = None

        self.init_x = 0
        self.init_y = 0
        self.init_z = 0
        self.init_yaw = 0.0

        self.target_x = None
        self.target_y = None
        self.target_z = None
        self.real_target_x = None
        self.real_target_y = None
        self.real_target_z = None
        self.gimbal_70_target_x = None
        self.gimbal_70_target_y = None
        self.target_catched = 0

        #MAIN GPS -> NED
        self.main_pts_ned = {}

        self.mission_built = False
        self.mission_state = "BOOT"
        self.plan = []
        self.plan_idx = 0
        self.is_stopping = False
        self.stop_start_time = None
        self.hold_active = False
        self.hold_x = self.hold_y = self.hold_z = 0.0
        self.hold_yaw = 0.0
        self.ref_yaw = 0.0


        self.init_ready = False
        self.init_ready_global = False
        self.offboard_engaged = False
        

        #
        # self.goto_phase = "ALIGN"
        # self.goto_goal = None
        # self.yaw_target = 0.0
        # self.yaw_hold_cnt = 0
        # self.dist_i = 0.0
        # self.prev_v_cmd = 0.0
        self.prev_v_vec = np.zeros(3, dtype=float)
        self.waypoint_range = 0.3
        # self.waypoint_num = len(WAYPOINTS)
        # self.waypoint_count = 0
        # self.takeoff = 0

        self.yaw_rate_limit = 1 #rad/s
        self.yaw_tol = math.radians(15) #정렬 허용오차
        self.yaw_hold_ticks = 5
        self.yaw_hold_valid = False
        
        self.desired_speed = MOVING_SPEED
        self.kP_dist = 0.8 #(m/s)/m
        self.kI_dist = 0.05 #(m/s)/(m*s)
        self.dist_i_limit = 2.0
        self.v_min = 0 # m/s
        self.slow_radius = 4 #이 안에서 부드럽게 감속
        self.v_slew = 1.5 #속도 코긔 변화 제한
        self.vec_slow = 2.0 #vx, vy, vz 변화 제한

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.1, self.timer_callback)

        
        self.hold_active = False

        self.last_lidar_time = None
        self.lidar_timeout_sec = 2.0
        self.lidar_failsafe_triggered = False

        self._setup_logger()

    def gimbal_callback(self, msg:Bool):
        last_gimbal = self.gimbal
        self.gimbal = msg.data
        print(self.gimbal)
        if (last_gimbal == 0 and self.gimbal == 1):
            self._log("gimbal changed")
            print("gimbal changed !")

    def lidar_height_callback(self, msg:Float32):
        self.lidar_height = float(msg.data)
        self.last_lidar_time = self.get_clock().now()
    
    def target_listener_callback(self, msg):
        # [핵심 로직] 메시지에서 x, y, z 좌표를 꺼내서 self 변수에 저장
        self.target_x = msg.point.x
        self.target_y = msg.point.y
        self.target_z = msg.point.z
        if(self.mission_state == "GOTO_2"):
            self.target_catched += 1
        
        if(self.mission_state == "GIMBAL_70"):
            self.gimbal_70_target_x = msg.point.x
            self.gimbal_70_target_y = msg.point.y
        self.received_first_data = True

        # 확인용 로그 출력 (실제 주행 땐 주석 처리 가능)
        self._log(
            f"Updated Target -> X: {self.target_x:.4f}, Y: {self.target_y:.4f}, Z: {self.target_z:.4f}"
        )


    def publish_delivery_open_flag(self, flag:bool):
        msg = Bool()
        msg.data = bool(flag)
        self.delivery_open_flag_pub.publish(msg)
        self.delivery_open_flag = bool(flag)

    def _lerp(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        return a + t * (b - a)

    def _build_vertical_quarter_lines_ned(self, prestart_m: float = 5.0):

        required = ["low_left", "low_right", "high_left", "high_right"]
        for k in required:
            if k not in self.main_pts_ned:
                self._log("LINES_SKIP", extra=f"main_pts_ned missing key={k}")
                return False

        ll = np.array(self.main_pts_ned["low_left"],  dtype=float)   # [N,E,D]
        lr = np.array(self.main_pts_ned["low_right"], dtype=float)
        hl = np.array(self.main_pts_ned["high_left"], dtype=float)
        hr = np.array(self.main_pts_ned["high_right"],dtype=float)

        self.lines_ned = {}

        # 4등분이면 내부 라인은 t = 1/4, 2/4, 3/4
        for i, t in enumerate([0.25, 0.50, 0.75], start=1):
            p_low  = self._lerp(ll, lr, t)  # low 변 교점
            p_high = self._lerp(hl, hr, t)  # high 변 교점

            v = (p_low - p_high)            # end->start 벡터 (high -> low)
            norm = float(np.linalg.norm(v))
            if norm < 1e-6:
                self._log("LINES_ERR", extra=f"line_{i} degenerate (norm~0)")
                continue

            # start_point는 low 교점에서 (end->start)방향으로 prestart_m 더 나감
            line_start_point = p_low + (prestart_m / norm) * v
            line_end_point   = p_high

            self.lines_ned[f"line_{i}"] = {
                "t": t,
                "low_intersect":  p_low.tolist(),
                "high_intersect": p_high.tolist(),
                "line_start_point": line_start_point.tolist(),
                "line_end_point":   line_end_point.tolist(),
            }

            self._log(
                "LINE_BUILT",
                extra=(
                    f"line_{i} t={t:.2f} "
                    f"low=({p_low[0]:.2f},{p_low[1]:.2f},{p_low[2]:.2f}) "
                    f"high=({p_high[0]:.2f},{p_high[1]:.2f},{p_high[2]:.2f}) "
                    f"start=({line_start_point[0]:.2f},{line_start_point[1]:.2f},{line_start_point[2]:.2f}) "
                    f"end=({line_end_point[0]:.2f},{line_end_point[1]:.2f},{line_end_point[2]:.2f})"
                )
            )

        return True

    def _setup_logger(self):
        home = os.path.expanduser("~")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(home, f"kros_offboard_{ts}.txt")
        self._log_fp = open(self.log_path, "w", buffering=1)
        self._log("START", extra=f"log_path={self.log_path}")

    def _log(self, tag: str, extra: str = ""):
        # ROS time 대신 wall time이 더 직관적이면 time.time() 사용
        t = time.time()
        cx = float(getattr(self.vehicle_odom, "position", [0,0,0])[0])
        cy = float(getattr(self.vehicle_odom, "position", [0,0,0])[1])
        cz = float(getattr(self.vehicle_odom, "position", [0,0,0])[2])
        line = (f"{t:.3f} | {tag} "
                f"| pos=({cx:.2f},{cy:.2f},{cz:.2f}) yaw={self.now_yaw:.3f} "
                f"| hold={int(self.hold_active)} | {extra}\n")
        self._log_fp.write(line)

    def _as_pos3(self, x, y, z, tag="pos"):
        px, py, pz = float(x), float(y), float(z)

        # NaN/inf 방지
        if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)):
            self.get_logger().error(f"[{tag}] Non-finite position: {(px,py,pz)} "
                                    f"raw={(x,y,z)} types={(type(x),type(y),type(z))}")
            return None

        return [px, py, pz]

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def vehicle_global_callback(self, msg: VehicleGlobalPosition):

        self.vehicle_global = msg
        if self.ref_lat is None:
            self.init_ready_global = True
            lat = float(msg.lat)
            lon = float(msg.lon)
            alt = float(msg.alt)
            self.ref_lat, self.ref_lon, self.ref_alt = lat, lon, alt
            self._log("GOT_REF", extra=f"{lat:.7f},{lon:.7f},{alt:.2f}")

            self._build_main_points_ned()
    
    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self, is_p):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()

        if is_p: # position control
            msg.velocity = False
            msg.position = True
        
        else: 
            msg.velocity = True
            msg.position = False

        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float, yaw: float = None):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        pos = self._as_pos3(x, y, z, tag="publish_position_setpoint")
        if pos is None:
            return
        msg.position = pos

        # position control에서 나머지는 NAN로
        msg.velocity = [NAN, NAN, NAN]
        msg.acceleration = [NAN, NAN, NAN]
        msg.jerk = [NAN, NAN, NAN]

        if yaw is None:
            yaw = float(self.now_yaw)
        msg.yaw = float(yaw)
        msg.yawspeed = 0.0

        self.trajectory_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
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

    def _build_main_points_ned(self):
        if self.ref_lat is None or self.ref_lon is None or self.ref_alt is None:
            return False

        self.main_pts_ned = {}
        for name, p in MAIN_POINT_GPS.items():
            ned = lla_to_ned(p["lat"], p["lon"], p["alt"], self.ref_lat, self.ref_lon, self.ref_alt)
            self.main_pts_ned[name] = ned
            self._log("MAIN_NED", extra=f"{name} -> NED=({ned[0]:.2f},{ned[1]:.2f},{ned[2]:.2f})")
        
        self._build_vertical_quarter_lines_ned(prestart_m = 10.0)
        return True



    def timer_callback(self) -> None:
        """Callback function for the timer."""
        #self.publish_offboard_control_heartbeat_signal(False)
        #print(self.offboard_setpoint_counter)
        now = self.get_clock().now()

            # --- Lidar failsafe: 2초 이상 수신 없으면 강제 LAND ---
        if not self.lidar_failsafe_triggered:
            if self.last_lidar_time is None and self.offboard_setpoint_counter >= 10:
                # 아직 한 번도 라이다를 못 받았으면:
                #  즉시 failsafe
                self._log("NO LIDAR", extra=f"never received-> LAND")
                self.land()
                return 
            elif self.last_lidar_time is None:
                self._log("WAITING LIDAR")
                self.offboard_setpoint_counter += 1
                return
            else:
                age = (now - self.last_lidar_time).nanoseconds * 1e-9
                if age > self.lidar_timeout_sec:
                    self.lidar_failsafe_triggered = True
                    self._log("FAILSAFE_LIDAR_TIMEOUT", extra=f"age={age:.2f}s -> LAND")
                    self.land()
                    return 


        desired_open = self.delivery_open
        if desired_open != self.delivery_open_flag:
            self.publish_delivery_open_flag(desired_open)
            self._log("DELIVERY_FLAG", extra=f"open={int(self.delivery_open_flag)} lidar={self.lidar_height:.3f}")

        if not self.init_ready :
            self.publish_offboard_control_heartbeat_signal(True)
            cx = float(self.vehicle_odom.position[0])
            cy = float(self.vehicle_odom.position[1])
            cz = float(self.vehicle_odom.position[2])
            self.publish_position_setpoint(cx, cy, cz, yaw=self.now_yaw)
        elif self.init_ready and self.offboard_setpoint_counter < 10:
            self.publish_offboard_control_heartbeat_signal(True)
            self.publish_position_setpoint(self.init_x, self.init_y, self.init_z, self.init_yaw)

        if self.offboard_setpoint_counter == 10 and not self.offboard_engaged:
            self.engage_offboard_mode()
            self.arm()
            self.offboard_engaged = True
            self._log("OFFBOARD_ARM")
        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

        
        if not (self.init_ready and self.init_ready_global):
            return
        

        if (self.mission_state == "BOOT"):
            self.publish_offboard_control_heartbeat_signal(True)
            self.publish_position_setpoint(self.init_x, self.init_y, self.init_z+self.takeoff_height, yaw=self.init_yaw)
            if(self.vehicle_odom.position[2] < self.init_z + self.takeoff_height + 0.3):
                self.takeoff = 1
                self.mission_state = "TAKEOFF"
            return
        
        elif (self.mission_state == "TAKEOFF"):
            self.publish_offboard_control_heartbeat_signal(True)
            self.publish_position_setpoint(self.init_x, self.init_y, self.init_z+self.takeoff_height, yaw=self.init_yaw)
            if (self.vehicle_odom.position[2] < self.init_z + self.takeoff_height + 0.3) :
                self.takeoff = 1
                self._log("TAKEOFF_DONE")
                self.mission_state = "ALIGN_1"
        
        elif (self.mission_state == "ALIGN_1"):
            # target = (10, 0, TAKEOFF_HEIGHT)
            # vel = 2
            # yaw = 0
            self._log("ALIGN_1")
            x = self.lines_ned[LINE]["line_start_point"][0]
            y = self.lines_ned[LINE]["line_start_point"][1]
            next_x = self.lines_ned[LINE]["line_end_point"][0]
            next_y = self.lines_ned[LINE]["line_end_point"][1]
            yaw = yaw_to_next_xy(x, y, next_x, next_y)
            self.goto_waypoint(self.init_x, self.init_y, TAKEOFF_HEIGHT, VELOCITY, 0.1, yaw)
            if self.is_departed == 1:
                self.mission_state = "GOTO_1"
                self.is_departed = 0
                self.is_near = 0

        elif (self.mission_state == "GOTO_1"):
            # target = (10, 0, TAKEOFF_HEIGHT)
            # vel = 2
            # yaw = 0
            self._log("GOTO_1")
            
            x = self.lines_ned[LINE]["line_start_point"][0]
            y = self.lines_ned[LINE]["line_start_point"][1]
            next_x = self.lines_ned[LINE]["line_end_point"][0]
            next_y = self.lines_ned[LINE]["line_end_point"][1]
            yaw = yaw_to_next_xy(x, y, next_x, next_y)
            self.goto_waypoint(x, y, TAKEOFF_HEIGHT, 3, 0.1, yaw)
            if self.is_departed == 1 or self.is_near == 1:
                self.mission_state = "GOTO_2"
                self.is_departed = 0
                self.is_near = 0
                self.hold_x = self.vehicle_odom.position[0]
                self.hold_y = self.vehicle_odom.position[1]
                self.hold_z = self.vehicle_odom.position[2]

        elif (self.mission_state == "GOTO_2"):
            self._log("GOTO_2")
            x = self.lines_ned[LINE]["line_end_point"][0]
            y = self.lines_ned[LINE]["line_end_point"][1]
            self.goto_waypoint(x, y, TAKEOFF_HEIGHT, 1, 0.1)
            if self.is_departed == 1:
                self.mission_state = "FAILED"
                self.is_departed = 0
                self.hold_x = self.vehicle_odom.position[0]
                self.hold_y = self.vehicle_odom.position[1]
                self.hold_z = self.vehicle_odom.position[2]

            if self.target_catched>=5:
                self.mission_state = "CATCHED"
                self._log("TARGET CATCHED")
                cx = self.vehicle_odom.position[0]
                cy = self.vehicle_odom.position[1]
                cz = self.vehicle_odom.position[2]
                vxy = np.array([self.prev_v_vec[0], self.prev_v_vec[1]], dtype=float)
                n = float(np.linalg.norm(vxy))
                if n < 1e-3:
                    yaw = float(self.now_yaw)
                    dir_xy = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
                else:
                    dir_xy = vxy / n  # 정규화된 수평 진행 방향
                print("NEXT: CATCHED")

                    # 2m 전진 목표점
                self.hold_x = cx + 2.0 * dir_xy[0]
                self.hold_y = cy + 2.0 * dir_xy[1]

        elif (self.mission_state == "CATCHED"):
            self._log("CATCHED")
            x = self.lines_ned[LINE]["line_start_point"][0]
            y = self.lines_ned[LINE]["line_start_point"][1]
            next_x = self.lines_ned[LINE]["line_end_point"][0]
            next_y = self.lines_ned[LINE]["line_end_point"][1]
            yaw = yaw_to_next_xy(x, y, next_x, next_y)
            self.goto_waypoint(self.hold_x, self.hold_y, TAKEOFF_HEIGHT, 1, 2, yaw)
            if (self.is_departed == 1):
                self.real_target_x = self.target_x
                self.real_target_y = self.target_y
                self.is_departed = 0
                self.mission_state = "APPROACHING"
                print("NEXT : APPROACHING")

                
        
        elif (self.mission_state == "APPROACHING"):
            self.goto_waypoint(self.real_target_x, self.real_target_y, TAKEOFF_HEIGHT, 1)
            if (self.gimbal == 1):
                self._log("CAM to 70")
                self.mission_state = "GIMBAL_70"
                self.hold_x = self.vehicle_odom.position[0]
                self.hold_y = self.vehicle_odom.position[1]
                print("CAM CHANGED TO 70")
            
            elif (self.is_departed == 1):
                self._log("Force to CAM to 70")
                self.mission_state = "GIMBAL_70"
                self.hold_x = self.vehicle_odom.position[0]
                self.hold_y = self.vehicle_odom.position[1]

        elif (self.mission_state == "GIMBAL_70"):
            self.goto_waypoint(self.hold_x, self.hold_y, TAKEOFF_HEIGHT, 1, 1)
            if self.is_departed == 1:
                self.is_departed = 0
                if (self.gimbal_70_target_x is not None):
                    self._log("TARGET CATCHED USING GIMBAl 70. GO DELIVERY")
                    self.mission_state = "DELIVERY"
                    print("GO TO DELIVERY")

        elif (self.mission_state == "DELIVERY"):
            self.goto_waypoint(self.gimbal_70_target_x, self.gimbal_70_target_y, -2, 0.5)
            if self.is_departed == 1:
                self.is_departed = 0
                self.delivery_open = True
                self.mission_state = "DELIVERY_FINISHED"
                print("DELIVERY FINISHED")
        
        elif (self.mission_state == "DELIVERY_FINISHED"):
            self.goto_waypoint(self.init_x, self.init_y, -2, 3, 0.1, self.hold_yaw)
            if self.is_departed == 1:
                self.mission_state = "LANDING"
                self.is_departed = 0
                self.delivery_open = True

        elif (self.mission_state == "LANDING"):
            target = (self.init_x, self.init_y, 0)
            vel = 0.5
            self.goto_waypoint(target[0], target[1], target[2], vel)
            if (self.vehicle_odom.position[2] > -0.5):
                self._log("LAND_CMD")
                self.land()
                exit(0)


    def vehicle_odom_callback(self, msg):
        self.vehicle_odom = msg
        self.now_yaw = euler_from_quaternion(msg.q[0], msg.q[1], msg.q[2], msg.q[3])
        if not self.yaw_hold_valid:
            self.yaw_hold = float(self.now_yaw)
            self.yaw_hold_valid = True

        if (self.offboard_setpoint_counter >= 10 and not self.init_ready):
            self.init_x = float(msg.position[0])
            self.init_y = float(msg.position[1])
            self.init_z = float(msg.position[2])
            self.init_yaw = float(self.now_yaw)

            self.ref_yaw = 0.0

            self._log("INIT_ODOM", extra=f"init=({self.init_x:.2f},{self.init_y:.2f},{self.init_z:.2f}) init_yaw={self.init_yaw:.3f}")
            self.init_ready = True

    def publish_yaw_with_hovering(self, x: float, y: float, z: float, yaw_target: float):
        # position control heartbeat
        self.publish_offboard_control_heartbeat_signal(True)

        # 현재 yaw
        # cur_yaw = euler_from_quaternion(
        #     self.vehicle_odom.q[0], self.vehicle_odom.q[1],
        #     self.vehicle_odom.q[2], self.vehicle_odom.q[3]
        # )

        # # 목표 yaw까지 천천히
        # err = angle_diff(yaw_target, cur_yaw)
        # max_step = float(self.yaw_rate_limit) * float(self.dt)
        # step = max(-max_step, min(max_step, err))
        # yaw_cmd = float(wrap_to_pi(cur_yaw + step))
        #self.previous_yaw = yaw_cmd

        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        msg.position = [float(x), float(y), float(z)]
        msg.velocity = [NAN, NAN, NAN]
        msg.acceleration = [NAN, NAN, NAN]
        msg.jerk = [NAN, NAN, NAN]

        msg.yaw = yaw_target
        msg.yawspeed = NAN  # yaw로 직접 제어

        self.trajectory_setpoint_publisher.publish(msg)

    def publish_velocity_setpoint(self, t_x: float, t_y: float, t_z: float,
                                v: float, yaw_target: float):
        cx = float(self.vehicle_odom.position[0])
        cy = float(self.vehicle_odom.position[1])
        cz = float(self.vehicle_odom.position[2])

        dx = float(t_x - cx)
        dy = float(t_y - cy)
        dz = float(t_z - cz)

        # === 튜닝 파라미터 (원하는 느낌에 따라 조절) ===
        z_priority_band = 0.8   # [m] 이 이상 z 오차면 z를 강하게 우선
        k_z = 1.0               # [1/s] z 비례 이득 (vz = k_z * dz)
        vz_max = min(1.0, float(v))  # [m/s] z축 최대 속도 (전체 v보다 크지 않게)
        eps = 1e-6

        if v <= 0.0:
            v_vec = np.zeros(3, dtype=float)
        else:
            # 먼저 z 속도 결정 (비례제어 + 제한)
            vz_cmd = k_z * dz
            vz_cmd = max(-vz_max, min(vz_max, vz_cmd))

            z_abs = abs(dz)
            if z_abs >= z_priority_band:
                xy_scale = max(0.0, 1.0 - (z_abs - z_priority_band) / max(z_priority_band, eps))
            else:
                xy_scale = 1.0

            # 남은 속도 예산으로 XY 속도 배분
            vxy_budget = math.sqrt(max(0.0, float(v)*float(v) - float(vz_cmd)*float(vz_cmd)))
            vxy_budget *= xy_scale

            dxy = math.hypot(dx, dy)

            if dxy < eps:
                vx_cmd, vy_cmd = 0.0, 0.0
            else:
                ux, uy = dx / dxy, dy / dxy
                vx_cmd = ux * vxy_budget
                vy_cmd = uy * vxy_budget

            v_vec = np.array([vx_cmd, vy_cmd, vz_cmd], dtype=float)

        dv_max = float(self.v_slew) * float(self.dt) 
        dv = v_vec - self.prev_v_vec
        dv_norm = float(np.linalg.norm(dv))
        if dv_norm > dv_max and dv_norm > 1e-9:
            v_vec = self.prev_v_vec + dv * (dv_max / dv_norm)
        self.prev_v_vec = v_vec

        # publish (velocity control)
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        msg.position = [NAN, NAN, NAN]
        msg.velocity = [float(v_vec[0]), float(v_vec[1]), float(v_vec[2])]
        msg.acceleration = [NAN, NAN, NAN]
        msg.jerk = [NAN, NAN, NAN]

        msg.yaw = float(yaw_target)
        msg.yawspeed = NAN

        self.trajectory_setpoint_publisher.publish(msg)

    # def publish_velocity_setpoint(self, t_x: float, t_y: float, t_z: float,
    #                             v: float, yaw_target: float):
    #     cx = float(self.vehicle_odom.position[0])
    #     cy = float(self.vehicle_odom.position[1])
    #     cz = float(self.vehicle_odom.position[2])

    #     dx = float(t_x - cx)
    #     dy = float(t_y - cy)
    #     dz = float(t_z - cz)

    #     dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    #     #print(dist)

    #     #desired yaw

    #     if dist < 1e-6 or v <= 0.0:
    #         v_vec = np.zeros(3, dtype=float)
    #     else:
    #         dir_vec = np.array([dx, dy, dz], dtype=float) / dist
    #         v_vec = dir_vec * float(v)


    #     # slew limit on velocity vector change
    #     dv_max = float(self.v_slew) * float(self.dt)  # m/s per tick
    #     dv = v_vec - self.prev_v_vec
    #     dv_norm = float(np.linalg.norm(dv))
    #     if dv_norm > dv_max and dv_norm > 1e-9:
    #         v_vec = self.prev_v_vec + dv * (dv_max / dv_norm)
    #     self.prev_v_vec = v_vec

    #     # publish (velocity control)
    #     msg = TrajectorySetpoint()
    #     msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

    #     msg.position = [NAN, NAN, NAN] 
    #     msg.velocity = [float(v_vec[0]), float(v_vec[1]), float(v_vec[2])]
    #     msg.acceleration = [NAN, NAN, NAN]
    #     msg.jerk = [NAN, NAN, NAN]

    #     msg.yaw = float(yaw_target)
    #     msg.yawspeed = NAN

    #     self.trajectory_setpoint_publisher.publish(msg)
    
    def goto_waypoint(self, to_x, to_y, to_z, v_max, stop_sec = 0.1, waypoint_yaw_rel = 999, slow_radius=-1):
        # 1) align yaw (with control yaw speed)
        # 2) move to target based on distance feedback

        # yaw_target = wrap_to_pi(float(self.ref_yaw) + float(waypoint_yaw_rel))
        if (slow_radius == -1):
            self.slow_radius = (20.0/9.0) * v_max #v_max일때 slow_radius는 5m, v_max가 0.2일때 slow_radius는 1m
        else:
            self.slow_radius = slow_radius
        
        cx = float(self.vehicle_odom.position[0])
        cy = float(self.vehicle_odom.position[1])
        cz = float(self.vehicle_odom.position[2])
        #to_z = -ground_z + to_z
        #ground_z = -cz-self.lidar_height 
        to_z = to_z - (-cz - self.lidar_height)
        dx = float(to_x - cx)
        dy = float(to_y - cy)
        dz = float(to_z - cz)

        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        yaw_target = 0
        if (waypoint_yaw_rel == 999):
            horiz = math.hypot(dx, dy)
            if horiz >= self.waypoint_range:
                yaw_target = wrap_to_pi(math.atan2(dy, dx))
                self.yaw_hold = yaw_target
                self.yaw_hold_valid = True
            else:
                if getattr(self, "yaw_hold_valid", False):
                    yaw_target = float(self.yaw_hold)
                else:
                    yaw_target = float(self.now_yaw)
                    self.yaw_hold = yaw_target
                    self.yaw_hold_valid = True
        else:
            yaw_target = wrap_to_pi(float(self.ref_yaw) + float(waypoint_yaw_rel))
            self.yaw_hold = yaw_target
            self.yaw_hold_valid = True
        
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        self.distance_target = dist
        yaw_err = abs(angle_diff(yaw_target, self.now_yaw))
        
        if dist < 5:
            self.is_near = 1

        if dist < 1.5 * self.waypoint_range and yaw_err > self.yaw_tol:
            self.publish_yaw_with_hovering(to_x, to_y, to_z, yaw_target)
            self._log("ALIGN_YAW", extra=f"yaw_err={yaw_err:.3f} tol={self.yaw_tol:.3f}")
            return
        
        if (dist < self.waypoint_range and yaw_err <= self.yaw_tol):

            if not self.is_stopping:
                self.is_stopping = True
                self.stop_start_time = self.get_clock().now()
                self.get_logger().info(
                    f"Waypoint {dx}, {dy}, {dz} : stopping for {stop_sec} sec"
                )
                self.hold_active = True
                self.hold_x = to_x
                self.hold_y = to_y
                self.hold_z = to_z
                self.hold_yaw = yaw_target 
                self._log("WP_ARRIVE",
                      extra=f"hold=({self.hold_x:.2f},{self.hold_y:.2f},{self.hold_z:.2f}) "
                            f"hold_yaw={self.hold_yaw:.3f} stop={stop_sec}")

            self.hover_here(to_x, to_y, to_z)

            elapsed = (self.get_clock().now() - self.stop_start_time).nanoseconds * 1e-9
            if elapsed >= stop_sec:
                #print("들어옴")
                self._log("WP_DEPART", extra=f"elapsed={elapsed:.2f}")
                self.is_stopping = False
                self.stop_start_time = None
                self.hold_active = False
                self.is_departed = 1
                #self.waypoint_count += 1
                self.prev_v_vec[:] = 0.0
            return
        
        self.is_stopping = False
        self.stop_start_time = None
        self.hold_active = False

        v_cmd = float(v_max)
        if dist < float(self.slow_radius):
            v_cmd *= (dist / float(self.slow_radius))
        v_cmd = max(0.0, min(float(v_max), v_cmd))
        if dist > self.waypoint_range:
            v_cmd = max(float(self.v_min), v_cmd)

        self.publish_offboard_control_heartbeat_signal(False)
        self.publish_velocity_setpoint(to_x, to_y, to_z, v_cmd, yaw_target)

        # self._log("GOTO",
        #         extra=f"to=({to_x:.2f},{to_y:.2f},{to_z:.2f}) dist={dist:.2f} v={v_cmd:.2f} yaw_t={yaw_target:.3f}")

        cur_lat = float(getattr(self.vehicle_global, "lat", float("nan")))
        cur_lon = float(getattr(self.vehicle_global, "lon", float("nan")))
        cur_alt = float(getattr(self.vehicle_global, "alt", float("nan")))

        cx = float(self.vehicle_odom.position[0])
        cy = float(self.vehicle_odom.position[1])
        cz = float(self.vehicle_odom.position[2])

        self._log(
            "GOTO",
            extra=(
                f"curNED=({cx:.2f},{cy:.2f},{cz:.2f}) "
                f"toNED=({float(to_x):.2f},{float(to_y):.2f},{float(to_z):.2f}) "
                f"dist={dist:.2f} v={v_cmd:.2f} yaw_t={yaw_target:.3f} | "
                f"curGPS=({cur_lat:.7f},{cur_lon:.7f},{cur_alt:.2f}) "
                f"refGPS=({float(self.ref_lat):.7f},{float(self.ref_lon):.7f},{float(self.ref_alt):.2f})"
            )
        )
    def hover_here(self, x, y, z):
        #print("hovering 중")

        self.publish_offboard_control_heartbeat_signal(True)
        self.publish_position_setpoint(x, y, z, yaw=self.hold_yaw)


def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    try:
        rclpy.spin(offboard_control)
    finally:
        try:
            offboard_control._log("SHUTDOWN")
            offboard_control._log_fp.close()
        except Exception:
            pass
        offboard_control.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
