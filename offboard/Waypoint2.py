#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, VehicleOdometry
import math
import numpy as np

FLOATING_SPEED = 0.5
MOVING_SPEED = 1.5
NAN = float('nan')
TAKEOFF_HEIGHT = -5

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

# 이륙 후, x방향으로 0.5m/s로 5이동
# y방향으로 1m/s로 5이동
# x방향으로 1.5m/s로 -5이동
# y방향으로 1.5m/s로 -5이동
# 착륙

WAYPOINTS =[
    {"x" : 0, "y" : 0, "z" : TAKEOFF_HEIGHT, "yaw_mode" : 1, "speed" : 0.5, "stop_seconds" : 0.1},
    {"x" : 5, "y" : 0, "z" : TAKEOFF_HEIGHT, "yaw_mode" : 1, "speed" : 0.5, "stop_seconds" : 0.1},
    {"x" : 5, "y" : 5, "z" : TAKEOFF_HEIGHT, "yaw_mode" : 1, "speed" : 1, "stop_seconds" : 0.1},
    {"x" : 0, "y" : 5, "z" : TAKEOFF_HEIGHT, "yaw_mode" : 1, "speed" : 1.5, "stop_seconds" : 0.1},
    {"x" : 0, "y" : 0, "z" : TAKEOFF_HEIGHT, "yaw_mode" : 1, "speed" : 1.5, "stop_seconds" : 0.1},
    {"x" : 0, "y" : 0, "z" : 0, "yaw_mode" : 1, "speed" : 0.5, "stop_seconds" : 0}
]

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
        
        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.vehicle_odom = VehicleOdometry()
        self.takeoff_height = TAKEOFF_HEIGHT
        self.dt = 0.1
        self.is_new_go = 0
        self.is_departed = 0
        self.wait_in_waypoint = 0
        self.previous_yaw = 0.0
        self.init_x = 0
        self.init_y = 0
        self.init_z = 0
        self.now_yaw = 0
        self.is_land = 0
        self.stop_start_time = None
        self.is_stopping = False
        

        #
        self.goto_phase = "ALIGN"
        self.goto_goal = None
        self.yaw_target = 0.0
        self.yaw_hold_cnt = 0
        self.dist_i = 0.0
        self.prev_v_cmd = 0.0
        self.prev_v_vec = np.zeros(3, dtype=float)
        self.waypoint_range = 0.4
        self.waypoint_num = len(WAYPOINTS)
        self.waypoint_count = 0
        self.takeoff = 0

        self.yaw_rate_limit = 1 #rad/s
        self.yaw_tol = math.radians(8) #정렬 허용오차
        self.yaw_hold_ticks = 5
        
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

    def timer_callback(self) -> None:
        """Callback function for the timer."""
        #self.publish_offboard_control_heartbeat_signal(False)
        #print(self.offboard_setpoint_counter)

        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()
        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

        
        if self.offboard_setpoint_counter < 10:
            return
        
        if (self.takeoff == 0):
            self.publish_offboard_control_heartbeat_signal(True)
            self.publish_position_setpoint(self.init_x, self.init_y, self.init_z+self.takeoff_height)
            if(self.vehicle_odom.position[2] < self.init_z + self.takeoff_height):
                self.takeoff = 1
            return
        #if self.vehicle_local_position.z > self.takeoff_height and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
        #self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)

        self.goto_waypoint(WAYPOINTS[self.waypoint_count]['x']+self.init_x, WAYPOINTS[self.waypoint_count]['y']+self.init_y, WAYPOINTS[self.waypoint_count]['z']+self.init_z, WAYPOINTS[self.waypoint_count]['speed'], WAYPOINTS[self.waypoint_count]['yaw_mode'])
        
        #self.goto_waypoint(0, 0, -5, 0.3, 0)
        

        if self.waypoint_count == len(WAYPOINTS)-1 and self.vehicle_odom.position[2] > -0.5 :
            self.land()
            self.is_land = 1
            exit(0)
        


    def vehicle_odom_callback(self, vehicle_odom):
        if(self.offboard_setpoint_counter == 10):
            self.init_x = self.vehicle_odom.position[0]
            self.init_y = self.vehicle_odom.position[1]
            self.init_z = self.vehicle_odom.position[2]
        self.vehicle_odom = vehicle_odom

        self.now_yaw = euler_from_quaternion(
            self.vehicle_odom.q[0], self.vehicle_odom.q[1],
            self.vehicle_odom.q[2], self.vehicle_odom.q[3]
        )

    def publish_yaw_with_hovering(self, x: float, y: float, z: float, yaw_target: float):
        # position control heartbeat
        self.publish_offboard_control_heartbeat_signal(True)

        # 현재 yaw
        cur_yaw = euler_from_quaternion(
            self.vehicle_odom.q[0], self.vehicle_odom.q[1],
            self.vehicle_odom.q[2], self.vehicle_odom.q[3]
        )

        # 목표 yaw까지 천천히
        err = angle_diff(yaw_target, cur_yaw)
        max_step = float(self.yaw_rate_limit) * float(self.dt)
        step = max(-max_step, min(max_step, err))
        yaw_cmd = float(wrap_to_pi(cur_yaw + step))
        #self.previous_yaw = yaw_cmd

        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        msg.position = [float(x), float(y), float(z)]
        msg.velocity = [NAN, NAN, NAN]
        msg.acceleration = [NAN, NAN, NAN]
        msg.jerk = [NAN, NAN, NAN]

        msg.yaw = yaw_cmd
        msg.yawspeed = NAN  # yaw로 직접 제어

        self.trajectory_setpoint_publisher.publish(msg)

    def publish_velocity_setpoint(self, t_x: float, t_y: float, t_z: float,
                                v: float, yaw_mode: float):
        cx = float(self.vehicle_odom.position[0])
        cy = float(self.vehicle_odom.position[1])
        cz = float(self.vehicle_odom.position[2])

        dx = float(t_x - cx)
        dy = float(t_y - cy)
        dz = float(t_z - cz)

        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        #print(dist)

        #desired yaw
        if (abs(dx) + abs(dy)) > 1:
            desired_yaw = math.atan2(dy, dx)
        else:
            desired_yaw = float(self.previous_yaw)

        # raw velocity vector
        if dist < 1e-6 or v <= 0.0:
            v_vec = np.zeros(3, dtype=float)
        else:
            dir_vec = np.array([dx, dy, dz], dtype=float) / dist
            v_vec = dir_vec * float(v)

        # slew limit on velocity vector change
        dv_max = float(self.v_slew) * float(self.dt)  # m/s per tick
        dv = v_vec - self.prev_v_vec
        dv_norm = float(np.linalg.norm(dv))
        if dv_norm > dv_max and dv_norm > 1e-9:
            v_vec = self.prev_v_vec + dv * (dv_max / dv_norm)
        self.prev_v_vec = v_vec

        # yaw command
        if yaw_mode == 0:
            yaw_cmd = float(self.previous_yaw)
            yawspeed_cmd = NAN
        else:
            cur_yaw = euler_from_quaternion(
                self.vehicle_odom.q[0], self.vehicle_odom.q[1],
                self.vehicle_odom.q[2], self.vehicle_odom.q[3]
            )
            err = angle_diff(desired_yaw, cur_yaw)

            max_step = float(self.yaw_rate_limit) * float(self.dt)
            step = max(-max_step, min(max_step, err))
            yaw_cmd = float(wrap_to_pi(cur_yaw + step))
            #self.previous_yaw = yaw_cmd

            yawspeed_cmd = NAN

        # publish (velocity control)
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        msg.position = [NAN, NAN, NAN] 
        msg.velocity = [float(v_vec[0]), float(v_vec[1]), float(v_vec[2])]
        msg.acceleration = [NAN, NAN, NAN]
        msg.jerk = [NAN, NAN, NAN]

        msg.yaw = float(yaw_cmd)
        msg.yawspeed = yawspeed_cmd

        self.trajectory_setpoint_publisher.publish(msg)
    
    def goto_waypoint(self, to_x, to_y, to_z, v_max, yaw_mode):
        # 1) align yaw (with control yaw speed)
        # 2) move to target based on distance feedback
        
        self.slow_radius = (20.0/9.0) * v_max + (5.0/9.0) #v_max일때 slow_radius는 5m, v_max가 0.2일때 slow_radius는 1m

        if self.is_new_go == 1:
            self.get_logger().info(f"To {[to_x, to_y, to_z]}, at {v_max}m/s")
            self.is_new_go = 0
        
        cx = float(self.vehicle_odom.position[0])
        cy = float(self.vehicle_odom.position[1])
        cz = float(self.vehicle_odom.position[2])

        dx = float(to_x - cx)
        dy = float(to_y - cy)
        dz = float(to_z - cz)
        
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        self.distance_target = dist
        

        stop_sec = WAYPOINTS[self.waypoint_count]['stop_seconds']
        if (dist < self.waypoint_range):

            if not self.is_stopping:
                self.is_stopping = True
                self.stop_start_time = self.get_clock().now()
                self.get_logger().info(
                    f"Waypoint {self.waypoint_count}: stopping for {stop_sec} sec"
                )
            self.hover_here(to_x, to_y, to_z)

            elapsed = (self.get_clock().now() - self.stop_start_time).nanoseconds * 1e-9
            if elapsed >= stop_sec:
                #print("들어옴")
                self.is_stopping = False
                self.stop_start_time = None
                self.is_new_go = 1
                self.waypoint_count += 1
                self.goto_phase = "ALIGN"
                self.yaw_hold_cnt = 0
                self.dist_i = 0
                self.prev_v_vec[:] = 0.0
            return
        else:
            self.is_stopping = False
            self.stop_start_time = None


        if (abs(dx)+abs(dy)) > 1:
            desired_yaw = math.atan2(dy, dx)
        else:
            desired_yaw = float(self.now_yaw)
            self.goto_phase = "GOTO"
            self.yaw_hold_cnt = 0
            self.dist_i = 0.0
            self.prev_v_vec[:] = 0.0

        
        self.publish_offboard_control_heartbeat_signal(False)
        # if self.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
        #     return
        
        #yaw align
        if yaw_mode != 0 and self.goto_phase == "ALIGN": # 처음 takeoff 할때는 align 할 필요 없음
            cur_yaw = euler_from_quaternion(
                self.vehicle_odom.q[0], self.vehicle_odom.q[1],
                self.vehicle_odom.q[2], self.vehicle_odom.q[3]
            )
            # if(self.waypoint_count == 0):
            #     desired_yaw = cur_yaw
            err = angle_diff(desired_yaw, cur_yaw)

            # yaw rate limit
            max_step = float(self.yaw_rate_limit) * float(self.dt)
            step = max(-max_step, min(max_step, err))
            yaw_cmd = float(wrap_to_pi(cur_yaw + step))
            #self.previous_yaw = yaw_cmd
            

            # fix location when yaw align
            #self.publish_offboard_control_heartbeat_signal(True)  # position control
            self.publish_yaw_with_hovering(cx, cy, cz, yaw_cmd)

            if abs(err) < self.yaw_tol:
                self.yaw_hold_cnt += 1
            else:
                self.yaw_hold_cnt = 0

            if self.yaw_hold_cnt >= self.yaw_hold_ticks:
                self.previous_yaw = self.now_yaw
                self.goto_phase = "GOTO"
                self.yaw_hold_cnt = 0
                self.dist_i = 0.0
                self.prev_v_vec[:] = 0.0

            return 

        # yaw_mode==0 -> skip yaw align
        if yaw_mode == 0:
            self.goto_phase = "GOTO"

        # =========================================================
        # PHASE 1: MOVE (feedback speed control)
        # =========================================================

        # distance feedback
        # v_cmd = float(self.kP_dist) * dist

        # if self.kI_dist > 0.0:
        #     self.dist_i += dist * self.dt
        #     self.dist_i = max(-self.dist_i_limit, min(self.dist_i_limit, self.dist_i))
        #     v_cmd += float(self.kI_dist) * self.dist_i
        v_cmd = v_max
        # decrease speed
        if dist < float(self.slow_radius):
            v_cmd *= (dist / float(self.slow_radius))

        # saturate
        v_cmd = max(0.0, min(float(v_max), v_cmd))
        if dist > self.waypoint_range:
            v_cmd = max(float(self.v_min), v_cmd)
        #print("명령 속도 : ", v_cmd)
        

        self.publish_velocity_setpoint(to_x, to_y, to_z, float(v_cmd), float(yaw_mode))
        
        # if(self.waypoint_count == (len(WAYPOINTS)-1) and self.is_departed and self.vehicle_odom.position[2] > -0.5):
        #     self.disarm()

    def hover_here(self, x, y, z):
        #print("hovering 중")
        cx = float(self.vehicle_odom.position[0])
        cy = float(self.vehicle_odom.position[1])
        cz = float(self.vehicle_odom.position[2])

        self.publish_offboard_control_heartbeat_signal(True)
        self.publish_position_setpoint(cx, cy, cz)


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
