#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013 PAL Robotics SL.
# Released under the BSD License.
#
# Authors:
#   * Siegfried-A. Gevatter

import curses
import math

import rospy
# from geometry_msgs.msg import Twist
from geometry_msgs.msg import Twist, Pose
# from datetime import datetime
import datetime
import os
from math import sin, cos, pi
import sys
import rospy
import numpy as np
import scipy.io as sio
import pdb
import pickle
import matplotlib.pyplot as plt
# from lpv_mpc.msg import control_actions
from std_msgs.msg import Bool, Float32

# https://medium.com/analytics-vidhya/exploring-data-acquisition-and-trajectory-tracking-with-android-devices-and-python-9fdef38f25ee

### under development ###
# class plot_data():

#     def __init__ 
#     # Create figure for plotting
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# xs = []
# ys = []

# # Initialize communication with TMP102
# tmp102.init()

# # This function is called periodically from FuncAnimation
# def animate(i, xs, ys):

#     # Read temperature (Celsius) from TMP102
#     temp_c = round(tmp102.read_temp(), 2)

#     # Add x and y to lists
#     xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
#     ys.append(temp_c)

#     # Limit x and y lists to 20 items
#     xs = xs[-20:]
#     ys = ys[-20:]

#     # Draw x and y lists
#     ax.clear()
#     ax.plot(xs, ys)

#     # Format plot
#     plt.xticks(rotation=45, ha='right')
#     plt.subplots_adjust(bottom=0.30)
#     plt.title('TMP102 Temperature over Time')
#     plt.ylabel('Temperature (deg C)')

# # Set up plot to call animate() function periodically
# ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1000)
# plt.show()


class ImuEncClass(object):
    """ Object collecting IMU + Encoder data
        The encoder measures the angular velocity at the DC motor output. Then
        it is transformed to wheel linear velocity and put over the message:
        /twist.linear.x
    """

    def __init__(self,t0):

        rospy.Subscriber('twist', Twist, self.Twist_callback, queue_size=1)

        rospy.Subscriber('pose', Pose, self.Pose_callback, queue_size=1)

        rospy.Subscriber('/wheel_rpm_feedback', Float32, self.RPM_callback, queue_size=1)

        self.transform = True

        self.roll    = 0.0
        self.pitch   = 0.0
        self.yaw     = 0.0
        self.vx      = 0.0
        self.vy      = 0.0
        self.psiDot  = 0.0
        self.ax      = 0.0
        self.ay      = 0.0
        self.az      = 0.0
        self.X      = 0.0
        self.Y      = 0.0
        self.p     = 0.0
        self.q     = 0.0
        self.r     = 0.0
        self.ph    = 0.0
        self.qh    = 0.0
        self.rh    = 0.0
        
        self.qx     = 0.0
        self.qy    = 0.0
        self.qz    = 0.0
        self.qw    = 0.0


        self.R_ypr = np.array([[cos(self.yaw)*cos(self.pitch) , cos(self.yaw)*sin(self.pitch)*sin(self.roll) - \
                                sin(self.yaw)*cos(self.roll) , cos(self.yaw)*sin(self.pitch)*cos(self.roll) + \
                                sin(self.yaw)*sin(self.roll)], \
                                [sin(self.yaw)*cos(self.pitch) , sin(self.yaw)*sin(self.pitch)*sin(self.roll) + \
                                cos(self.yaw)*cos(self.roll) , sin(self.yaw)*sin(self.pitch)*cos(self.roll) - \
                                cos(self.yaw)*sin(self.roll)], \
                                [-sin(self.pitch) , cos(self.pitch)*sin(self.roll) , cos(self.pitch)*cos(self.roll)]])
                                
        self.ax_window = 1
        self.ay_window = 1
        self.az_window = 1

        self.yaw_offset     = 0.0
        self.psiDot_offset  = 0.0
        self.ax_offset      = 0.0
        self.ay_offset      = 0.0
        self.az_offset      = 0.0
        self.vx_offset      = 0.0
        self.vy_offset      = 0.0

        self.co_yaw     = 0.0
        self.co_psiDot  = 0.0
        self.co_ax      = 0.0
        self.co_ay      = 0.0
        self.co_az      = 0.0
        
        self.co_vx      = 0.0
        self.co_vy      = 0.0

        self.wheel_rpm = 0.0
        # time stamp
        self.t0     = t0

        # Time for yawDot integration
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.prev_time = self.curr_time

                #history
        self.twist_hist = {"timestamp_ms":[],"vx":[],"vy":[],"psiDot":[],"ax":[],"ay":[],"az":[]}
        self.pose_hist = {"timestamp_ms":[],"roll":[],"pitch":[],"yaw":[]}
        self.wheel_rpm_hist = {"timestamp_ms":[],"wheel_rpm":[]}


    def gravity_compensate(self):
        g = [0.0, 0.0, 0.0]
        q = [self.qx , self.qy, self.qz, self.qz]
        acc = [self.ax, self.ay, self.az]
        # get expected direction of gravity
        g[0] = 2 * (q[1] * q[3] - q[0] * q[2])
        g[1] = 2 * (q[0] * q[1] + q[2] * q[3])
        g[2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]
        self.ax, self.ay, self.az =  [acc[0] + g[0], acc[1] + g[1], acc[2] + g[2]]
        # compensate accelerometer readings with the expected direction of gravity
        # return [acc[0] - g[0], acc[1] - g[1], acc[2] - g[2]*self.az_offset]


    def coordinate_transform(self):

        # https://www.basicairdata.eu/knowledge-center/compensation/inertial-measurement-unit-placement/
        
        ############ IMU to Body frame transformation ##########
        # dp = (self.p-self.ph)/(self.curr_time-self.prev_time) 
        # dq = (self.q-self.qh)/(self.curr_time-self.prev_time)
        # dr = (self.r-self.rh)/(self.curr_time-self.prev_time)

        # x_loc = 117.7278*10**(-3)
        # y_loc = 6.1270*10**(-3)
        # z_loc = -0.2799*10**(-3)
        # self.ax = self.ax + x_loc*(self.q**2-self.r**2) - y_loc*(self.p*self.q - self.p-dr) - z_loc*(self.p*self.r - dq)
        # self.ay = self.ay - x_loc*(self.p*self.q + dr) + y_loc*(self.p**2 + self.r**2) - z_loc*(self.q*self.r - dp)  
        # self.az = self.az - x_loc*(self.p*self.r - dq) - y_loc*(self.q*self.r + dp) + z_loc*(self.p**2 + self.q**2)  
        

        ############ Body frame to world frame transformation ##########
        # http://planning.cs.uiuc.edu/node102.html
        self.R_ypr = np.array([[cos(self.yaw)*cos(self.pitch) , cos(self.yaw)*sin(self.pitch)*sin(self.roll) - \
                                sin(self.yaw)*cos(self.roll) , cos(self.yaw)*sin(self.pitch)*cos(self.roll) + \
                                sin(self.yaw)*sin(self.roll)], \
                                [sin(self.yaw)*cos(self.pitch) , sin(self.yaw)*sin(self.pitch)*sin(self.roll) + \
                                cos(self.yaw)*cos(self.roll) , sin(self.yaw)*sin(self.pitch)*cos(self.roll) - \
                                cos(self.yaw)*sin(self.roll)], \
                                [-sin(self.pitch) , cos(self.pitch)*sin(self.roll) , cos(self.pitch)*cos(self.roll)]])
        
        ### gravity compensation
        # http://www.varesano.net/blog/fabio/simple-gravity-compensation-9-dom-imus
        gravity = np.dot(self.R_ypr.T,np.array([0, 0, self.az]).T)
        self.ax = self.ax + gravity[0]
        self.ay = self.ay + gravity[1]
        self.az = self.az #- gravity[2]

        # self.gravity_compensate()
        # self.ax, self.ay, self.az = np.dot(self.R_ypr,np.array([self.ax, self.ay, self.az]).T)


    def Twist_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        # self.ax     = -1*data.linear.y - self.ax_offset
        # self.ay     = data.linear.x - self.ay_offset
        # self.az     = data.linear.z #- self.az_offset

        self.ax     = data.linear.x #- self.ax_offset
        self.ay     = data.linear.y #- self.ay_offset
        self.az     = data.linear.z #- self.az_offset
        
        self.twist_hist["ax"].append(self.ax)
        self.twist_hist["ay"].append(self.ay)
        self.twist_hist["az"].append(self.az)
        
        if (len(self.twist_hist["ax"])>self.ax_window):
            self.ax     = np.mean(np.array(self.twist_hist["ax"][-1*self.ax_window:])) 
            self.ay     = np.mean(np.array(self.twist_hist["ay"][-1*self.ay_window:]))
            self.az     = np.mean(np.array(self.twist_hist["az"][-1*self.az_window:]))


        if self.transform == True:
            self.coordinate_transform()
        

        self.vx     = self.vx+self.ax*(self.curr_time-self.prev_time)  # from IMU
        self.vy     = self.vy+self.ay*(self.curr_time-self.prev_time)

        self.X = self.X +  self.vx*(self.curr_time-self.prev_time)
        self.Y = self.Y +  self.vy*(self.curr_time-self.prev_time)
        
        self.psiDot = data.angular.z - self.psiDot_offset# from IMU

        # p is called roll rate, q pitch rate and r yaw rate.
        self.p = data.angular.x    
        self.q = data.angular.y  
        self.r = data.angular.z   
        self.ph = self.p
        self.qh = self.q
        self.rh = self.r


        self.twist_hist["timestamp_ms"].append(self.curr_time)
        self.twist_hist["vx"].append(self.vx)
        self.twist_hist["vy"].append(self.vy)
        self.twist_hist["psiDot"].append(self.psiDot)
        self.prev_time = self.curr_time

    def Pose_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0

        self.roll   = data.orientation.x
        self.pitch  = data.orientation.y
        self.yaw    = data.orientation.z

        # self.qx   = data.orientation.x
        # self.qy   = data.orientation.y
        # self.qz   = data.orientation.z
        # self.qw   = data.orientation.w
        # self.yaw    = wrap(data.orientation.z) - self.yaw_offset  # from IMU


        # if (len(self.pose_hist["yaw"])>self.ax_window):
        #     self.yaw     = np.mean(np.array(self.pose_hist["yaw"][-1*self.ax_window:])) 
            


        self.pose_hist["timestamp_ms"].append(self.curr_time)
        self.pose_hist["roll"].append(self.roll)
        self.pose_hist["pitch"].append(self.pitch)
        self.pose_hist["yaw"].append(self.yaw)
        self.prev_time = self.curr_time
        


    def RPM_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.wheel_rpm = data.data

        self.wheel_rpm_hist["timestamp_ms"].append(self.curr_time)
        self.wheel_rpm_hist["wheel_rpm"].append(self.wheel_rpm)
        self.prev_time = self.curr_time


    def calibrate_imu(self):

        pose_info = np.array(self.pose_hist["yaw"])
        pisDot_info = np.array(self.twist_hist["psiDot"])
        ax_info = np.array(self.twist_hist["ax"])
        ay_info = np.array(self.twist_hist["ay"])
        az_info = np.array(self.twist_hist["az"])
        
        vx_info = np.array(self.twist_hist["vx"])
        vy_info = np.array(self.twist_hist["vy"])
        yaw_info = np.array(self.pose_hist["yaw"])
        
        self.yaw_offset     = np.mean(pose_info)
        self.psiDot_offset  = np.mean(pisDot_info)
        self.ax_offset      = np.mean(ax_info)
        self.ay_offset      = np.mean(ay_info)
        self.az_offset      = np.mean(az_info)
        self.vx_offset      = np.mean(vx_info)
        self.vy_offset      = np.mean(vy_info)
        self.yaw_offset     = np.mean(yaw_info)

        self.co_yaw     = np.var(pose_info)
        self.co_psiDot  = np.var(pisDot_info)
        self.co_ax      = np.var(ax_info)
        self.co_ay      = np.var(ay_info)
        self.co_vx      = np.var(vx_info)
        self.co_vy      = np.var(vy_info)

        self.vx  = 0
        self.vy  = 0
        self.yaw = 0
        self.X   = 0
        self.Y   = 0


def wrap(angle):
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle


class Velocity(object):

    def __init__(self, min_velocity, max_velocity, num_steps):
        assert min_velocity > 0 and max_velocity > 0 and num_steps > 0
        self._min = min_velocity
        self._max = max_velocity
        self._num_steps = num_steps
        if self._num_steps > 1:
            self._step_incr = (max_velocity - min_velocity) / (self._num_steps - 1)
        else:
            # If num_steps is one, we always use the minimum velocity.
            self._step_incr = 0

    def __call__(self, value, step):
        """
        Takes a value in the range [0, 1] and the step and returns the
        velocity (usually m/s or rad/s).
        """
        if step == 0:
            return 0

        assert step > 0 and step <= self._num_steps
        max_value = self._min + self._step_incr * (step - 1)
        return value * max_value

class TextWindow():

    _screen = None
    _window = None
    _num_lines = None

    def __init__(self, stdscr, lines=10):
        self._screen = stdscr
        self._screen.nodelay(True)
        curses.curs_set(0)

        self._num_lines = lines

    def read_key(self):
        keycode = self._screen.getch()
        return keycode if keycode != -1 else None

    def clear(self):
        self._screen.clear()

    def write_line(self, lineno, message):
        if lineno < 0 or lineno >= self._num_lines:
            raise ValueError('lineno out of bounds')
        height, width = self._screen.getmaxyx()
        y = int((height / self._num_lines) * lineno)
        x = 10
        for text in message.split('\n'):
            text = text.ljust(width)
            # print ("text",text)
            self._screen.addstr(y, x, text)
            y += 1

    def refresh(self):
        self._screen.refresh()

    def beep(self):
        curses.flash()


class KeyTeleop():

    _interface = None

    _linear = None
    _angular = None

    def __init__(self, interface):
        self._interface = interface
        
        # self._pub_cmd = rospy.Publisher('key_vel', Twist)
        
        self.control_hist = []

        self.time0 = rospy.get_time()

        
        ##################### control command publisher ######################
        self.accel_commands     = rospy.Publisher('control/accel', Float32, queue_size=1)
        self.steering_commands  = rospy.Publisher('control/steering', Float32, queue_size=1)
        self.controller_Flag    = rospy.Publisher('controller_flag', Bool, queue_size=1)

        self.past_linear = 0.0 
        self.past_angular = 0.0 
        

        # self.imu_vx  = rospy.Publisher('imu_vx', Float32, queue_size=1)
        # self.imu_vy  = rospy.Publisher('imu_vy', Float32, queue_size=1)

        # self.imu_X  = rospy.Publisher('imu_pos_x', Float32, queue_size=1)
        # self.imu_Y  = rospy.Publisher('imu_pos_y', Float32, queue_size=1)

        # self.imu_ax  = rospy.Publisher('imu_ax', Float32, queue_size=1)
        # self.imu_ay  = rospy.Publisher('imu_ay', Float32, queue_size=1)
        # self.imu_az  = rospy.Publisher('imu_az', Float32, queue_size=1)

        # self.roll  = rospy.Publisher('imu_roll', Float32, queue_size=1)
        # self.yaw  = rospy.Publisher('imu_yaw', Float32, queue_size=1)
        # self.pitch  = rospy.Publisher('imu_pitch', Float32, queue_size=1)
        
        # self.imu_wz  = rospy.Publisher('imu_wx', Float32, queue_size=1)
        

        ##################### sensor subscriber ################################
        # self.imu_enc = ImuEncClass(self.time0)

        # rospy.sleep(5)
        # print ("self.imu_enc number",len(self.imu_enc.twist_hist["vx"]))
        # self.imu_enc.calibrate_imu()      
        self.record_data = False
        self.record_data = rospy.get_param('keyboard_control/record_data')
        
        self.control_commands_his  = {"real_timestamp_ms":[],"timestamp_ms":[],"acceleration":[],"steering":[]}
        
        self._hz = rospy.get_param('~hz', 1/0.033)

        self._num_steps = rospy.get_param('~turbo/steps', 1.01)
        self._num_steps2 = rospy.get_param('~turbo/steps', 0.26)

        forward_min = rospy.get_param('~turbo/linear_forward_min', 0.025)
        forward_max = rospy.get_param('~turbo/linear_forward_max', 0.5)
        self._forward = Velocity(forward_min, forward_max, self._num_steps)

        backward_min = rospy.get_param('~turbo/linear_backward_min', 0.025)
        backward_max = rospy.get_param('~turbo/linear_backward_max', 0.5)
        self._backward = Velocity(backward_min, backward_max, self._num_steps)

        angular_min = rospy.get_param('~turbo/angular_min', 0.7)
        angular_max = rospy.get_param('~turbo/angular_max', 1.2)
        self._rotation = Velocity(angular_min, angular_max, self._num_steps2)

    def run(self):
        self._linear = 0
        self._angular = 0

        rate = rospy.Rate(self._hz)
  
                    
        while True:
            keycode = self._interface.read_key()
            if keycode:
                if self._key_pressed(keycode):
                    
                    self._publish()
                    # print (self.imu_enc.vx,"self.imu_enc.vx")
            else:
                self._publish()
                rate.sleep()
            if keycode == ord('q'):
                if self.record_data == True:
                    self.save_data()
                    rospy.sleep(5.)
                break


    def _get_twist(self, linear, angular):
        twist = Twist()
        if linear >= 0:
            twist.linear.x = self._forward(1.0, linear)
        else:
            twist.linear.x = self._backward(-1.0, -linear)
        twist.angular.z = self._rotation(math.copysign(1, angular), abs(angular))
        return twist

    def _key_pressed(self, keycode):
        dt = 0.01
        movement_bindings = {
            curses.KEY_UP:    ( 1.0,  0.0),
            curses.KEY_DOWN:  (-1.0,  0.0),
            curses.KEY_LEFT:  ( 0.0, 0.1),
            curses.KEY_RIGHT: ( 0.0, -0.1),
        }
        speed_bindings = {
            ord(' '): (0.0, 0.0),
        }
        if keycode in movement_bindings:
            acc = movement_bindings[keycode]
            
            ok = False
            if acc[0]:
                linear = self._linear + acc[0]*dt*1.0
                if abs(linear) <= self._num_steps:
                    self._linear = linear
                    ok = True
            if acc[1]:
                angular = self._angular + acc[1]*dt*5.0
                if abs(angular) <= self._num_steps2:
                    self._angular = angular
                    ok = True
            if not ok:
                self._interface.beep()
        elif keycode in speed_bindings:
            acc = speed_bindings[keycode]
            # acc = acc
            # Note: bounds aren't enforced here!
            if acc[0] is not None:
                self._linear = acc[0]
            if acc[1] is not None:
                self._angular = acc[1]

        if keycode == ord('q'):
            self._running = False
            rospy.signal_shutdown('Bye')
        else:
            return False

        return True

    def _publish(self):
        self._interface.clear()
        self._interface.write_line(2, 'Acceleration: %f, Steering: %f' % (self._linear, self._angular))
        self._interface.write_line(5, 'Use arrow keys to move, space to stop, q to exit.')
        self._interface.refresh()


        twist = self._get_twist(self._linear, self._angular)
        # self._pub_cmd.publish(twist)
        
        self.time1 = rospy.get_time()
        # print ("time at start publishing",self.time1-self.time0)

        if  (self.past_linear != self._linear): 
        
            self.accel_commands.publish(self._linear)

        if (self.past_angular != self._angular): 
 
            self.steering_commands.publish(self._angular)

        self.past_linear = self._linear 
        self.past_angular = self._angular 
        
        # self.imu_vx.publish(self.imu_enc.vx)
        # self.imu_vy.publish(self.imu_enc.vy)
        # self.imu_X.publish(self.imu_enc.X)
        # self.imu_Y.publish(self.imu_enc.Y)
        # self.imu_ax.publish(self.imu_enc.ax)
        # self.imu_ay.publish(self.imu_enc.ay)
        # self.imu_az.publish(self.imu_enc.az)

        # self.roll.publish(self.imu_enc.roll*180.0/pi)
        # self.yaw.publish(self.imu_enc.yaw*180/pi)
        # self.pitch.publish(self.imu_enc.pitch*180/pi)

        # # print ("time for finishing publishing",rospy.get_time()-self.time1)

        # self.control_commands_his["real_timestamp_ms"]. append(rospy.get_time())
        # self.control_commands_his["timestamp_ms"]. append(rospy.get_time()-self.time0)
        # self.control_commands_his["acceleration"]. append(self._linear)
        # self.control_commands_his["steering"]. append(self._angular)

    def save_data(self):
        path = ('/').join(__file__.split('/')[:-2]) + '/data/' 
        
        now = datetime.datetime.now()
        path = path + now.strftime("d%d_m%m_y%Y/")
        dt_string = now.strftime("d%d_m%m_y%Y_hr%H_min%M_sec%S")
        
        control_path = path + 'control_his_'+ dt_string
        print('control save path and file:    ', control_path)

        imu_enc_pose_path = path + 'imu_enc_pose_his_'+ dt_string
        imu_enc_twist_path = path + 'imu_enc_twist_his_'+ dt_string
        
        if not os.path.exists(path):
            os.makedirs(path)
        # print ("self.pose_hist", self.imu_enc.pose_hist)
        # print ("control_his",self.control_commands_his)
        np.save(control_path,self.control_commands_his)
        np.save(imu_enc_pose_path,self.imu_enc.pose_hist)
        np.save(imu_enc_twist_path,self.imu_enc.twist_hist)
        
        # print ("self.twist_hist", self.imu_enc.twist_hist)
        # print ("self.imu_enc.vx", self.imu_enc.vx)
        # print ("control_his",self.control_commands_his)
        # np.save(control_path,self.control_commands_his)
        # np.save(imu_enc_twist_path,self.twist_hist)
    # def saveHistory(self):
    #     data = {'real_time':real_time_his,'timestamp_ms':self.time_his,'x_his':self.x_his,'y_his':self.y_his,'psi_his':self.psi_his,'vx_his':self.vx_his,'vy_his':self.vy_his,'psiDot_his':self.psiDot_his,'noise_hist':self.noise_hist}
    #     path = ('/').join(__file__.split('/')[:-2]) + '/data/' 
        
    #     now = datetime.datetime.now()
    #     path = path + now.strftime("d%d_m%m_y%Y/")
    #     dt_string = now.strftime("d%d_m%m_y%Y_hr%H_min%M_sec%S")
        
    #     simulator_path = path + 'vehicle_simulator_his_'+ dt_string

    #     if not os.path.exists(path):
    #         os.makedirs(path)

    #     np.save(simulator_path,data)



def main(stdscr):
    rospy.init_node('keyboard_control')


    # app = SimpleKeyTeleop(TextWindow(stdscr))
    app = KeyTeleop(TextWindow(stdscr))
    app.run()

if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except rospy.ROSInterruptException:
        pass

