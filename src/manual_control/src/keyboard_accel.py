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

import sys
import datetime
import rospy
import numpy as np
import scipy.io as sio
import pdb
import pickle
import matplotlib.pyplot as plt
# from lpv_mpc.msg import control_actions
from std_msgs.msg import Bool, Float32



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
class motor_feedback(object):
    """ Object collecting IMU + Encoder data
        The encoder measures the angular velocity at the DC motor output. Then
        it is transformed to wheel linear velocity and put over the message:
        /twist.linear.x
    """

    def __init__(self,t0):

        rospy.Subscriber('/vehicle_velocity_desired', Float32, self.vel_des_callback, queue_size=1)

        rospy.Subscriber('/vehicle_velocity_feed', Float32, self.vel_feed_callback, queue_size=1)

        self.vel_x_des    = 0.0
        self.vel_x_feed   = 0.0
        
        # time stamp
        self.t0     = t0

        # Time for yawDot integration
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.prev_time = self.curr_time

                #history
        # self.twist_hist = {"timestamp_ms":[],"vx":[],"vy":[],"psiDot":[]}
        # self.pose_hist = {"timestamp_ms":[],"roll":[],"pitch":[],"yaw":[]}

    def vel_des_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.vel_x_des     = data.data  # from DC motor encoder
        
    def vel_feed_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.vel_x_feed     = data.data  # from DC motor encoder
        
class ImuEncClass(object):
    """ Object collecting IMU + Encoder data
        The encoder measures the angular velocity at the DC motor output. Then
        it is transformed to wheel linear velocity and put over the message:
        /twist.linear.x
    """

    def __init__(self,t0):

        rospy.Subscriber('twist', Twist, self.Twist_callback, queue_size=1)

        rospy.Subscriber('pose', Pose, self.Pose_callback, queue_size=1)

        self.roll    = 0.0
        self.pitch   = 0.0
        self.yaw     = 0.0
        self.vx      = 0.0
        self.vy      = 0.0
        self.psiDot  = 0.0

        # time stamp
        self.t0     = t0

        # Time for yawDot integration
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.prev_time = self.curr_time

                #history
        # self.twist_hist = {"timestamp_ms":[],"vx":[],"vy":[],"psiDot":[]}
        # self.pose_hist = {"timestamp_ms":[],"roll":[],"pitch":[],"yaw":[]}

    def Twist_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.vx     = data.linear.x  # from DC motor encoder
        self.vy     = data.linear.y
        self.psiDot = data.angular.z # from IMU

        # self.twist_hist["timestamp_ms"].append(self.curr_time)
        # self.twist_hist["vx"].append(self.vx)
        # self.twist_hist["vy"].append(self.vy)
        # self.twist_hist["psiDot"].append(self.psiDot)


    def Pose_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0

        self.roll   = data.orientation.x
        self.pitch  = data.orientation.y
        self.yaw    = wrap(data.orientation.z) # from IMU

        # self.pose_hist["timestamp_ms"].append(self.curr_time)
        # self.pose_hist["roll"].append(self.roll)
        # self.pose_hist["pitch"].append(self.pitch)
        # self.pose_hist["yaw"].append(self.yaw)

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
        
        self.vel_past = 0
        self.vel_cur = 0
        self.vel_cur_feed = 0
        self.vel_past_feed = 0
        self.dt = 0.033
        self.control_hist = []
        self.accel_feed = 0
        self.moving_avg_window_size = 20
        self.accel_feed_hist = [0]*self.moving_avg_window_size
        
        self.time0 = rospy.get_time()

        
        ##################### control command publisher ######################
        self.accel_commands     = rospy.Publisher('control/accel', Float32, queue_size=1)
        self.steering_commands  = rospy.Publisher('control/steering', Float32, queue_size=1)
        self.controller_Flag    = rospy.Publisher('controller_flag', Bool, queue_size=1)
        self.accel_desired     = rospy.Publisher('control/accel_desired', Float32, queue_size=1)
        self.accel_feedback  = rospy.Publisher('control/accel_feedback', Float32, queue_size=1)
        self.accel_feedback_filter = rospy.Publisher('control/accel_feedback_filter', Float32, queue_size=1)


        ##################### sensor subscriber ################################
        self.imu_enc = ImuEncClass(self.time0)
        self.enc_feedback = motor_feedback(self.time0)
        self.record_data = False
        self.record_data = rospy.get_param('keyboard_control/record_data')
        
        self.control_commands_his  = {"real_timestamp_ms":[],"timestamp_ms":[],"acceleration":[],"steering":[],"desired_velocity":[],"feedback_velocity":[]}
        self.twist_hist = {"real_timestamp_ms":[],"timestamp_ms":[],"vx":[],"vy":[],"psiDot":[]}
        self.pose_hist = {"real_timestamp_ms":[],"timestamp_ms":[],"roll":[],"pitch":[],"yaw":[]}

        self._hz = rospy.get_param('~hz', 1/self.dt)

        self._num_steps = rospy.get_param('~turbo/steps', 3)
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
                angular = self._angular + acc[1]*dt*10
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
        self.vel_cur_feed = self.enc_feedback.vel_x_feed 
        self.vel_cur = self.vel_past + self._linear*self.dt;
        if self.vel_cur>0.15:
            self.vel_cur = 0.15
        if self.vel_cur < -0.15:
            self.vel_cur = -0.15

        # if self._linear>0.3:
        #     self._linear = 0.3
        # if self._linear < -0.3:
        #     self._linear = -0.3

        self.accel_commands.publish(self.vel_cur)
        self.steering_commands.publish(self._angular)

        # motor_feedback
        self.accel_desired.publish(self._linear)
        self.accel_feed  = (self.vel_cur_feed - self.vel_past_feed)/self.dt

        self.accel_feedback.publish(self.accel_feed)
        
        self.accel_feed_hist.append(self.accel_feed)
        self.accel_feed_hist.pop(0)
        self.accel_feedback_filter.publish(sum(self.accel_feed_hist)/len(self.accel_feed_hist))

        # print ("time or finishing publishing",rospy.get_time()-self.time1)
        self.vel_past = self.vel_cur
        self.vel_past_feed = self.vel_cur_feed
        t00 = rospy.get_time()
        self.control_commands_his["real_timestamp_ms"].append(t00)
        self.control_commands_his["timestamp_ms"].append(t00-self.time0)
        self.control_commands_his["acceleration"].append(self._linear)
        self.control_commands_his["steering"]. append(self._angular)
        self.control_commands_his["desired_velocity"].append(self.vel_cur)
        self.control_commands_his["feedback_velocity"].append(self.vel_cur_feed)

        self.twist_hist["real_timestamp_ms"].append(t00)        
        self.twist_hist["timestamp_ms"].append(t00-self.time0)
        self.twist_hist["vx"].append(self.imu_enc.vx)
        self.twist_hist["vy"].append(self.imu_enc.vy)
        self.twist_hist["psiDot"].append(self.imu_enc.psiDot)

        self.pose_hist["real_timestamp_ms"].append(t00)
        self.pose_hist["timestamp_ms"].append(t00-self.time0)
        self.pose_hist["roll"].append(self.imu_enc.roll)
        self.pose_hist["pitch"].append(self.imu_enc.pitch)
        self.pose_hist["yaw"].append(self.imu_enc.yaw)


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
        np.save(imu_enc_pose_path,self.pose_hist)
        np.save(imu_enc_twist_path,self.twist_hist)
        
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

