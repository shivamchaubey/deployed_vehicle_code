#### Takagi-Sugeno SLAM with MPC controller

## - ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) ` Repository documentation in process!!!`

**Takagi-SugenoSLAM** is a python-ROS based package for real-time 6 states, <br />
![equation](https://latex.codecogs.com/gif.latex?[v_x,&space;v_y,&space;\omega,&space;X,&space;Y,&space;\theta]^T) 
<br />
estimation of the robot navigating in a 2D map. <br />
<br />
First, the Gauss-Newton scan matching approach roughly estimate the state ![equation](https://latex.codecogs.com/gif.latex?[X,&space;Y,&space;\theta]^T) from the LIDAR endpoints using [HectorSLAM](http://wiki.ros.org/hector_slam) and then model-based Takagi-Sugeno Kalman filter is applied to correct and estimate the full state of the vehicle. <br />
LIDAR, and IMU sensors are used.<br />
<br />

**MPC controller** is also added to the ROS package for full body control of the vehicle following limits and constraints. The pipeline for SLAM, estimation and controller can be seen in the figure below 
![Alt text](https://i.ibb.co/zsq8ZD6/scheme.png)

<!---
The vehicle model is represented as, <br /> <br />
![equation](https://latex.codecogs.com/gif.latex?\dot{v_x}&space;=&space;\frac{1}{m}(F_{rx}&space;-&space;F_{flat}\sin(\delta)&space;&plus;&space;mv_y&space;\omega)&space;\newline&space;\dot{v_y}&space;=&space;\frac{1}{m}(F_{flat}\cos(\delta)&space;&plus;&space;F_{ry}&space;-&space;mv_x&space;\omega)\newline&space;\dot{\omega}&space;=&space;\frac{1}{I_z}(l_f&space;F_{flat}\cos(\delta)&space;-&space;l_r&space;F_{ry})&space;\newline&space;\dot{X}&space;=&space;v_x&space;cos(\theta)&space;-&space;v_y&space;sin(\theta)&space;\label{eq:mod_final_X}\newline&space;\dot{Y}&space;=&space;v_x&space;sin(\theta)&space;&plus;&space;v_y&space;cos(\theta)&space;\label{eq:mod_final_Y}\newline&space;\dot{\theta}&space;=&space;\omega&space;\newline) <br /> <br />
where the longitudinal force and lateral forces are, <br /> <br />
![equation](https://latex.codecogs.com/gif.latex?F_{rx}&space;=&space;(C_{m0}&space;-&space;C_{m_1}v_x)D&space;-C_{0}v_x&space;-&space;C_1&space;-&space;\frac{C_D&space;A&space;\rho&space;v_x^2}{2}&space;\label{eq:mod_final_frx}&space;\newline&space;F_{flat}&space;=&space;2C_{af}\left(&space;\delta&space;-&space;\arctan&space;\left(\frac{v_y&space;&plus;&space;l_f&space;\dot{\theta}}{v_x}&space;\right)\right)&space;\label{eq:mod_final_fflat}&space;\newline&space;F_{ry}&space;=&space;-&space;2C_{ar}\arctan&space;\left(&space;\frac{v_y&space;-&space;l_r&space;\dot{\theta}}{v_x}\right)) 
--->
#### Requirements
The code is developed in `Ubunutu 16.04` with `ROS Kinetic Kame`.
Following libraries needed to be installed:
* OSQP
* scipy

#### To launch
Launch files are dependent of each other at this stage of development. A single launch file can be created to launch all the nodes together. 
* To run on the simulation: <br />
   
* To run on the real vehicle: <br /> 



#### ROS implementation
Nodes:

* switching_lqr_observer: Implementation of Takagi-Sugeno Kalman estimator <br /> 
  * *Published Topics*: 
      * /est_state_info -> Publishes estimated states
      * /ol_state_info  -> Publishes open-loop prediction using the model
      * /meas_state_info -> Publishes fused measurement from different sensors.  
  * *Subscribed Topics*: 
      * control/accel -> Dutycycle (D) control input from controller. 
      * control/steering -> Steering angle (![equation](https://latex.codecogs.com/gif.latex?\delta)) control input from controller. 
      * /slam_out_pose -> Roughly estmated pose from LIDAR and scan matching algorithm.
      * /wheel_rpm_feedback -> Subscribes to Motor encoder feedback
      * /twist -> Subscribes to angular velocity from IMU
      * /pose -> Subscribes to orientation from IMU
      * /fused_cam_pose -> Subscribes to pose obtained from fisheye cam if used.

* control: MPC controller implementation <br /> 
  * *Published Topics*: 
      * control/LPV_prediction -> Publishes model predicted states using LPV model
      * control/MPC_prediction  -> Publishes optimizor generated states
      * control/accel -> MPC control output 1. Motor dutycycle (D) 
      * control/steering -> MPC control output 2. Steering angle (![equation](https://latex.codecogs.com/gif.latex?\delta))
  * *Subscribed Topics*: 
      * /est_state_info -> Subscribed to estimated state from the node *switching_lqr_observer*




Detail interconnection of nodes, topics is shown in the below image. 

![Alt text](https://i.ibb.co/FhZ9kkw/rosgraph-real.png)
