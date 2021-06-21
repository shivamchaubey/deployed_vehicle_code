#### Takagi-Sugeno SLAM with MPC controller

## Repository documentation in process!!! 


**Takagi-SugenoSLAM** is a python-ROS based package for real-time 6 states, <br />
![equation](https://latex.codecogs.com/gif.latex?[v_x,&space;v_y,&space;\omega,&space;X,&space;Y,&space;\theta]^T) 
<br />
estimation of the robot navigating in a 2D map. <br />
<br />
First, the Gauss-Newton scan matching approach roughly estimate the state ![equation](https://latex.codecogs.com/gif.latex?[X,&space;Y,&space;\theta]^T) from the LIDAR endpoints and then model-based Takagi-Sugeno Kalman filter is applied to correct and estimate the full state of the vehicle. <br />
LIDAR, and IMU sensors are used.<br />
<br />

**MPC controller** is also added to the ROS package for full body control of the vehicle following limits and constraints. The pipeline for SLAM, estimation and controller can be seen in the figure below 
![Alt text](https://i.ibb.co/zsq8ZD6/scheme.png)

The vehicle model is represented as, <br />
![equation](https://latex.codecogs.com/gif.latex?\dot{v_x}&space;=&space;\frac{1}{m}(F_{rx}&space;-&space;F_{flat}\sin(\delta)&space;&plus;&space;mv_y&space;\omega)&space;\\&space;\dot{v_y}&space;=&space;\frac{1}{m}(F_{flat}\cos(\delta)&space;&plus;&space;F_{ry}&space;-&space;mv_x&space;\omega)&space;\\&space;\dot{\omega}&space;=&space;\frac{1}{I_z}(l_f&space;F_{flat}\cos(\delta)&space;-&space;l_r&space;F_{ry})&space;\\&space;\dot{X}&space;=&space;v_x&space;cos(\theta)&space;-&space;v_y&space;sin(\theta)&space;\\&space;\dot{Y}&space;=&space;v_x&space;sin(\theta)&space;&plus;&space;v_y&space;cos(\theta)&space;\\&space;\dot{\theta}&space;=&space;\omega&space;\label{eq:mod_final_yaw}) <br />
where the longitudinal force and lateral forces are, <br />
!equation(https://latex.codecogs.com/gif.latex?F_{rx}&space;=&space;(C_{m0}&space;-&space;C_{m_1}v_x)D&space;-C_{0}v_x&space;-&space;C_1&space;-&space;\frac{C_D&space;A&space;\rho&space;v_x^2}{2}&space;\label{eq:mod_final_frx}&space;\newline&space;F_{flat}&space;=&space;2C_{af}\left(&space;\delta&space;-&space;\arctan&space;\left(\frac{v_y&space;&plus;&space;l_f&space;\dot{\theta}}{v_x}&space;\right)\right)&space;\label{eq:mod_final_fflat}&space;\newline&space;F_{ry}&space;=&space;-&space;2C_{ar}\arctan&space;\left(&space;\frac{v_y&space;-&space;l_r&space;\dot{\theta}}{v_x}\right))
