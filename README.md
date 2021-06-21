#### Takagi-SugenoSLAM with MPC controller

## Repository documentation in process!!! 


**Takagi-SugenoSLAM** is a python-ROS based package for real-time 6 states, <br />
![equation](https://latex.codecogs.com/gif.latex?[v_x, v_y, \omega, X,&space;Y,&space;\theta]^T) 
<br />
estimation of the robot navigating in a 2D map. <br />
<br />
First, the Gauss-Newton scan matching approach roughly estimate the state ![equation](https://latex.codecogs.com/gif.latex?[X,&space;Y,&space;\theta]^T) from the LIDAR endpoints and then model-based Takagi-Sugeno Kalman filter is applied to correct and estimate the full state of the vehicle. <br />
LIDAR, and IMU sensors are used.<br />
<br />

**MPC controller** is also added to the ROS package for full body control of the vehicle following limits and constraints. The pipeline for SLAM, estimation and controller can be seen in the figure below 
![Alt text](https://i.ibb.co/zsq8ZD6/scheme.png)


- <img src="https://latex.codecogs.com/gif.latex?O_t=\text { Onset event at time bin } t " /> 
- <img src="https://latex.codecogs.com/gif.latex?s=\text { sensor reading }  " /> 
- <img src="https://latex.codecogs.com/gif.latex?P(s | O_t )=\text { Probability of a sensor reading value when sleep onset is observed at a time bin } t " />
