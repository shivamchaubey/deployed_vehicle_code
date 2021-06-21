#### Takagi-SugenoSLAM with MPC controller

## Repository documentation in process!!! 


**Takagi-SugenoSLAM** is a python-ROS based package for real-time 6 states, < /br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![equation](https://bit.ly/3vH03DH) < /br>
estimation of the robot navigating in a 2D map. < /br>
< /br>
First, the Gauss-Newton scan matching approach roughly estimate the state ![equation](https://bit.ly/3qdQ9bC) from the LIDAR endpoints and then model-based Takagi-Sugeno Kalman filter is applied to correct and estimate the full state of the vehicle. < /br>
LIDAR, and IMU sensors are used.< /br>
< /br>

**MPC controller** is also added to the ROS package for full body control of the vehicle following limits and constraints. The pipeline for SLAM, estimation and controller can be seen in the figure below
![Alt text](https://i.ibb.co/zsq8ZD6/scheme.png)
