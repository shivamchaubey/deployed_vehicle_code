cmake_minimum_required(VERSION 2.8.3)
project(eefig_learning)

add_compile_options(-std=c++11)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs message_generation)

## Services
add_service_files(
  FILES
  LPVMat.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  eefig_learning
)

## Declare a catkin package
catkin_package(CATKIN_DEPENDS std_msgs)

## Build talker and listener
include_directories(include ${catkin_INCLUDE_DIRS})