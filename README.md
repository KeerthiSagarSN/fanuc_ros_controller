# ROS1 Controller for FANUC- CRX Robot with ROS-Noetic


### Software and Library Requirements 

* Ubuntu 20.04 LTS
* ROS Noetic
If you are new to ROS, go [here](http://wiki.ros.org/catkin/Tutorials/create_a_workspace) to learn how to create a catkin workspace. 

* [pykdl_utils](http://wiki.ros.org/pykdl_utils) Higher Level Python wrapper for PyKDL in ROS for Kinematic Solver
* Python native libraries [Scipy](https://scipy.org/), [Numpy](https://numpy.org/)



### Clone the repo in the Catkin workspace - ROS1
In a Terminal
```
cd ~/catkin_ws/src/
git clone https://gitlab.com/KeerthiSagarSN/fanuc_ros_controller
```

### Library Installation - Only if not preinstalled

### Clone Fanuc Robot- Modified (Experimental) from original repository [fanuc_experimental](https://github.com/ros-industrial/fanuc_experimental)
```
git clone https://github.com/KeerthiSagarSN/fanuc_experimental
git checkout noetic-devel
```

### Clone Fanuc repository- For Fanuc_drivers 

```
git clone https://github.com/ros-industrial/fanuc
git checkout melodic-devel
```

### Clone PyKDL - From OROCOS-KDL Repository - Latest Branch

```
git clone https://github.com/orocos/orocos_kinematics_dynamics.git
```

### Clone Pykdl - Kinematics Wrapper Repository - Edited for Python 3 and ROS Noetic Compatible
```
git clone https://github.com/KeerthiSagarSN/hrl-kdl.git
git checkout Noetic-devel
sudo apt-get install ros-noetic-urdf-parser-plugin
sudo apt-get install ros-noetic-urdfdom-py

```

### Catkin Build/ Catkin Make - Build & Source all repositories in Catkin Workspace
```
cd ..
catkin build
source devel/setup.bash

```

#### IF you encounter CMAKE Error : Install all ROS- Dependencies - (Only when CMAKE Error !)
##### Replace underscores "_" with "-" in the package name while typing in the MISSING-PACKAGE below
```
sudo apt install ros-noetic-MISSING-PACKAGE

```

#### ROS - Preliminary Dependencies - Install Only if required

```
$ sudo apt-get install ros-$ROS_DISTRO-robot-state-publisher ros-$ROS_DISTRO-joint-state-publisher
$ sudo apt-get install ros-$ROS_DISTRO-joint-state-publisher-gui
```

### To Visualize UR5 robot in Rviz
```
$ roslaunch fanuc_crx10ia_support test_crx10ial.launch
```

### Launch the Controller
#### Fanuc CRX
```
$ roslaunch fanuc_ros_controller fanuc_ros_controller.launch

```
