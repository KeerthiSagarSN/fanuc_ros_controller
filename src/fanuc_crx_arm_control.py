#!/usr/bin/env python3
## 
###
############# ROS Dependencies #####################################
import rospy
import os
from geometry_msgs import msg
from geometry_msgs.msg import Pose, Twist, PoseStamped, TwistStamped, WrenchStamped, PointStamped
from std_msgs.msg import Bool, Float32,Int16,String,MultiArrayDimension,MultiArrayLayout

from sensor_msgs.msg import Joy, JointState, PointCloud
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from scipy.spatial import distance as dist_scipy
from numpy import sum
from tf.transformations import quaternion_matrix
from ipaddress import collapse_addresses
from itertools import chain
import queue
from re import T
from numpy.core.numeric import cross
from geometry_msgs import msg
import rospy
from numpy import matrix, matmul, transpose, isclose, array, rad2deg, abs, vstack, hstack, shape, eye, zeros

from threading import Thread, Lock

import threading



from numpy.linalg import norm, det
from math import atan2, pi, asin, acos
from geometry_msgs.msg import Pose, Twist, PoseStamped, TwistStamped, WrenchStamped, PointStamped
from std_msgs.msg import Bool,Float64
from sensor_msgs.msg import Joy, JointState, PointCloud,Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from copy import copy

from numpy import flip

import tf_conversions as tf_c
from tf2_ros import TransformBroadcaster





# Plot for ROS - Geometry message

from geometry_msgs.msg import Polygon, PolygonStamped, Point32

import PyKDL

from urdf_parser_py.urdf import URDF

# from kdl_parser_py import KDL
from kdl_parser_py import urdf

#import open3d as o3d
### For service - Fixture line detection
from std_srvs.srv import Trigger, TriggerRequest

from std_msgs.msg import Float64MultiArray,Int32

from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.joint_kinematics import JointKinematics


# For joint angle in URDF to screw


mutex = Lock()

# Import URDF of the robot - # todo the param file for fetching the URDF without file location

## Find the appropriate urdf here
robot_description = URDF.from_xml_file(
    '/home/imr/catkin_telebot_ws/src/fanuc_experimental/fanuc_crx10ia_support/urdf/crx10ial_urdf.urdf')


robot_urdf = URDF.from_xml_file(
    '/home/imr/catkin_telebot_ws/src/fanuc_experimental/fanuc_crx10ia_support/urdf/crx10ial_urdf.urdf')

# Build the tree here from the URDF parser file from the file location

build_ok, kdl_tree = urdf.treeFromFile(
    '/home/imr/catkin_telebot_ws/src/fanuc_experimental/fanuc_crx10ia_support/urdf/crx10ial_urdf.urdf')

if build_ok == True:
    print('KDL chain built successfully !!')
else:
    print('KDL chain unsuccessful')

base_link = "base_link"

tip_link = "link_6"


robot_suffix = "_fanuc"

# Build the kdl_chain here
kdl_chain = kdl_tree.getChain(base_link, tip_link)


##############################

# PyKDL_Util here
pykdl_util_kin = KDLKinematics(robot_description, base_link, tip_link)


# Differential jacobian here


vel_ik_solver = PyKDL.ChainIkSolverVel_pinv(kdl_chain, 0.0001, 1000)


# Read documentation from Fanuc-CRX before altering the code below:
# fanuc_experimental


class Geomagic2FANUC():
    def __init__(self):
        rospy.init_node('Geo2FANUC', anonymous=True)

     
        self.desired_pose = Pose()
        self.no_of_joints = kdl_chain.getNrOfJoints()

        self.q_in = PyKDL.JntArray(kdl_chain.getNrOfJoints())
        self.q_in_numpy = zeros(6)




        ## This is bad cant publish to the same topic being subscribed too---- Only for simulation
        self.robot_joint_state_publisher = rospy.Publisher("/joint_states",JointState, queue_size = 100)
        self.fanuc_joint_states_publisher = rospy.Publisher("/position_trajectory_controller/command", JointTrajectory, queue_size=1)
        
        


        # self.meca_pose_publisher = rospy.Publisher("MecademicRobot_pose", Pose, queue_size=1)
        self.fanuc_joints_publisher = rospy.Publisher(
            "fanuc_vel", TwistStamped, queue_size=1)

        
        self.end_effector_pos = rospy.Publisher("/end_effector_pos",PoseStamped,queue_size=1)
        self.end_effector_position = rospy.Publisher("/end_effector_position",PointStamped,queue_size=1)

        #self.fixture_plane_distance = rospy.Publisher('/fixture_plane_distances', Float64, latch=True, queue_size=3)
        # self.meca_twist_publisher = rospy.Publisher("twist_test", Twist, queue_size=1)
        self.scalerXYZ = [0.5, 0.5, 0.5]
        
        self.fanuc_joint_states = JointState()
        self.fanuc_joint_states.position = [0, 0, 0, 0, 0, 0]

        
        self.pub_rate = 1000  # Hz
        self.vel_scale = array([5.0, 5.0, 5.0])

        

        self.cartesian_twist = PyKDL.Twist()
        self.twist_output = PyKDL.Twist()
        self.test_output_twist = PyKDL.Twist()
        self.qdot_out = PyKDL.JntArray(kdl_chain.getNrOfJoints())

        # Moore penrsoe generalized pseudo inverse
        # self.vel_ik_solver = PyKDL.ChainIkSolverVel_pinv(kdl_chain,0.0001,2)

        # SVD damped
        self.vel_ik_solver = PyKDL.ChainIkSolverVel_wdls(kdl_chain, 0.00001, 150)

        self.vel_ik_pinv_solver = PyKDL.ChainIkSolverVel_pinv(
            kdl_chain, 0.00001, 150)

        self.vel_fk_solver = PyKDL.ChainFkSolverPos_recursive(kdl_chain)
        self.jacobian_solver = PyKDL.ChainJntToJacSolver(kdl_chain)


        self.eeFrame = PyKDL.Frame()
        self.end_effector_pos = zeros(shape=(3))
        self.fk_jacobian = PyKDL.Jacobian(self.no_of_joints)
        self.ik_jacobian_K = PyKDL.Jacobian(self.no_of_joints)

        # Numpy jacobian array
        self.fk_jacobian_arr = zeros(shape=(self.no_of_joints, self.no_of_joints))


        br1 = TransformBroadcaster()


        # Limits of all jointts are here

        self.robot_joint_names = ['joint_1', 'joint_2',
            'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.robot_joint_names_pub = ['joint_1', 'joint_2',
            'joint_3', 'joint_4', 'joint_5', 'joint_6']

        self.q_upper_limit = [
            robot_description.joint_map[i].limit.upper - 0.07 for i in self.robot_joint_names]
        self.q_lower_limit = [
            robot_description.joint_map[i].limit.lower + 0.07 for i in self.robot_joint_names]

        self.qdot_limit = [
            robot_description.joint_map[i].limit.velocity for i in self.robot_joint_names]

        

        # Gripper state message declaration

        self.gripper_state_msg = Bool()
        # self.gripper_state_msg.data = False

        ########################
        # print('JOInt liimits  are:',self.q_upper_limit,self.q_lower_limit,self.qdot_limit )

        ## Get frame transformation here - End effector frame with respect to base
        ## Frame - frame.p = position 3x1 vector, frame.M = Rotational matrix of the frame
        self.eeFrame = kdl_chain.getSegment(0).getFrameToTip()

        self.baseFrame = PyKDL.Frame.Identity()

        self.cam_rot = PyKDL.Rotation()
        self.cam_rot = self.cam_rot.RPY(0,0,0)


 


        self.qdot_limit = [
            robot_urdf.joint_map[i].limit.velocity for i in self.robot_joint_names]

        self.qdot_max = array(self.qdot_limit)
        self.qdot_min = -1*self.qdot_max




        print('self.qdot_max', self.qdot_max)
        print('self.qdot_min', self.qdot_min)
        self.q_in = zeros(6)


        self.pykdl_util_kin = KDLKinematics(
            robot_urdf, base_link, tip_link, None)
        #self.q_bounds = zeros(len(self.q_upper_limit),2)

        

        self.q_upper_limit = array([self.pykdl_util_kin.joint_limits_upper]).T
        self.q_lower_limit = array([self.pykdl_util_kin.joint_limits_lower]).T

        self.q_bounds = hstack((self.q_lower_limit, self.q_upper_limit))




        self.test_urdf_file()

        ############3 Python Attributes ####################################

        # self.joints_name = list(tm._joints)
    
    def joint_state_publisher_robot(self,q_joints):
        

        
        q_in = q_joints
        msg = JointState()

        msg.name = [self.robot_joint_names_pub[0], self.robot_joint_names_pub[1],self.robot_joint_names_pub[2],self.robot_joint_names_pub[3],
                    self.robot_joint_names_pub[4], self.robot_joint_names_pub[5]]
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'fanuc_base'
        msg.position = [q_in[0],q_in[1],q_in[2],q_in[3],q_in[4],q_in[5]]
        msg.velocity = []
        msg.effort = []        
        self.robot_joint_state_publisher.publish(msg)
        
    def test_urdf_file(self):

        test_sixteen_arr = array([0.0,0.0,0.0,0.0,0.0,0.0])
        q_in = test_sixteen_arr
        #for i in range(shape(test_sixteen_arr)[0]):
        # Change the for loop with while(rospy not shutdown) for callback loop
        for i in range(100):
            q_in[0] += 0.03
            q_in[2] += 0.03
            q_in[4] += 0.03

            #q_in = flip(q_in)
            mutex.acquire()
            pos_act = array(self.pykdl_util_kin.forward(q_in)[0:3, 3])
            pos_matrix = array(self.pykdl_util_kin.forward(q_in))

            print('pos_act',pos_act)
            print('pos_matrix is',pos_matrix)
            print('seq of joints',q_in)
            pointmsg = PointStamped()
            #pointmsg.header = geo_fanuc_twist.header
            pointmsg.point.x = pos_act[0]
            pointmsg.point.y = pos_act[1]
            pointmsg.point.z = pos_act[2]
            self.end_effector_position.publish(pointmsg)

            #pos_act = array(self.pykdl_util_kin._do_kdl_fk(q_in,5)[0:3,3])

            J_Hess = array(self.pykdl_util_kin.jacobian(q_in))

            self.joint_state_publisher_robot(q_in)
            mutex.release()
            #input('first test stop')

            rospy.sleep(1)
    
if __name__ == '__main__':
    print("Fanuc arm control start up v1 File\n")
    controller = Geomagic2FANUC()
    # controller.start()
    rospy.spin()