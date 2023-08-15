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
#from pytransform3d.urdf import UrdfTransformManager
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

from kuka_rsi_hw_interface.srv import *



# Polygon plot for ROS - Geometry message
from jsk_recognition_msgs.msg import PolygonArray, SegmentArray
from geometry_msgs.msg import Polygon, PolygonStamped, Point32


## Do not distribute this library
from rospygradientpolytope.linearalgebra import proj_point_plane,V_unit


# Old library components - Refer to catkin_telebot_ws for the Rviz plane messages
#from rospygradientpolytope.visual_polytope import velocity_polytope, desired_polytope, velocity_polytope_with_estimation
#from rospygradientpolytope.polytope_ros_message import create_plane_msg,create_polytopes_msg, create_polygon_msg, create_capacity_vertex_msg, create_segment_msg



## Do not distribute this library

from rospygradientpolytope.visual_polytope import velocity_polytope, desired_polytope, velocity_polytope_with_estimation
from rospygradientpolytope.polytope_ros_message import create_polytopes_msg, create_polygon_msg, create_capacity_vertex_msg, create_segment_msg
from rospygradientpolytope.polytope_functions import get_polytope_hyperplane, get_capacity_margin
#from rospygradientpolytope.polytope_gradient_functions_optimized import Gamma_hat_gradient
#from rospygradientpolytope.polytope_gradient_functions import Gamma_hat_gradient,Gamma_hat_gradient_dq

from rospygradientpolytope.sawyer_functions import jacobianE0, position_70
from rospygradientpolytope.robot_functions import getHessian, getJ_pinv
from rospygradientpolytope.linearalgebra import check_ndarray

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




# getting the node namespace
namespace = rospy.get_namespace()

# For joint angle in URDF to screw


#tm = UrdfTransformManager()


'''
BASE_DIR = '/home/imr/Unity-Robotics-Hub/tutorials/pick_and_place/ROS/src/kuka_experimental/kuka_kr4_support/urdf/'

tm.load_urdf(f.read(),mesh_path=BASE_DIR)



robot_name_inp = kr4r600
with open(BASE_DIR + str(robot_name_inp) + str(".urdf"), "r") as f:
    tm.load_urdf(f.read(),mesh_path=BASE_DIR)
'''


mutex = Lock()

# Import URDF of the robot - # todo the param file for fetching the URDF without file location


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


# Read documentation from KUKA KR4 before altering the code below:


class Geomagic2FANUC():
    def __init__(self):
        rospy.init_node('Geo2FANUC', anonymous=True)



        # PyKDL_Util here
        #### Polytope publish for computing the intersection of the polytopes
        self.p_Hrep_A = Float64MultiArray()
        self.p_Hrep_b = Float64MultiArray()

        self.p_Hrep_size = Float64MultiArray()
        self.p_Hrep_size_dim = MultiArrayLayout()
        #dim_array = MultiArrayDimension()
        #dim_array.size = 2
        #self.p_Hrep_size_dim.dim = dim_array
        #self.p_Hrep_size.layout = self.p_Hrep_size_dim

        self.publish_velocity_polytope = rospy.Publisher(
            "/available_velocity_polytope"+robot_suffix, PolygonArray, queue_size=100)
        

        # Polytope in its H-rep from the polytope module

        self.publish_velocity_polytope_Hrep_A = rospy.Publisher(
            "/available_velocity_polytope_p_H_A"+robot_suffix, Float64MultiArray, queue_size=100)
        
        self.publish_velocity_polytope_Hrep_b = rospy.Publisher(
            "/available_velocity_polytope_p_H_b"+robot_suffix, Float64MultiArray, queue_size=100)
        
        self.publish_velocity_polytope_Hrep_size = rospy.Publisher(
            "/available_velocity_polytope_p_H_size"+robot_suffix, Float64MultiArray, queue_size=100)


        self.publish_desired_polytope = rospy.Publisher(
            "/desired_velocity_polytope"+robot_suffix, PolygonArray, queue_size=100)
        self.publish_capacity_margin_polytope = rospy.Publisher(
            "/capacity_margin_polytope"+robot_suffix, PolygonArray, queue_size=100)
        self.publish_vertex_capacity = rospy.Publisher(
            "/capacity_margin_vertex"+robot_suffix, PointStamped, queue_size=1)
        self.publish_vertex_proj_capacity = rospy.Publisher(
            "/capacity_margin_proj_vertex"+robot_suffix, PointStamped, queue_size=1)
        self.publish_capacity_margin_actual = rospy.Publisher(
            "/capacity_margin_actual"+robot_suffix, SegmentArray, queue_size=1)

        # publish plytope --- Estimated Polytope - Publisher

        self.publish_velocity_polytope_est = rospy.Publisher(
            "/available_velocity_polytope_est"+robot_suffix, PolygonArray, queue_size=100)
        self.publish_capacity_margin_polytope_est = rospy.Publisher(
            "/capacity_margin_polytope_est"+robot_suffix, PolygonArray, queue_size=100)
        self.publish_vertex_proj_capacity_est = rospy.Publisher(
            "/capacity_margin_proj_vertex_est"+robot_suffix, PointStamped, queue_size=1)
        self.publish_capacity_margin_actual_est = rospy.Publisher(
            "/capacity_margin_actual_est"+robot_suffix, SegmentArray, queue_size=1)

        self.publish_vertex_pose = rospy.Publisher(
            "/ef_pose_vertex"+robot_suffix, PointStamped, queue_size=1)
        self.publish_vertex_desired_pose = rospy.Publisher(
            "/ef_desired_pose_vertex"+robot_suffix, PointStamped, queue_size=1)

        self.polytope_display = True

        self.polytope_display_on_sub = rospy.Subscriber("polytope_show",Bool,self.polytope_show_on_callback)
        #self.start_interactive_ik_sub = rospy.Subscriber("run_ik",Bool,self.start_interactive_ik)
        #self.pub_end_ik = rospy.Publisher("ik_progress",Int16,queue_size=1)
        #self.pub_status_ik = rospy.Publisher("status_ik",String,queue_size=1)
        #self.sub_ik_pos = rospy.Subscriber("interactive_sphere",Pose,self.ik_pose_callback)


        self.cartesian_desired_vertices = 2.0*array([[0.20000, 0.50000, 0.50000],
                                                     [0.50000, -0.10000, 0.50000],
                                                     [0.50000, 0.50000, -0.60000],
                                                     [0.50000, -0.10000, -0.60000],
                                                     [-0.30000, 0.50000, 0.50000],
                                                     [-0.30000, -0.10000, 0.50000],
                                                     [-0.30000, 0.50000, -0.60000],
                                                     [-0.30000, -0.10000, -0.60000]])
        

        # Create an interactive marker server
        # Create an interactive marker server

        self.desired_vertices = self.cartesian_desired_vertices
        self.sigmoid_slope = 150

        self.sigmoid_slope_input = 150
        
        self.desired_pose = Pose()

        self.hot_wire_toggle = False

        self.no_of_joints = kdl_chain.getNrOfJoints()

        self.q_in = PyKDL.JntArray(kdl_chain.getNrOfJoints())
        self.q_in_numpy = zeros(6)




        self.cut_xz_active = False

        self.cut_xy_active = False
        self.geomagic_subscriber = rospy.Subscriber(
            "geomagic_twist_kuka", TwistStamped, self.geo_to_kuka_callback, queue_size=1)

        self.EnterFlag = True

        '''
        self.kuka_joint_states_subscriber = rospy.Subscriber(
            "/joint_states", JointState, self.kuka_callback, queue_size=1)
        '''

        ## This is bad cant publish to the same topic being subscribed too---- Only for simulation
        self.robot_joint_state_publisher = rospy.Publisher("/joint_states",JointState, queue_size = 100)
        self.kuka_joint_states_publisher = rospy.Publisher("/position_trajectory_controller/command", JointTrajectory, queue_size=1)
        
        

        self.button_kuka_subscriber = rospy.Subscriber(
            "buttons_kuka", Joy, self.button_kuka_update_callback, queue_size=1)
        
        self.collider_state_subscriber = rospy.Subscriber(
            "object_collider", Bool, self.collision_state_callback, queue_size=1)

        # self.meca_pose_publisher = rospy.Publisher("MecademicRobot_pose", Pose, queue_size=1)
        self.kuka_joints_publisher = rospy.Publisher(
            "KUKA_vel", TwistStamped, queue_size=1)
        self.kuka_gripper_state_subscriber = rospy.Subscriber(
            "kuka_gripper_state_topic", Bool, self.gripper_actuate_callback, queue_size=1)
        self.kuka_gripper_current_state = rospy.Publisher(
            "kuka_gripper_current_state", Bool, queue_size=1)

        '''
        self.force_torque_ati_kuka = rospy.Subscriber(
            "/ati_force_torque_sensor_2/transformed_world", WrenchStamped, self.ft_kuka_callback, queue_size=1)
        '''

        self.publish_fixture_plane_plot_1 = rospy.Publisher("/fixture_plane_plot_1", PolygonArray, queue_size=100)
        self.publish_fixture_plane_plot_2 = rospy.Publisher("/fixture_plane_plot_2", PolygonArray, queue_size=100)
        self.publish_fixture_plane_plot_3 = rospy.Publisher("/fixture_plane_plot_3", PolygonArray, queue_size=100)
        # Subscriber for Dexterous mode
        '''
        self.dexterous_mode = rospy.Subscriber(
            "dexterous_mode_topic", Bool, self.dexterous_mode_callback, queue_size=1)
        

        self.hot_wire_mode_on = rospy.Subscriber(
            "hot_wire_state_on_topic", Bool, self.hot_wire_mode_on_callback, queue_size=1)
        self.hot_wire_mode_on = rospy.Subscriber(
            "hot_wire_state_off_topic", Bool, self.hot_wire_mode_off_callback, queue_size=1)


        self.cut_xy_plane = rospy.Subscriber("cut_xy_topic", Bool, self.cut_xy_callback, queue_size=1)


        self.cut_xz_plane = rospy.Subscriber("cut_xz_topic", Bool, self.cut_xz_callback, queue_size=1)

        self.active_cam = rospy.Subscriber("active_cam",Int32,self.active_cam_callback,queue_size=1)

        '''
        # Fixture line detection service - Initializing parameter to call the service
        self.initialisationSub = rospy.Subscriber("/fixture_line_topic", Bool, self.fixture_line_topic_callback,queue_size=1)



        # Rosbag - start service - Initializing parameter to call the service
        self.start_rosbag = rospy.Subscriber("/rosbag_start_topic", Bool, self.rosbag_start_record_callback,queue_size=1)

        self.stop_rosbag = rospy.Subscriber("/rosbag_stop_topic", Bool, self.rosbag_stop_record_callback,queue_size=1)

        # Fixture line segment topic subscriber - Publishing only once insdie the service
        ### Bad implementation - Using image for the matrix representation - TODO - Change to better semantic representation
        self.fixture_line_subscriber = rospy.Subscriber("fixture_line_publisher",Float64MultiArray,self.fixture_line_segment_callback,queue_size=1)
        #### Get the force-torque sensor values f
        
        
        ##  Fixture planes for Generating force feedback
        self.fixture_plane_wrench = rospy.Publisher("/virtual_fixture/plane", WrenchStamped, queue_size=1)

        self.end_effector_pos = rospy.Publisher("/end_effector_pos",PoseStamped,queue_size=1)
        self.end_effector_position = rospy.Publisher("/end_effector_position",PointStamped,queue_size=1)

        #self.fixture_plane_distance = rospy.Publisher('/fixture_plane_distances', Float64, latch=True, queue_size=3)
        # self.meca_twist_publisher = rospy.Publisher("twist_test", Twist, queue_size=1)
        self.scalerXYZ = [0.5, 0.5, 0.5]
        self.mecaOffsetXYZ = [0.160, 0.0, 0.225]

        self.geomagic_offset = [0.1314, -0.16, 0.1]
        self.haptic_twist = TwistStamped()
        self.kuka_joint_states = JointState()
        self.kuka_joint_states.position = [0, 0, 0, 0, 0, 0]

        self.start_linearvelocity_state = False
        self.start_angularvelocity_state = False
        self.geo_pose_orientation_prev = array([0.0, 0.0, 0.0])
        # self.geo_pose_wy = PoseStamped()
        # self.geo_pose_wz = PoseStamped()
        # self.geo_pose_orientation_prev = PoseStamped()

        # self.geo_twist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pub_rate = 1000  # Hz
        self.vel_scale = array([5.0, 5.0, 5.0])

        self.dexterous_lin_vel_scale = 5*array([0.5,0.5,0.5])
        self.dexterous_ang_vel_scale = 5*array([0.5,0.5,0.5])
        self.angular_vel_scale = array([9.0, 9.0, 9.0])
        self.W_quaternion = matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [
                                   0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.dt_quaternion = matrix([[0.0], [0.0], [0.0], [0.0]])
        self.angular_velocity_vector = matrix([[0.0], [0.0], [0.0]])
        self.previous_msg_state = False
        self.previous_msg = matrix([[0, 0, 0, 0, 0, 0]])
        self.button_kuka_state = [0, 0]
        self.change_gripper_state = False
        self.changing_state = False

        self.current_collision_state = True
        self.angle_rot_previous_z = 0.0
        self.angle_rot_z = 0.0
        self.angle_rot_previous_y = 0.0
        self.angle_rot_y = 0.0
        self.angle_rot_previous_x = 0.0
        self.angle_rot_x = 0.0

        # Gripper current state
        self.gripper_closed = False
        self.ang_vel_x_prev = 0
        self.ang_vel_y_prev = 0
        self.ang_vel_z_prev = 0

        self.flag_linear = False
        self.flag_angular = False

        self.lin_vel_x_prev = 0
        self.lin_vel_y_prev = 0
        self.lin_vel_z_prev = 0

        # Dexterous mode initial state

        self.dexterous_mode_state = False

        # Collision zone flags here
        self.collision_zone_counter = 0
        self.leave_flag_collision_zone = False

        # Cartesian Wrench is declared here
        # self.cartesian_wrench = WrenchStamped()

        # ft_sensor - for KUKA data is declared here
        self.ft_sensor = WrenchStamped()

        self.ft_sensor_gc = WrenchStamped()





        '''
        self.q_in[0] = 0
        self.q_in[1] = -1.57
        self.q_in[2] = 1.57
        self.q_in[3] = 0
        self.q_in[4] = 1.57
        self.q_in[5] = 0
        '''

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

        # Stiffness matrix
        # self.Kx_kuka = PyKDL.Stiffness()

        self.eeFrame = PyKDL.Frame()

        self.end_effector_pos = zeros(shape=(3))
        self.fk_jacobian = PyKDL.Jacobian(self.no_of_joints)
        self.ik_jacobian_K = PyKDL.Jacobian(self.no_of_joints)

        # Numpy jacobian array
        self.fk_jacobian_arr = zeros(shape=(self.no_of_joints, self.no_of_joints))

        self.JT_Twist_0 = PyKDL.Twist()
        self.JT_Twist_1 = PyKDL.Twist()
        self.JT_Twist_2 = PyKDL.Twist()
        self.JT_Twist_3 = PyKDL.Twist()
        self.JT_Twist_4 = PyKDL.Twist()
        self.JT_Twist_5 = PyKDL.Twist()

        br1 = TransformBroadcaster()

        # print('Jacobian class is:')
        # print(self.get_jacobian_kuka)

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

        # KDL_Dynamics here

        # KDL Dynamics here
        # Define stiffness matrix here
        # Cartesian stiffness matrix is declared here

        self.K_C = eye(6)

        self.K_C[0, 0] = 10
        self.K_C[1, 1] = 10
        self.K_C[2, 2] = 1e-10
        # Polytope parameters are below
        # Velocity polytope
        # self.q_max = array([100,100,100,100,100,100,100])
        # self.q_min = -1*array([100,100,100,100,100,100,100])

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


        ## Toggle to switch off the hot-wire only once in the service
        
        #print('rotation is',self.cam_rot.RPY(0,0,1.57))
        

        # print('self.eeFrame is')
        # print(self.eeFrame)
        # transform1 = tf.Transform

        # self.l_6_Frame = PyKDL.Frame()


        ### Fixture line segments from the Fixture line detection server
        self.fixture_lines = []
        self.fixture_planes = [] ## 0:2 columns - normal to the plane, 3:5 point on the plane]

        self.plane_verts = []
        self.pose_verts = []

        ## Distance between virtual guide planes
        self.dist_plane = 0.005 # 3 mm 
        self.dist_tol = 0.002

        self.qdot_limit = [
            robot_urdf.joint_map[i].limit.velocity for i in self.robot_joint_names]

        self.qdot_max = array(self.qdot_limit)
        self.qdot_min = -1*self.qdot_max
        self.fun_iter = Int16()
        self.fun_iter.data = 0
        self.start_optimization_bool = False

        self.msg_status_ik = String()

        print('self.qdot_max', self.qdot_max)
        print('self.qdot_min', self.qdot_min)
        self.q_in = zeros(6)


        self.plot_polytope_thread = None
        self.thread_is_running = False
        #self.thread_cm_is_running = False


        #self.q_test = zeros(7)

        self.pykdl_util_kin = KDLKinematics(
            robot_urdf, base_link, tip_link, None)
        #self.q_bounds = zeros(len(self.q_upper_limit),2)

        

        self.q_upper_limit = array([self.pykdl_util_kin.joint_limits_upper]).T
        #self.q_upper_limit = self.pykdl_util_kin.joint_limits_upper
        self.q_lower_limit = array([self.pykdl_util_kin.joint_limits_lower]).T
        #self.q_lower_limit = self.pykdl_util_kin.joint_limits_lower

        self.q_bounds = hstack((self.q_lower_limit, self.q_upper_limit))
        sigmoid_slope_test = array([50, 100, 150, 200, 400])

        self.sigmoid_slope_array = array([50, 100, 150, 200, 400])

        self.cm_est = None

        self.time_counter = 0
        self.fun_counter = 0

        self.color_array_cm = ['g','r']
        self.cm_est_arr = zeros(shape=(2))
        self.cm_est_arr[:] = -10000
        self.time_arr = zeros(shape=(2))

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
            #pointmsg.header = geo_kuka_twist.header
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
    
    def polytope_show_on_callback(self,show_bool):
        self.polytope_display = show_bool.data

        if self.polytope_display:
            self.start_plot_thread()
            #self.start_cm_plot_thread()
        else:
            self.stop_thread()
            #self.stop_cm_thread()
    def start_plot_thread(self):
        if self.thread_is_running:
            print("Thread already running!")
            return
        self.plot_polytope_thread = threading.Thread(target=self.plot_polytope)
        self.thread_is_running = True
        self.plot_polytope_thread.start()

        #input('I have started thread')
    def stop_thread(self):
        self.thread_is_running = False
        print('Stopping thread')
    
    def plot_polytope(self):
        
        while self.thread_is_running:
            if self.polytope_display:

                
                #print('I am plotting')


                mutex.acquire()
                pos_act = array(self.pykdl_util_kin.forward(self.q_in)[0:3, 3])

                #pos_act = array(self.pykdl_util_kin._do_kdl_fk(q_in,5)[0:3,3])

                J_Hess = array(self.pykdl_util_kin.jacobian(self.q_in))

                # print('pos_act_mat',pos_act_mat)

                #pos_act = position_70(q_in)
                # print('self.pos_reference',self.pos_reference)
                #print('Current position in optimization is', pos_act)
                #input('Wait here ')
                # print('norm,',norm(pos_act-self.pos_reference))
                # print('self.pos_reference',self.pos_reference)
                #print('pos_act', pos_act)

                #print('distance error norm',distance_error)
                

                scaling_factor = 30.0
                
                ### Polytope plot with estimation
                
                #pykdl_kin_jac = pykdl_util_kin.jacobian(self.q_in_numpy)
                polytope_verts, polytope_faces, facet_vertex_idx, capacity_faces, capacity_margin_proj_vertex, \
                    polytope_verts_est, polytope_faces_est, capacity_faces_est, capacity_margin_proj_vertex_est,p_H,p_H_est,cm_index = \
                                                velocity_polytope_with_estimation(J_Hess,self.qdot_max,self.qdot_min,self.desired_vertices,self.sigmoid_slope_input)
                desired_polytope_verts, desired_polytope_faces = desired_polytope(self.desired_vertices)



                #print('polytopes A is',p_H.A)
                #print('polytopes b is',p_H.b)
                #print('polytope estimated A is',p_H_est.A)
                p_H_A_flatten = p_H.A
                self.p_Hrep_A.data = p_H_A_flatten.ravel()
                
                self.p_Hrep_b.data = p_H.b
                p_H_A_size = shape(p_H.A)
                


                self.p_Hrep_size.data = array(p_H_A_size) 
                

                self.publish_velocity_polytope_Hrep_A.publish(self.p_Hrep_A)
                self.publish_velocity_polytope_Hrep_b.publish(self.p_Hrep_b)
                self.publish_velocity_polytope_Hrep_size.publish(self.p_Hrep_size)

                self.cm_est = cm_index
                #print('self.cm_est',self.cm_est)
                #print('facet_vertex_idx',facet_vertex_idx)
                #print('capacity_margin_proj_vertex',capacity_margin_proj_vertex)
                #print('capacity_margin_proj_vertex_est',capacity_margin_proj_vertex_est)
                

                # Only for visualization - Polytope at end-effector - No physical significance
                #ef_pose = pykdl_util_kin.forward(self.q_in_numpy)[:3,3]

                
                

                # Get end-effector of the robot here for the polytope offset

                #ef_pose = position_70(q_in)

                ef_pose = pos_act
                ef_pose = ef_pose[:,0]
                #print('ef_pose is',ef_pose)
                #input('stop to test ef_pose')

                ########### Actual POlytope plot ###########################################################################
                # Publish polytope faces
                polyArray_message = self.publish_velocity_polytope.publish(create_polytopes_msg(polytope_verts, polytope_faces, \
                                                                                                    ef_pose,"base_link", scaling_factor))
                
                
                ### Desired polytope set - Publish

                DesiredpolyArray_message = self.publish_desired_polytope.publish(create_polytopes_msg(desired_polytope_verts, desired_polytope_faces, \
                                                                                                    ef_pose,"base_link", scaling_factor))


                ### Vertex for capacity margin on the Desired Polytope
                #print('facet_vertex_idx',facet_vertex_idx)
                closest_vertex = self.cartesian_desired_vertices[facet_vertex_idx[0,1]]
                #print('closest_vertex',closest_vertex)

                CapacityvertexArray_message = self.publish_vertex_capacity.publish(create_capacity_vertex_msg(closest_vertex, \
                                                                                            ef_pose, "base_link", scaling_factor))
                
                ### Vertex for capacity margin on the Available Polytope
                CapacityprojvertexArray_message = self.publish_vertex_proj_capacity.publish(create_capacity_vertex_msg(capacity_margin_proj_vertex, \
                                                                                            ef_pose, "base_link", scaling_factor))


                ### Vertex for capacity margin on the Available Polytope
                ActualposevertexArray_message = self.publish_vertex_pose.publish(create_capacity_vertex_msg(ef_pose, \
                                                                                            array([0,0,0]), "base_link", 1))
                
                ### Vertex for capacity margin on the Available Polytope
                '''
                DesiredposevertexArray_message = self.publish_vertex_desired_pose.publish(create_capacity_vertex_msg(self.pos_reference, \
                                                                                        array([0,0,0]), "base_link", 1))
                '''
                ### Plane for capacity margin 
                #print('capacity_faces',capacity_faces)

                ### Vertex for capacity margin on the Available Polytope
                CapacitymarginactualArray_message = self.publish_capacity_margin_actual.publish(create_segment_msg(closest_vertex, \
                                                    capacity_margin_proj_vertex,ef_pose, "base_link", scaling_factor))
                
                capacityArray_message = self.publish_capacity_margin_polytope.publish(create_polytopes_msg(polytope_verts, capacity_faces, \
                                                                                                    ef_pose,"base_link", scaling_factor))


                ########### Estimated Polytope plot ###########################################################################
                
                # Publish polytope faces
                EstpolyArray_message = self.publish_velocity_polytope_est.publish(create_polytopes_msg(polytope_verts_est, polytope_faces_est, \
                                                                                                    ef_pose,"base_link", scaling_factor))
                
                

                ### Vertex for capacity margin on the Available Polytope
                EstCapacityprojvertexArray_message = self.publish_vertex_proj_capacity_est.publish(create_capacity_vertex_msg(capacity_margin_proj_vertex_est, \
                                                                                            ef_pose, "base_link", scaling_factor))


                ### Vertex for capacity margin on the Available Polytope
                EstCapacitymarginactualArray_message = self.publish_capacity_margin_actual_est.publish(create_segment_msg(closest_vertex, \
                                                    capacity_margin_proj_vertex_est,ef_pose, "base_link", scaling_factor))
                

                EstcapacityArray_message = self.publish_capacity_margin_polytope_est.publish(create_polytopes_msg(polytope_verts_est, capacity_faces_est, \
                                                                                                    ef_pose,"base_link", scaling_factor))

                


                ### Vertex 
                
                ##############################################################################################################
                
                
                #print('facet_vertex_idx',facet_vertex_idx)
                
                mutex.release()

    def rosbag_start_record_callback(self,start_rosbag_record):
        if(start_rosbag_record.data):
            try:
                startRosbagRecordSrv = rospy.ServiceProxy('/data_recording/start_recording', Trigger, persistent=True)
                resp2 = startRosbagRecordSrv()
                

                if resp2.success:
                    print("Rosbag Start Recording Service started ")
                    
                    
                    return resp2.success
                else:
                    print("Rosbag Start Recording Service Error ")
                    return resp2.success

                
            except rospy.ServiceException:
                print("Rosbag Start Recording Service call failed")
    

    def rosbag_stop_record_callback(self,stop_rosbag_record):
        if(stop_rosbag_record.data):
            try:
                stopRosbagRecordSrv = rospy.ServiceProxy('/data_recording/stop_recording', Trigger, persistent=True)
                resp3 = stopRosbagRecordSrv()
                

                if resp3.success:
                    print("Rosbag Stop Recording Service started ")
                    
                    
                    return resp3.success
                else:
                    print("Rosbag Stop Recording Service Error ")
                    return resp3.success

                
            except rospy.ServiceException:
                print("Rosbag Stop Recording Service call failed")

    def fixture_line_topic_callback(self,line_detection_bool):
            
        if(line_detection_bool.data):
            try:
                linedetectionSrv = rospy.ServiceProxy('fixture_line_detection_server', Trigger, persistent=True)
                resp = linedetectionSrv()
                

                if resp.success:
                    print("Fixture line detection Service Called ")
                    
                    
                    return resp.success
                else:
                    print("Fixture line detection Service Error ")
                    return resp.success

                
            except rospy.ServiceException:
                print("Fixture line detection Service call failed")


    def fixture_line_segment_callback(self,line_segments_array):
        
        '''
        
            self.fixture_lines = []
        '''
         
        self.fixture_lines.append(line_segments_array.data)
        #print('fixture lines are',self.fixture_lines)

        fixture_line_data = array(line_segments_array.data)
        #print('fixture_data',fixture_line_data)
        #print(fixture_line_data[3:6])
        v1 = fixture_line_data[3:6] - fixture_line_data[0:3]

        #print('v1 is',v1)
        v2 = array([0,1,0])
        normal_fixture = cross(v2,v1)
        
        fixture_line_1 = zeros(shape=(6))
        fixture_line_2 = zeros(shape=(6))

        fixture_line_1[0:3] = fixture_line_data[0:3] + self.dist_plane*(normal_fixture)
        fixture_line_1[3:6] = fixture_line_data[3:6] + self.dist_plane*(normal_fixture)
        
        fixture_points_1 = V_unit(cross(v2,normal_fixture)) + fixture_line_data[0:3]
        fixture_points_2 = V_unit(cross(v1,normal_fixture)) + fixture_line_data[3:6]
        plane_verts = vstack((fixture_line_data[0:3],fixture_line_data[3:6],fixture_points_1,fixture_points_2))
        self.plane_verts.append(plane_verts)
        self.pose_verts.append(fixture_line_data[0:3])
        

        append_arr = hstack((normal_fixture,fixture_line_1))
        self.fixture_planes.append(append_arr)
        fixture_line_2[0:3] = fixture_line_data[0:3] - self.dist_plane*(normal_fixture)
        fixture_line_2[3:6] = fixture_line_data[3:6] - self.dist_plane*(normal_fixture)
        append_arr = hstack((normal_fixture,fixture_line_2))
        self.fixture_planes.append(append_arr)
            
            #print('fixture planes are',self.fixture_planes)
        if len(self.fixture_lines) > 3:
            for i in range(len(self.plane_verts)):
                    if i == 0:
                        self.publish_fixture_plane_plot_1.publish(create_plane_msg(self.plane_verts[i],array(self.pose_verts[i])+array([-0.0,0,-0.02]),"base_link",scaling_factor=10))

                    if i ==1:
                        self.publish_fixture_plane_plot_2.publish(create_plane_msg(self.plane_verts[i],array(self.pose_verts[i])+array([-0.03,0,0.0]),"base_link",scaling_factor=10))

                    if i == 2:
                        self.publish_fixture_plane_plot_3.publish(create_plane_msg(self.plane_verts[i],array(self.pose_verts[i])+array([-0.03,0,0.0]),"base_link",scaling_factor=10))
                    print('plotting vertices planes')




    '''
    def ft_kuka_callback(self, ft_data_kuka):

        self.ft_sensor = ft_data_kuka


        force_comp = self.ft_sensor.wrench.force
        torque_comp = self.ft_sensor.wrench.torque
        direction_force = cross(array([torque_comp.x,torque_comp.y,torque_comp.z]),\
                                 array([force_comp.x,force_comp.y,force_comp.z]))
        

        direction_force = direction_force*(norm(direction_force)**(-1))
        #print('force_comp',force_comp)

        #print('torque_comp',torque_comp)

        #print('direction of force is',direction_force)

        #self.force_direction = cross(self.ft_sensor)
        # print('Ft data is',self.ft_sensor)

        # self.ft_sensor_gc =
    '''
    def button_kuka_update_callback(self, geo_kuka_buttons):

        self.button_kuka_state = [
            geo_kuka_buttons.buttons[0], geo_kuka_buttons.buttons[1]]
        # print('buttons: ', self.button_kuka_state)

    def kuka_digital_output_service(self, out1, out2, out3, out4, out5, out6, out7, out8):
        # print('Deadly inside')
        rospy.wait_for_service(
            '/kuka_hardware_interface/write_8_digital_outputs', timeout=None)
        print('I crossed timeout')
        try:
            write_8_outputs_func = rospy.ServiceProxy(
                '/kuka_hardware_interface/write_8_digital_outputs', write_8_outputs, persistent=True)
            resp1 = write_8_outputs_func(out1, out2, out3, out4, out5, out6, out7, out8)
            print("I have actuated it ")
            return resp1
            # rospy.spin()
            # resp1 = write_8_bool_outputs_resp(False,False,False,False,False,False,False,False)
        except rospy.ServiceException:
            print("Service call failed: KUKA_Digital_outputs")

    def collision_state_callback(self, collision_state):
        self.current_collision_state = collision_state.data
        # print('collision_state is', self.current_collision_state)

    def dexterous_mode_callback(self, dexterous_mode):

        self.dexterous_mode_state = dexterous_mode.data
    
    def hot_wire_mode_on_callback(self,hot_wire_state_on):

        if (hot_wire_state_on.data):
            # if (self.change_gripper_state and self.changing_state):
            print("\n\n====================Switchin ON Hot Wire ===============\n\n")
            
            self.hot_wire_toggle = True
            self.kuka_digital_output_service(
                False, False, True, False, False, False, False, False)
            rospy.sleep(0.75)
    
    def active_cam_callback(self,active_cam):

        '''
        if (active_cam_on.data):
            self.cam_rot = self.cam_rot.RPY(0,0,1.57)
        else:
            self.cam_rot = self.cam_rot.RPY(0,0,0)
        '''

        if (active_cam.data == 0):
            self.cam_rot = self.cam_rot.RPY(0,0,0)
            print('i am in front camera')
        if (active_cam.data == 1):
            self.cam_rot = self.cam_rot.RPY(0,0,1.57)
            print('i am in side camera')
        if (active_cam.data == 2):
            self.cam_rot = self.cam_rot.RPY(0,-1.57,0)
            print('i am in robot camera')
            


    def hot_wire_mode_off_callback(self,hot_wire_state_off):


        if (hot_wire_state_off.data) and self.hot_wire_toggle:
            print("\n\n====================Switchin OFF Hot Wire ===============\n\n")
            self.kuka_digital_output_service(
                False, False, False, False, False, False, False, False)
            rospy.sleep(0.75)
        
            self.hot_wire_toggle = False
        
            
            

            # self.changing_state=False
    def gripper_actuate_callback(self, change_gripper_states):

                        # Gripper close here
        if (change_gripper_states.data):
        # if (self.change_gripper_state and self.changing_state):
            print("\n\n====================Changing state now===============\n\n")
            if(self.gripper_closed):
                print('Open Gripper')

                self.kuka_digital_output_service(
                    True, False, False, False, False, False, False, False)
                rospy.sleep(0.75)
                self.kuka_digital_output_service(
                    False, False, False, False, False, False, False, False)
                self.gripper_closed = False
                print('Finish Open Gripper')
            else:
                print('Close Gripper')

                # self.kuka_digital_output_service(False,False,False,False,False,False,False,False)
                self.kuka_digital_output_service(
                    False, True, False, False, False, False, False, False)
                rospy.sleep(0.75)
                self.kuka_digital_output_service(
                    False, False, False, False, False, False, False, False)
                self.gripper_closed = True
                print('Finish Close Gripper')
            print('\n\n====================Changing state back to false===============\n\n')
            # self.changing_state=False

    def geo_orientation_x_callback(self, geomagic_pen_wx):
        self.geo_pose_orientation_x = geomagic_pen_wx

    def geo_orientation_y_callback(self, geomagic_pen_wy):
        self.geo_pose_orientation_y = geomagic_pen_wy

    def geo_orientation_z_callback(self, geomagic_pen_wz):
        self.geo_pose_orientation_z = geomagic_pen_wz

    '''
    def quaternion_rotation(self,quaternion_1, quaternion_2):

        # Function to find the minimum angle of rotation between two quaternions

        from math import atan2
        from numpy import rad2deg,array,cross,dot,hstack
        from numpy.linalg import norm

        # Return theta in degrees here

        # https://www.mathworks.com/help/aeroblks/quaternionmultiplication.html

        # q1 is the conjugate - Only when quaternion is a unit quaternion - Unity provides unit quaternion
        q1 = array([quaternion_1.pose.orientation.w, quaternion_1.pose.orientation.x,
                   quaternion_1.pose.orientation.y, quaternion_1.pose.orientation.z])
        q2 = array([quaternion_2.pose.orientation.w, -quaternion_2.pose.orientation.x, - \
                   quaternion_2.pose.orientation.y, -quaternion_2.pose.orientation.z])
        # Quaternion product


        q0 = array([0.0, 0.0, 0.0, 0.0])

        q0[0] = q2[0]*q1[0] - q2[1]*q1[1] - q2[2]*q1[2] - q2[3]*q1[3]
        q0[1] = q2[0]*q1[1] + q2[1]*q1[0] - q2[2]*q1[3] + q2[3]*q1[2]
        q0[2] = q2[0]*q1[2] + q2[1]*q1[3] + q2[2]*q1[0] - q2[3]*q1[1]
        q0[3] = q2[0]*q1[3] - q2[1]*q1[2] + q2[2]*q1[1] + q2[3]*q1[0]


        return 2*atan2( norm(array([q0[1],q0[2],q0[3]])),q0[0])

    '''
    

    def cut_xy_callback(self,cut_xy_state):
        self.cut_xy_active = cut_xy_state.data


    def cut_xz_callback(self,cut_xz_state):
        self.cut_xz_active = cut_xz_state.data
    
    def kuka_callback(self, kuka_qin_joints):
        # Callback for getting current joint states of fanuc

        self.q_in[0] = kuka_qin_joints.position[0]
        self.q_in_numpy[0] = kuka_qin_joints.position[0]
        self.q_in[1] = kuka_qin_joints.position[1]
        self.q_in_numpy[1] = kuka_qin_joints.position[1]
        self.q_in[2] = kuka_qin_joints.position[2]
        self.q_in_numpy[2] = kuka_qin_joints.position[2]
        self.q_in[3] = kuka_qin_joints.position[3]
        self.q_in_numpy[3] = kuka_qin_joints.position[3]
        self.q_in[4] = kuka_qin_joints.position[4]
        self.q_in_numpy[4] = kuka_qin_joints.position[4]
        self.q_in[5] = kuka_qin_joints.position[5]
        self.q_in_numpy[5] = kuka_qin_joints.position[5]
        print('self.q_in',self.q_in)
    
    def geo_to_kuka_callback(self, geo_kuka_twist):
        # Callback for Twist control
        # rospy.wait_for_service('write_8_outputs')

        if self.EnterFlag == True:

            #self.kuka_joint_states.position = [0, 0, 0, 0, 0, 0]
            self.EnterFlag = False
        for k in range(self.no_of_joints):
            self.kuka_joint_states.position[k] = self.q_in[k]

        dt = geo_kuka_twist.header.stamp.secs

        # Gripper state message

        self.gripper_state_msg.data = self.gripper_closed
        self.kuka_gripper_current_state.publish(self.gripper_state_msg)

        ## Always be publishing the wrench stamped message for fixtures
        ## Bad implementation TODO put the wrench stamped to a seperate ROS node and run the Pykdl chain there


        ### Fixture wrench stamped subscriber here
        # Run Forward kinematics to get EE frame

        wrench_msg = WrenchStamped()
        self.vel_fk_solver.JntToCart(self.q_in, self.eeFrame)
        #print('end-effector is',self.eeFrame)
        #print('end-effector translation is',self.eeFrame.p)

        #print('self.eeFrame.p[0][0]',self.eeFrame.p[0])
        self.end_effector_pos[0] = copy(self.eeFrame.p[0])
        self.end_effector_pos[1] = copy(self.eeFrame.p[1])
        self.end_effector_pos[2] = copy(self.eeFrame.p[2])

        pointmsg = PointStamped()
        pointmsg.header = geo_kuka_twist.header
        pointmsg.point.x = self.end_effector_pos[0]
        pointmsg.point.y = self.end_effector_pos[1]
        pointmsg.point.z = self.end_effector_pos[2]
        self.end_effector_position.publish(pointmsg)
        
        #print('self.end_effector_pos',self.end_effector_pos)
        if len(self.fixture_lines) > 2:
            #print('self.fixture_planes',self.fixture_planes[0][3:6])
            #print('proj_point_plane(self.fixture_planes[0,0:3],self.fixture_planes[0,3:6],self.eeFrame.p)',proj_point_plane(self.fixture_planes[0][0:3],self.fixture_planes[0][3:6],self.end_effector_pos))
            [p1,d1] = proj_point_plane(self.fixture_planes[0][0:3],self.fixture_planes[0][3:6],self.end_effector_pos)
            dist_ef_plane_1 = norm(self.end_effector_pos - array(p1))
            [p2,d2] = proj_point_plane(self.fixture_planes[1][0:3],self.fixture_planes[1][3:6],self.end_effector_pos)
            dist_ef_plane_2 = norm(self.end_effector_pos - array(p2))
            [p3,d3] = proj_point_plane(self.fixture_planes[2][0:3],self.fixture_planes[2][3:6],self.end_effector_pos)
            dist_ef_plane_3 = norm(self.end_effector_pos - array(p3))
            [p4,d4] = proj_point_plane(self.fixture_planes[3][0:3],self.fixture_planes[3][3:6],self.end_effector_pos)
            dist_ef_plane_4 = norm(self.end_effector_pos - array(p4))
            [p5,d5] = proj_point_plane(self.fixture_planes[4][0:3],self.fixture_planes[4][3:6],self.end_effector_pos)
            dist_ef_plane_5 = norm(self.end_effector_pos - array(p5))
            [p6,d6] = proj_point_plane(self.fixture_planes[5][0:3],self.fixture_planes[5][3:6],self.end_effector_pos)
            dist_ef_plane_6 = norm(self.end_effector_pos - array(p6))
            
            #print('dist_ef_plane_1',dist_ef_plane_1)
            #print('dist_ef_plane_2',dist_ef_plane_2)
            #print('dist_ef_plane_3',dist_ef_plane_3)
            #print('fixture_plannes',self.fixture_planes)
            
            
            if (dist_ef_plane_1 < self.dist_tol )or (dist_ef_plane_2 < self.dist_tol) or (dist_ef_plane_3 < self.dist_tol) or (dist_ef_plane_4 < self.dist_tol) or\
                  (dist_ef_plane_5 < self.dist_tol) or (dist_ef_plane_6 < self.dist_tol):

                print('Inside fixture wrench zone')
                #input('stop')
                if dist_ef_plane_1 < self.dist_tol:
                    # Fixture zone
                    wrench_msg.header = geo_kuka_twist.header
                    force_vector = self.fixture_planes[0][0:3]
                    if d1<0:
                        dist_ef_plane_1 = dist_ef_plane_1*1
                    else:
                        dist_ef_plane_1 = -dist_ef_plane_1*10
                    force_vector = 1*force_vector*dist_ef_plane_1
                    wrench_msg.wrench.force.x = force_vector[0]
                    wrench_msg.wrench.force.y = force_vector[1]
                    wrench_msg.wrench.force.z = force_vector[2]
                    wrench_msg.wrench.torque.x = 0
                    wrench_msg.wrench.torque.y = 0
                    wrench_msg.wrench.torque.z = 0
                    self.fixture_plane_wrench.publish(wrench_msg)
                    #self.fixture_plane_distance.publish(dist_ef_plane_1)
                '''
                if dist_ef_plane_2 < self.dist_tol:
                    # Fixture zone
                    wrench_msg.header = geo_kuka_twist.header
                    force_vector = self.fixture_planes[1][0:3]
                    if d2>0:
                        dist_ef_plane_2 = dist_ef_plane_2*2
                    #force_vector = -1*force_vector
                    force_vector = force_vector*dist_ef_plane_2
                    wrench_msg.wrench.force.x = force_vector[0]
                    wrench_msg.wrench.force.y = force_vector[1]
                    wrench_msg.wrench.force.z = force_vector[2]
                    wrench_msg.wrench.torque.x = 0
                    wrench_msg.wrench.torque.y = 0
                    wrench_msg.wrench.torque.z = 0
                    self.fixture_plane_wrench.publish(wrench_msg)
                    #self.fixture_plane_distance.publish(dist_ef_plane_2)
                '''    
                if dist_ef_plane_3 < self.dist_tol:
                    # Fixture zone
                    wrench_msg.header = geo_kuka_twist.header
                    force_vector = self.fixture_planes[2][0:3]
                    if d3<0:
                        dist_ef_plane_3 = dist_ef_plane_3*1
                    else:
                        dist_ef_plane_3 = -dist_ef_plane_3*10
                    force_vector = -1*force_vector*dist_ef_plane_3
                    wrench_msg.wrench.force.x = force_vector[0]
                    wrench_msg.wrench.force.y = force_vector[1]
                    wrench_msg.wrench.force.z = force_vector[2]
                    wrench_msg.wrench.torque.x = 0
                    wrench_msg.wrench.torque.y = 0
                    wrench_msg.wrench.torque.z = 0
                    self.fixture_plane_wrench.publish(wrench_msg)
                    #self.fixture_plane_distance.publish(dist_ef_plane_3)
                '''    
                if dist_ef_plane_4 < self.dist_tol:
                    # Fixture zone
                    wrench_msg.header = geo_kuka_twist.header
                    force_vector = self.fixture_planes[3][0:3]
                    if d4 > 0:
                        dist_ef_plane_4 = dist_ef_plane_4*2
                    #force_vector = -1*force_vector
                    force_vector = force_vector*dist_ef_plane_4
                    wrench_msg.wrench.force.x = force_vector[0]
                    wrench_msg.wrench.force.y = force_vector[1]
                    wrench_msg.wrench.force.z = force_vector[2]
                    wrench_msg.wrench.torque.x = 0
                    wrench_msg.wrench.torque.y = 0
                    wrench_msg.wrench.torque.z = 0
                    self.fixture_plane_wrench.publish(wrench_msg)
                    #self.fixture_plane_distance.publish(dist_ef_plane_4)
                '''    
                if dist_ef_plane_5 < self.dist_tol:
                    # Fixture zone
                    wrench_msg.header = geo_kuka_twist.header
                    force_vector = self.fixture_planes[4][0:3]
                    if d5<0:
                        dist_ef_plane_5 = dist_ef_plane_5*1
                    else:
                        dist_ef_plane_5 = -dist_ef_plane_5*10

                    force_vector = -1*force_vector*dist_ef_plane_5
                    wrench_msg.wrench.force.x = force_vector[0]
                    wrench_msg.wrench.force.y = force_vector[1]
                    wrench_msg.wrench.force.z = force_vector[2]
                    wrench_msg.wrench.torque.x = 0
                    wrench_msg.wrench.torque.y = 0
                    wrench_msg.wrench.torque.z = 0
                    self.fixture_plane_wrench.publish(wrench_msg)
                    #self.fixture_plane_distance.publish(dist_ef_plane_5
                '''    
                if dist_ef_plane_6 < self.dist_tol:
                    # Fixture zone
                    wrench_msg.header = geo_kuka_twist.header
                    force_vector = self.fixture_planes[5][0:3]
                    if d6 > 0:
                        dist_ef_plane_6 = dist_ef_plane_6*2
                    #force_vector = -1*force_vector
                    force_vector = force_vector*dist_ef_plane_6
                    wrench_msg.wrench.force.x = force_vector[0]
                    wrench_msg.wrench.force.y = force_vector[1]
                    wrench_msg.wrench.force.z = force_vector[2]
                    wrench_msg.wrench.torque.x = 0
                    wrench_msg.wrench.torque.y = 0
                    wrench_msg.wrench.torque.z = 0
                    self.fixture_plane_wrench.publish(wrench_msg)
                    #self.fixture_plane_distance.publish(dist_ef_plane_6)
                '''    
            else:
                #wrench_msg.wrench.force = dist_ef_plane_3*0.0*self.fixture_planes[2][0:3]
                wrench_msg.wrench.force.x = 0
                wrench_msg.wrench.force.y = 0
                wrench_msg.wrench.force.z = 0
                wrench_msg.wrench.torque.x = 0
                wrench_msg.wrench.torque.y = 0
                wrench_msg.wrench.torque.z = 0
                self.fixture_plane_wrench.publish(wrench_msg)

            

        ##### Linear Velocity control here ###############################

        if (dt > 0.000001) and (self.dexterous_mode_state == False) and (self.button_kuka_state[0] == 1) and (self.button_kuka_state[1] == 0):

            msg = JointTrajectory()
            kuka_joints = JointTrajectoryPoint()
            msg.joint_names = self.robot_joint_names
            # msg.header = geo_kuka_twist.header

            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'base_kuka'

            if self.flag_linear == False:
                self.flag_linear = True
                self.lin_vel_x_prev = 0
                self.lin_vel_y_prev = 0
                self.lin_vel_z_prev = 0

            self.cartesian_twist.vel[0] = self.lin_vel_x_prev + 0.000004 * \
                (geo_kuka_twist.twist.linear.x - self.lin_vel_x_prev)
            self.cartesian_twist.vel[1] = self.lin_vel_y_prev + 0.000004 * \
                (geo_kuka_twist.twist.linear.y - self.lin_vel_y_prev)
            self.cartesian_twist.vel[2] = self.lin_vel_z_prev + 0.000004 * \
                (geo_kuka_twist.twist.linear.z - self.lin_vel_z_prev)

            self.lin_vel_x_prev = self.cartesian_twist.vel[0]
            self.lin_vel_y_prev = self.cartesian_twist.vel[1]
            self.lin_vel_z_prev = self.cartesian_twist.vel[2]

            self.cartesian_twist.vel[0] = self.cartesian_twist.vel[0]*self.vel_scale[0]
            self.cartesian_twist.vel[1] = self.cartesian_twist.vel[1]*self.vel_scale[1]
            self.cartesian_twist.vel[2] = self.cartesian_twist.vel[2]*self.vel_scale[2]
            
            if self.cut_xy_active:
                self.cartesian_twist.vel[2] = 0.0
            if self.cut_xz_active:
                self.cartesian_twist.vel[1] = 0.0

                        # Collision state move extremely slow
            if (self.current_collision_state == True) and (self.leave_flag_collision_zone == False):

                # self.cartesian_wrench.wrench.force.x = 0.5 + self.cartesian_wrench.wrench.force.x
                self.leave_flag_collision_zone = False

                self.cartesian_twist.vel[0] = self.cartesian_twist.vel[0]*0.01
                self.cartesian_twist.vel[1] = self.cartesian_twist.vel[1]*0.01
                self.cartesian_twist.vel[2] = self.cartesian_twist.vel[2]*0.01

                self.lin_vel_x_prev = self.cartesian_twist.vel[0]
                self.lin_vel_y_prev = self.cartesian_twist.vel[1]
                self.lin_vel_z_prev = self.cartesian_twist.vel[2]

                print('reducing speed in collision zone')

                self.collision_zone_counter += 1
                print('self.collision_zone_counter', self.collision_zone_counter)
                if self.collision_zone_counter > 400:
                    self.collision_zone_counter = 0
                    self.leave_flag_collision_zone = True
                    self.current_collision_state = False
                    print('I have exited collision zone ###########################################',
                          self.collision_zone_counter)

            else:
                print('Normal zone')



            self.cartesian_twist.rot[0] = 0
            self.cartesian_twist.rot[1] = 0
            self.cartesian_twist.rot[2] = 0

            # Impedance control loop here
            # cartesian_vel = Ja^T*Km*Ja*qdot
            self.impedance_control_loop = False
            while self.impedance_control_loop:

                self.vel_ik_solver.CartToJnt(
                    self.q_in, self.cartesian_twist, self.qdot_out)
                self.jacobian_solver.JntToJac(self.q_in, self.fk_jacobian)

                # twist0 = array([list(self.fk_jacobian.getColumn(0))])
                # twist1 = array([list(self.fk_jacobian.getColumn(1))])
                # twist2 = array([list(self.fk_jacobian.getColumn(2))])
                # twist3 = array([list(self.fk_jacobian.getColumn(3))])
                # twist4 = array([list(self.fk_jacobian.getColumn(4))])
                # twist5 = array([list(self.fk_jacobian.getColumn(5))])

                # JT = vstack(([twist0,twist1,twist2,twist3,twist4,twist5]))

                # JT = transpose(J)

                # self.ik_jacobian[0,0] = 1000.0
                # print('self.ik_jacobian',self.ik_jacobian)
                # print('self.fk_jacobian_0 is',self.fk_jacobian[0,0])
                # self.JT_Twist_0 = PyKDL.Twist(JT[0,:])
                # self.JT_Twist_1 = PyKDL.Twist(JT[1,:])
                # self.JT_Twist_2 = PyKDL.Twist(JT[2,:])
                # self.JT_Twist_3 = PyKDL.Twist(JT[3,:])
                # self.JT_Twist_4 = PyKDL.Twist(JT[4,:])
                # self.JT_Twist_5 = PyKDL.Twist(JT[5,:])

                print('Forward jacobian is', self.fk_jacobian)

                # Manual numpy Jacobian

                for i in range(6):
                    for j in range(6):
                        self.fk_jacobian_arr[i, j] = self.fk_jacobian[i, j]

                print('shape(array(self.fk_jacobian_arr))',
                      shape(array(self.fk_jacobian_arr)))
                polytope_verts, polytope_faces = pycapacity_polytope(
                    array(self.fk_jacobian_arr), self.q_max, self.q_min)
                print('polytope_verts', polytope_verts)
                print('polytope_faces', polytope_faces)
                # print('IK sasd jacobian is',ttt)

                # Manual transpose here
                # KDL doesnt provide transpose
                for i in range(6):
                    for j in range(6):
                        self.ik_jacobian_K[i, j] = self.fk_jacobian[j, i]*self.K_C[i, j]

                print('IK jacobian is', self.ik_jacobian_K)

                # Impedance control
                # cartesian_velocity = Ja^T*Kc*Ja*q
                PyKDL.MultiplyJacobian(self.fk_jacobian, self.qdot_out, self.twist_output)

                for i in range(6):
                    self.test_output_twist[i] = self.ik_jacobian_K[i, i]*self.twist_output[i]
                # PyKDL.MultiplyJacobian(self.ik_jacobian_K,self.twist_output,self.test_output_twist)

                #print('twist_output', self.test_output_twist)
                #print('self.cartesian_twist', self.cartesian_twist)

                self.cartesian_twist = self.test_output_twist

                # print('self.get_jacobian_kuka Inv',self.vel_ik_solver.getSVDResult())

                self.impedance_control_loop = False

            
        
        

            #print('self.baseFrame is',self.baseFrame)
            
            R_B_cam = self.cam_rot
            #print('Rotation of cam is',R_B_cam)
            self.cartesian_twist = R_B_cam*self.cartesian_twist
            self.vel_ik_solver.CartToJnt(self.q_in, self.cartesian_twist, self.qdot_out)

            for no_of_joints in range(kdl_chain.getNrOfJoints()):
                if (abs(self.qdot_out[no_of_joints]) - abs(self.qdot_limit[no_of_joints])) > 0.050000:
                    for i in range(kdl_chain.getNrOfJoints()):
                        self.qdot_out[no_of_joints] = 0.0
                        print('Torque error')
                        input('wait here - Torque error')
                    return
            for no_of_joints in range(kdl_chain.getNrOfJoints()):
                if ((self.kuka_joint_states.position[no_of_joints] + self.qdot_out[no_of_joints]) >
                self.q_upper_limit[no_of_joints]) or ((self.kuka_joint_states.position[no_of_joints] + self.qdot_out[no_of_joints])
                 < self.q_lower_limit[no_of_joints]):
                    self.qdot_out[no_of_joints] = 0

                # self.kuka_joint_states.position[no_of_joints] = self.kuka_joint_states.position[no_of_joints] + self.qdot_out[no_of_joints]

                # self.q_in[no_of_joints] = self.kuka_joint_states.position[no_of_joints]

            for no_of_joints in range(kdl_chain.getNrOfJoints()):
                self.kuka_joint_states.position[no_of_joints] = self.kuka_joint_states.position[no_of_joints] + \
                    self.qdot_out[no_of_joints]
                # self.q_in[no_of_joints] = self.kuka_joint_states.position[no_of_joints]

            self.kuka_joint_states.header = geo_kuka_twist.header
            kuka_joints.positions = self.kuka_joint_states.position

            kuka_joints.accelerations = []
            kuka_joints.effort = []
            kuka_joints.time_from_start = rospy.Duration(0, 400000000)

            # kuka_joints.time_from_start.nsecs = 778523489 # Taken from rqt_joint_trajectory_controller
            msg.points.append(kuka_joints)

            # self.kuka_joint_states_dummy_publisher.publish(msg)
            self.kuka_joint_states_publisher.publish(msg)



            # self.get_jacobian_kuka.JntToJac(self.q_in,self.jacob_kuka)
            # twist1 = array([list(self.jacob_kuka.getColumn(1))])
            # twist2 = array([list(self.jacob_kuka.getColumn(2))])
            # twist3 = array([list(self.jacob_kuka.getColumn(3))])
            # print('shape(twist1)',shape(twist1))
            # print('twist1',twist1)

            # screw_matrix = hstack((transpose(twist1),transpose(twist2),transpose(twist3)))
            # det_screw= det(screw_matrix[0:3,:])
            # print('det_jacob_matrix',det_screw)
            # print('matrix_rank',)
            # self.eeFrame
            # print('ee frame is',self.eeFrame)
            # kdl_2_tf_frame = tf_c.toTf(self.eeFrame)

            # self.eeFrame[0,3] = self.eeFrame[0,3] - 0.20

            # br1.sendTransform()
            # br1.sendTransform(kdl_2_tf_frame)

            # print('kdl_2_tf_frame')
            # print(kdl_2_tf_frame)

            # print('ee frame new is',self.eeFrame)
            # print('base_link frame is',/_tf_base_link)

            # print('br1 frame is',br1[0])

        ##### Angular Velocity control here ###############################
        elif (dt > 0.000001) and (self.dexterous_mode_state == False) and self.button_kuka_state[1] == 1 and self.button_kuka_state[0] == 0:

            msg = JointTrajectory()
            kuka_joints = JointTrajectoryPoint()
            msg.joint_names = ['joint_1', 'joint_2',
                'joint_3', 'joint_4', 'joint_5', 'joint_6']
            # msg.header = geo_kuka_twist.header

            msg.header.stamp = rospy.Time.now()

            msg.header.frame_id = 'tool0_fanuc'

            if self.flag_angular == False:
                self.flag_angular = True
                self.ang_vel_x_prev = 0
                self.ang_vel_y_prev = 0
                self.ang_vel_z_prev = 0
            self.cartesian_twist.rot[0] = self.ang_vel_x_prev + 0.00004 * \
                (geo_kuka_twist.twist.angular.x - self.ang_vel_x_prev)
            self.cartesian_twist.rot[1] = self.ang_vel_y_prev + 0.00004 * \
                (geo_kuka_twist.twist.angular.y - self.ang_vel_y_prev)
            self.cartesian_twist.rot[2] = self.ang_vel_z_prev + 0.00004 * \
                (geo_kuka_twist.twist.angular.z - self.ang_vel_z_prev)

            self.ang_vel_x_prev = self.cartesian_twist.rot[0]
            self.ang_vel_y_prev = self.cartesian_twist.rot[1]
            self.ang_vel_z_prev = self.cartesian_twist.rot[2]

            self.cartesian_twist.rot[0] = self.cartesian_twist.rot[0] * \
                self.angular_vel_scale[0]
            self.cartesian_twist.rot[1] = self.cartesian_twist.rot[1] * \
                self.angular_vel_scale[1]
            self.cartesian_twist.rot[2] = self.cartesian_twist.rot[2] * \
                self.angular_vel_scale[2]

            if (self.current_collision_state == True) and (self.leave_flag_collision_zone == False):

                # self.cartesian_wrench.wrench.force.x = 0.5 + self.cartesian_wrench.wrench.force.x
                self.leave_flag_collision_zone = False

                self.cartesian_twist.rot[0] = self.cartesian_twist.vel[0]*0.01
                self.cartesian_twist.rot[1] = self.cartesian_twist.vel[1]*0.01
                self.cartesian_twist.rot[2] = self.cartesian_twist.vel[2]*0.01

                self.ang_vel_x_prev = self.cartesian_twist.vel[0]
                self.ang_vel_y_prev = self.cartesian_twist.vel[1]
                self.ang_vel_z_prev = self.cartesian_twist.vel[2]

                print('reducing speed in collision zone')

                self.collision_zone_counter += 1
                print('self.collision_zone_counter', self.collision_zone_counter)
                if self.collision_zone_counter > 400:
                    self.collision_zone_counter = 0
                    self.leave_flag_collision_zone = True
                    self.current_collision_state = False
                    print('I have exited collision zone ###########################################',
                          self.collision_zone_counter)

            self.cartesian_twist.vel[0] = 0.0
            self.cartesian_twist.vel[1] = 0.0
            self.cartesian_twist.vel[2] = 0.0
            self.vel_fk_solver.JntToCart(self.q_in, self.eeFrame)
            R_B_tool = self.eeFrame.M ### R^B_t
            #print('R_B_tool',R_B_tool)
            cartesian_twist_transform = R_B_tool*self.cartesian_twist

            self.vel_ik_solver.CartToJnt(self.q_in, cartesian_twist_transform, self.qdot_out)

            

            for no_of_joints in range(kdl_chain.getNrOfJoints()):
                if (abs(self.qdot_out[no_of_joints]) - self.qdot_limit[no_of_joints]) > 0.050000:
                    for i in range(kdl_chain.getNrOfJoints()):
                        self.qdot_out[i] = 0.0
                        print('Torque error')
                    return
                if ((self.kuka_joint_states.position[no_of_joints] + self.qdot_out[no_of_joints]) >
                self.q_upper_limit[no_of_joints]) or ((self.kuka_joint_states.position[no_of_joints] + self.qdot_out[no_of_joints])
                 < self.q_lower_limit[no_of_joints]):
                    self.qdot_out[no_of_joints] = 0

                # self.kuka_joint_states.position[no_of_joints] = self.kuka_joint_states.position[no_of_joints] + self.qdot_out[no_of_joints]

                # self.q_in[no_of_joints] = self.kuka_joint_states.position[no_of_joints]

            for no_of_joints in range(kdl_chain.getNrOfJoints()):
                self.kuka_joint_states.position[no_of_joints] = self.kuka_joint_states.position[no_of_joints] + \
                    self.qdot_out[no_of_joints]
                # self.q_in[no_of_joints] = self.kuka_joint_states.position[no_of_joints]

            self.kuka_joint_states.header = geo_kuka_twist.header
            kuka_joints.positions = self.kuka_joint_states.position

            kuka_joints.accelerations = []
            kuka_joints.effort = []
            kuka_joints.time_from_start = rospy.Duration(0, 400000000)

            # kuka_joints.time_from_start.nsecs = 778523489 # Taken from rqt_joint_trajectory_controller
            msg.points.append(kuka_joints)

            # self.vel_fk_solver.JntToCart(self.kuka_joint_states.position, self.eeFrame)

            # Converting KDL frame to tf frame here
            # br1 = tf_c.toTf(self.eeFrame)
            # tf.TransformBroadcaster([0,0,0],[0,0,0,1],)
            # print('br1 frame is',br1)

            # self.kuka_joint_states_dummy_publisher.publish(msg)
            self.kuka_joint_states_publisher.publish(msg)
        ##### Dexterous Mode- Control here ###############################
        # Both linear and angular velocity allowed in this zone
        elif (dt > 0.000001) and (self.dexterous_mode_state == True) and (self.button_kuka_state[1] == 0) and (self.button_kuka_state[0] == 1):
            msg = JointTrajectory()
            kuka_joints = JointTrajectoryPoint()
            msg.joint_names = ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6']
            msg.header.stamp = rospy.Time.now()
            # When in dexterous mode- Only access to move in the tool frame- No base Frame Caution !!!
            msg.header.frame_id='tool0_kuka'
            if self.flag_linear == False:
                self.flag_linear = True
                self.lin_vel_x_prev = 0
                self.lin_vel_y_prev = 0
                self.lin_vel_z_prev = 0

            if self.flag_angular == False:
                self.flag_angular = True
                self.ang_vel_x_prev = 0
                self.ang_vel_y_prev = 0
                self.ang_vel_z_prev = 0

            self.cartesian_twist.vel[0] = self.lin_vel_x_prev + 0.000004 * \
                (geo_kuka_twist.twist.linear.x - self.lin_vel_x_prev)
            self.cartesian_twist.vel[1] = self.lin_vel_y_prev + 0.000004 * \
                (geo_kuka_twist.twist.linear.y - self.lin_vel_y_prev)
            self.cartesian_twist.vel[2] = self.lin_vel_z_prev + 0.000004 * \
                (geo_kuka_twist.twist.linear.z - self.lin_vel_z_prev)

            self.lin_vel_x_prev = self.cartesian_twist.vel[0]
            self.lin_vel_y_prev = self.cartesian_twist.vel[1]
            self.lin_vel_z_prev = self.cartesian_twist.vel[2]

            self.cartesian_twist.vel[0] = self.cartesian_twist.vel[0]*self.dexterous_lin_vel_scale[0]
            self.cartesian_twist.vel[1] = self.cartesian_twist.vel[1]*self.dexterous_lin_vel_scale[1]
            self.cartesian_twist.vel[2] = self.cartesian_twist.vel[2]*self.dexterous_lin_vel_scale[2]

            self.cartesian_twist.rot[0] = self.ang_vel_x_prev + 0.00004 * \
                (geo_kuka_twist.twist.angular.x - self.ang_vel_x_prev)
            self.cartesian_twist.rot[1] = self.ang_vel_y_prev + 0.00004 * \
                (geo_kuka_twist.twist.angular.y - self.ang_vel_y_prev)
            self.cartesian_twist.rot[2] = self.ang_vel_z_prev + 0.00004 * \
                (geo_kuka_twist.twist.angular.z - self.ang_vel_z_prev)

            self.ang_vel_x_prev = self.cartesian_twist.rot[0]
            self.ang_vel_y_prev = self.cartesian_twist.rot[1]
            self.ang_vel_z_prev = self.cartesian_twist.rot[2]

            self.cartesian_twist.rot[0] = self.cartesian_twist.rot[0] * \
                self.dexterous_ang_vel_scale[0]
            self.cartesian_twist.rot[1] = self.cartesian_twist.rot[1] * \
                self.dexterous_ang_vel_scale[1]
            self.cartesian_twist.rot[2] = self.cartesian_twist.rot[2] * \
                self.dexterous_ang_vel_scale[2]


            ### Collision zone constraints are here
            if (self.current_collision_state == True) and (self.leave_flag_collision_zone == False):

                # self.cartesian_wrench.wrench.force.x = 0.5 + self.cartesian_wrench.wrench.force.x
                self.leave_flag_collision_zone = False

                self.cartesian_twist.vel[0] = self.cartesian_twist.vel[0]*0.01
                self.cartesian_twist.vel[1] = self.cartesian_twist.vel[1]*0.01
                self.cartesian_twist.vel[2] = self.cartesian_twist.vel[2]*0.01

                

                self.lin_vel_x_prev = self.cartesian_twist.vel[0]
                self.lin_vel_y_prev = self.cartesian_twist.vel[1]
                self.lin_vel_z_prev = self.cartesian_twist.vel[2]


                self.cartesian_twist.rot[0] = self.cartesian_twist.vel[0]*0.01
                self.cartesian_twist.rot[1] = self.cartesian_twist.vel[1]*0.01
                self.cartesian_twist.rot[2] = self.cartesian_twist.vel[2]*0.01

                self.ang_vel_x_prev = self.cartesian_twist.vel[0]
                self.ang_vel_y_prev = self.cartesian_twist.vel[1]
                self.ang_vel_z_prev = self.cartesian_twist.vel[2]

                print('reducing speed in collision zone')

                self.collision_zone_counter += 1
                print('self.collision_zone_counter', self.collision_zone_counter)
                if self.collision_zone_counter > 400:
                    self.collision_zone_counter = 0
                    self.leave_flag_collision_zone = True
                    self.current_collision_state = False
                    print('I have exited collision zone ###########################################',
                          self.collision_zone_counter)


            else:
                print('Normal zone')
            
            self.vel_fk_solver.JntToCart(self.q_in, self.eeFrame)
            R_B_tool = self.eeFrame.M ### R^B_t
            #print('R_B_tool',R_B_tool)
            cartesian_twist_transform = R_B_tool*self.cartesian_twist
            self.vel_ik_solver.CartToJnt(self.q_in, cartesian_twist_transform, self.qdot_out)

            for no_of_joints in range(kdl_chain.getNrOfJoints()):
                if (abs(self.qdot_out[no_of_joints]) - self.qdot_limit[no_of_joints]) > 0.050000:
                    for i in range(kdl_chain.getNrOfJoints()):
                        self.qdot_out[i] = 0.0
                        print('Torque error')
                    return
                if ((self.kuka_joint_states.position[no_of_joints] + self.qdot_out[no_of_joints]) >
                self.q_upper_limit[no_of_joints]) or ((self.kuka_joint_states.position[no_of_joints] + self.qdot_out[no_of_joints])
                 < self.q_lower_limit[no_of_joints]):
                    self.qdot_out[no_of_joints] = 0

                # self.kuka_joint_states.position[no_of_joints] = self.kuka_joint_states.position[no_of_joints] + self.qdot_out[no_of_joints]

                # self.q_in[no_of_joints] = self.kuka_joint_states.position[no_of_joints]

            for no_of_joints in range(kdl_chain.getNrOfJoints()):
                self.kuka_joint_states.position[no_of_joints] = self.kuka_joint_states.position[no_of_joints] + \
                    self.qdot_out[no_of_joints]
                # self.q_in[no_of_joints] = self.kuka_joint_states.position[no_of_joints]

            self.kuka_joint_states.header = geo_kuka_twist.header
            kuka_joints.positions = self.kuka_joint_states.position

            kuka_joints.accelerations = []
            kuka_joints.effort = []
            kuka_joints.time_from_start = rospy.Duration(0, 400000000)

            # kuka_joints.time_from_start.nsecs = 778523489 # Taken from rqt_joint_trajectory_controller
            msg.points.append(kuka_joints)

            # self.vel_fk_solver.JntToCart(self.kuka_joint_states.position, self.eeFrame)

            # Converting KDL frame to tf frame here
            # br1 = tf_c.toTf(self.eeFrame)
            # tf.TransformBroadcaster([0,0,0],[0,0,0,1],)
            # print('br1 frame is',br1)

            # self.kuka_joint_states_dummy_publisher.publish(msg)
            self.kuka_joint_states_publisher.publish(msg)


        else:

            self.flag_linear = False
            self.flag_angular = False



        
        rate = rospy.Rate(self.pub_rate) # Hz
        rate.sleep()
if __name__ == '__main__':
    print("Fanuc arm control start up v1 File\n")
    controller = Geomagic2FANUC()
    # controller.start()
    rospy.spin()