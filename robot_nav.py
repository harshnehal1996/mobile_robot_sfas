#!/usr/bin/env python
import rospy, tf, random, actionlib, tf_conversions
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import *
from nav_msgs.msg import OccupancyGrid
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from threading import Lock
import numpy as np
import math
import matplotlib.pyplot as plt

class Var:
	INT_MAX = 1e9
	mutex = Lock()
	explore_phase = True
	planning_radius = 20
	cost_threshold = 60
	heatmap = None

class GlobalFrame:
	width = -1
	height = -1
	resolution = 1
	pose = None
	occupancy_grid = None

class LocalFrame:
	width = -1
	height = -1
	pose = None
	occupancy_grid = None

# /move_base/global_costmap/inflation_layer/set_parameters dynamic_reconfigure/Reconfigure config
def call_service(service, className, *args):
    rospy.wait_for_service(service)
    try:
        proxy = rospy.ServiceProxy(service, eval(className))
        resp1 = proxy(*args)
        return resp1
    except rospy.ServiceException, e:
        print("Service call failed: %s"%e)
        return None

def get_next_randomized_goal():
	if LocalFrame.pose is None:
		return None

	Var.mutex.acquire()
	x_pos, y_pos = (np.array([LocalFrame.pose.position.x - GlobalFrame.pose.position.x, LocalFrame.pose.position.y - GlobalFrame.pose.position.y]) / GlobalFrame.resolution).astype(np.int32)
	y_pos, x_pos = y_pos + (LocalFrame.height // 2) - Var.planning_radius, x_pos + (LocalFrame.width // 2) - Var.planning_radius 
	local_heatmap = np.copy(Var.heatmap[y_pos:y_pos+2*Var.planning_radius+1, x_pos:x_pos+2*Var.planning_radius+1])
	local_heatmap[LocalFrame.occupancy_grid[(LocalFrame.height//2)-Var.planning_radius:(LocalFrame.height//2)+Var.planning_radius+1, (LocalFrame.width//2)-Var.planning_radius:(LocalFrame.width//2)+Var.planning_radius+1] >= Var.cost_threshold] = Var.INT_MAX
	width = 2 * Var.planning_radius + 1
	height = width
	assert len(local_heatmap) == height and len(local_heatmap[0]) == width
	index = np.random.choice(np.flatnonzero(local_heatmap == local_heatmap.min()))
	iy = index / width
	ix = index % width
	iy += y_pos
	ix += x_pos
	orientation = tf_conversions.transformations.quaternion_from_euler(0, 0, np.random.uniform(0, 2 * np.pi))
	goal_pose = MoveBaseGoal()
	goal_pose.target_pose.header.frame_id = 'map'
	goal_pose.target_pose.pose.position.x = (float(ix) * GlobalFrame.resolution) + GlobalFrame.pose.position.x
	goal_pose.target_pose.pose.position.y = (float(iy) * GlobalFrame.resolution) + GlobalFrame.pose.position.y
	goal_pose.target_pose.pose.position.z = 0
	goal_pose.target_pose.pose.orientation.x = orientation[0]
	goal_pose.target_pose.pose.orientation.y = orientation[1]
	goal_pose.target_pose.pose.orientation.z = orientation[2]
	goal_pose.target_pose.pose.orientation.w = orientation[3]
	Var.mutex.release()
	print(goal_pose)
	return goal_pose

def increment_heatmap(x, y, kernel):
	update_radius = len(kernel) // 2
	y = np.round((y - GlobalFrame.pose.position.y) /  GlobalFrame.resolution).astype(np.int32) 
	x = np.round((x - GlobalFrame.pose.position.x) /  GlobalFrame.resolution).astype(np.int32)
	Var.heatmap[y-update_radius : y+update_radius+1, x-update_radius : x+update_radius+1] += kernel
	# plt.imshow(Var.heatmap)
	# plt.show()

def global_map_callback(global_map):	
	GlobalFrame.width = global_map.info.width
	GlobalFrame.height = global_map.info.height
	GlobalFrame.resolution = global_map.info.resolution #units per pixel
	GlobalFrame.occupancy_grid = np.array([global_map.data]).reshape(GlobalFrame.height, GlobalFrame.width)
	GlobalFrame.pose = global_map.info.origin
	Var.heatmap = np.zeros((GlobalFrame.height, GlobalFrame.width))

def local_map_callback(data):
	Var.mutex.acquire()
	LocalFrame.width = data.info.width
	LocalFrame.height = data.info.height
	LocalFrame.pose = data.info.origin 
	LocalFrame.occupancy_grid = np.array([data.data]).reshape(LocalFrame.height, LocalFrame.width)
	Var.mutex.release()

rospy.init_node('robot_navigation')
rospy.Subscriber('/move_base/global_costmap/costmap', OccupancyGrid, global_map_callback, queue_size=1000)
rospy.Subscriber('/move_base/local_costmap/costmap', OccupancyGrid, local_map_callback, queue_size=1000)
client = actionlib.SimpleActionClient('move_base', MoveBaseAction) 
client.wait_for_server()
r = rospy.Rate(2)
size = 15
sigma = 5
kernel = 1e3 * np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))

while not rospy.is_shutdown():
	if Var.explore_phase:
		goal = get_next_randomized_goal()
		if goal is not None:
			client.send_goal(goal)
			ret = client.wait_for_result(timeout=rospy.Duration(10))
			print('ret=', ret)
			if ret:
				increment_heatmap(goal.target_pose.pose.position.x, goal.target_pose.pose.position.y, kernel)
	r.sleep()


# print([LocalFrame.pose.position.x - GlobalFrame.pose.position.x, LocalFrame.pose.position.y - GlobalFrame.pose.position.y])
# print(np.array([LocalFrame.pose.position.x - GlobalFrame.pose.position.x, LocalFrame.pose.position.y - GlobalFrame.pose.position.y]) / GlobalFrame.resolution)
# print((LocalFrame.occupancy_grid >= Var.cost_threshold).shape)
# print(local_heatmap.shape)
# print(Var.heatmap.shape)
# print(x_pos, y_pos)
# print(local_heatmap)
# print(LocalFrame.occupancy_grid[(LocalFrame.height//2)-Var.planning_radius:(LocalFrame.height//2)+Var.planning_radius+1, (LocalFrame.width//2)-Var.planning_radius:(LocalFrame.width//2)+Var.planning_radius+1][::-1])
# plt.imshow(LocalFrame.occupancy_grid[(LocalFrame.height//2)-Var.planning_radius:(LocalFrame.height//2)+Var.planning_radius+1, (LocalFrame.width//2)-Var.planning_radius:(LocalFrame.width//2)+Var.planning_radius+1][::-1])
# plt.show()
