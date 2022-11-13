#!/usr/bin/env python
import rospy, tf, random, actionlib, tf_conversions
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import *
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from threading import Lock
import numpy as np
import math
import re
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt

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

class Var:
	INT_MAX = 1e9
	mutex = Lock()
	qr_mutex = Lock()
	global_mutex = Lock()
	tolerance = 1
	explore_phase = True
	planning_radius = 20
	cost_threshold = 55
	heatmap = None
	hidden_2_world = None
	y_max = 0
	x_max = 0
	y_min = 0
	x_min = 0
	jobs = []
	tick = 0
	update_frequency = 4
	secret_code = "-----"
	Qr_flag = False
	QRMsgs = [None for _ in range(5)]

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

class QrMessage:
    def __init__(self, name):
        self.x_map = 0
        self.y_map = 0
        self.z_map = 0
        self.x_orientation_map = 0
        self.y_orientation_map = 0
        self.z_orientation_map = 0
        self.w_orientation_map = 0
        self.x = 0
        self.y = 0
        self.x_target = 0
        self.y_target = 0
        self.N = 0
        self.L = ""
        self.observation_complete = False
        self.name = name

    def print_qr(self):
        print("\n-------------------------")
        print("x_map = " + str(self.x_map))
        print("y_map = " + str(self.y_map))
        print("z_map = " + str(self.z_map))
        print("x_orientation_map = " + str(self.x_orientation_map))
        print("y_orientation_map = " + str(self.y_orientation_map))
        print("z_orientation_map = " + str(self.z_orientation_map))
        print("w_orientation_map = " + str(self.w_orientation_map))
        print("x = " + str(self.x))
        print("y = " + str(self.y))
        print("x_target = " + str(self.x_target))
        print("y_target = " + str(self.y_target))
        print("N = " + str(self.N))
        print("L = " + str(self.L))

def change_code(pos, L):
	s = list(Var.secret_code)
	s[pos-1] = L
	Var.secret_code = "".join(s)

def qr_message_callback(message): 
    if (len(message.data) > 1):
		Var.qr_mutex.acquire()
		qr_message = re.split("\r\n", message.data)
		N = int("".join(list(qr_message[4])[2:]))
		if Var.QRMsgs[N-1] is None:
			QR = QrMessage("QR")
			QR.x = float("".join(list(qr_message[0])[2:]))
			QR.y = float("".join(list(qr_message[1])[2:]))
			QR.x_target = float("".join(list(qr_message[2])[7:]))
			QR.y_target = float("".join(list(qr_message[3])[7:]))
			QR.N = N
			QR.L = "".join(list(qr_message[5])[2:])
			QR.timestamp = rospy.Time.now().secs
			Var.QRMsgs[QR.N-1] = QR
			change_code(QR.N, QR.L)
			Var.jobs.append(QR.N-1)

		Var.qr_mutex.release()

def qr_pos_callback(message): 
	if(len(Var.jobs) > 0):
		Var.qr_mutex.acquire()
		while len(Var.jobs):
			QR = Var.QRMsgs[Var.jobs[0]]
			print("timing : ", message.header.stamp.secs, QR.timestamp)
			if abs(message.header.stamp.secs - QR.timestamp) > Var.tolerance:
				Var.QRMsgs[Var.jobs[0]] = None
			else:
				QR.x_map = message.pose.position.x 
				QR.y_map = message.pose.position.y 
				QR.z_map = message.pose.position.z 
				QR.x_orientation_map = message.pose.orientation.x 
				QR.y_orientation_map = message.pose.orientation.y 
				QR.z_orientation_map = message.pose.orientation.z
				QR.w_orientation_map = message.pose.orientation.w
				QR.observation_complete = True
			Var.jobs.pop(0)

		Var.qr_mutex.release()


def has_hidden_loc():
	qr_msg = [msg for msg in Var.QRMsgs if msg is not None and msg.observation_complete]
	if len(qr_msg) <= 1:
		return False

	Var.qr_mutex.acquire()
	X_1, Y_1 = qr_msg[0].x_map, qr_msg[0].y_map
	X_2, Y_2 = qr_msg[1].x_map, qr_msg[1].y_map
	x_1, y_1 = qr_msg[0].x, qr_msg[0].y
	x_2, y_2 = qr_msg[1].x, qr_msg[1].y
	A = np.array([[x_1, -y_1, 1, 0],\
		          [y_1,  x_1, 0, 1],\
		          [x_2, -y_2, 1, 0],\
		          [y_2,  x_2, 0, 1]])
	b = np.array([X_1, Y_1, X_2, Y_2])
	cos_t, sin_t, t_x, t_y = np.linalg.inv(A).dot(b)
	Var.hidden_2_world = np.array([[cos_t,-sin_t, 0, t_x],\
		      		  		 	   [sin_t, cos_t, 0, t_y],\
		              		 	   [    0,     0, 1,   0],\
		              	     	   [    0,     0, 0,   1]])
	Var.qr_mutex.release()

	return True

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
	# print(goal_pose)
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
	Var.y_max = int(( 4.0  -  GlobalFrame.pose.position.y) / GlobalFrame.resolution)
	Var.y_min = int((-4.1  -  GlobalFrame.pose.position.y) / GlobalFrame.resolution)
	Var.x_max = int(( 7.2  -  GlobalFrame.pose.position.x) / GlobalFrame.resolution)
	Var.x_min = int((-7.7  -  GlobalFrame.pose.position.x) / GlobalFrame.resolution)

def update_global_map_callback(data):
	if Var.tick == 0 and GlobalFrame.pose is not None:
		Var.global_mutex.acquire()
		x, y = data.x, data.y
		width, height = data.width, data.height
		GlobalFrame.occupancy_grid[y:y + height, x:x + width] = np.array(data.data).reshape(height, width)
		Var.global_mutex.release()

	Var.tick = (Var.tick + 1) % Var.update_frequency

def local_map_callback(data):
	Var.mutex.acquire()
	LocalFrame.width = data.info.width
	LocalFrame.height = data.info.height
	LocalFrame.pose = data.info.origin 
	LocalFrame.occupancy_grid = np.array([data.data]).reshape(LocalFrame.height, LocalFrame.width)
	Var.mutex.release()

def execute_360_turn(client, listener):
	i = 0
	for angle in [0, np.pi, 2*np.pi]:
		(trans,rot) = listener.lookupTransform('/map', '/base_footprint', rospy.Time())
		orientation = tf_conversions.transformations.quaternion_from_euler(0,0,angle)
		goal_pose = MoveBaseGoal()
		goal_pose.target_pose.header.frame_id = 'map'
		goal_pose.target_pose.pose.position.x = trans[0]
		goal_pose.target_pose.pose.position.y = trans[1]
		goal_pose.target_pose.pose.position.z = 0
		goal_pose.target_pose.pose.orientation.x = orientation[0]
		goal_pose.target_pose.pose.orientation.y = orientation[1]
		goal_pose.target_pose.pose.orientation.z = orientation[2]
		goal_pose.target_pose.pose.orientation.w = orientation[3]
		client.send_goal(goal_pose)
		ret = client.wait_for_result(timeout=rospy.Duration(10))
		print("completed turn %d"%i)
		i += 1

def is_next_visited(i, epsilon=0.1):
	Var.qr_mutex.acquire()
	point = np.array([Var.QRMsgs[i].x_target, Var.QRMsgs[i].y_target])
	for j in range(5):
		if Var.QRMsgs[j] is not None:
			next_point = np.array([Var.QRMsgs[j].x, Var.QRMsgs[j].y])
			if np.linalg.norm(point - next_point) < epsilon:
				Var.qr_mutex.release()
				return True
	Var.qr_mutex.release()
	return False

def rand(x, y):
	return np.random.randint(x, y+1)

def near_random_sample(point, inner_radius, outer_radius, prioritize_next=False, max_try=100000):
	Var.global_mutex.acquire()
	oy_min, oy_max = max(point[1] - outer_radius, Var.y_min), min(point[1] + outer_radius, Var.y_max)
	ox_min, ox_max = max(point[0] - outer_radius, Var.x_min), min(point[0] + outer_radius, Var.x_max)
	iy_min, iy_max = max(point[1] - inner_radius, Var.y_min), min(point[1] + inner_radius, Var.y_max)
	ix_min, ix_max = max(point[0] - inner_radius, Var.x_min), min(point[0] + inner_radius, Var.x_max)

	for i in range(max_try):
		coin = np.random.randint(4)
		if coin == 0:
			x, y = rand(ox_min, ox_max), rand(oy_min, iy_min)
		elif coin == 1:
			x, y = rand(ox_min, ix_min), rand(oy_min, oy_max)
		elif coin == 2:
			x, y = rand(ox_min, ox_max), rand(iy_max, oy_max)
		else:
			x, y = rand(ix_max, ox_max), rand(oy_min, oy_max)

		if GlobalFrame.occupancy_grid[y, x] <= Var.cost_threshold:
			Var.global_mutex.release()
			return x, y

	Var.global_mutex.release()

	return -1, -1

def execute_round_search(client, inner_radius, outer_radius, timeout):
	for i in range(5):
		if Var.QRMsgs[i] is None:
			continue
		
		Var.qr_mutex.acquire()
		point = np.array([Var.QRMsgs[i].x_target, Var.QRMsgs[i].y_target, 0, 0])
		point = ((Var.hidden_2_world.dot(point)[:2] - np.array([GlobalFrame.pose.position.x, GlobalFrame.pose.position.y])) / GlobalFrame.resolution).astype(np.int32)
		Var.qr_mutex.release()

		while not is_next_visited(i):
			x, y = near_random_sample(point, inner_radius, outer_radius)
			if x == -1:
				raise Exception("Probably No suitable point to observe= %f %f" % (Var.QRMsgs[i].x_target, Var.QRMsgs[i].y_target))

			theta = np.arctan2(point[1] - y, point[0] - x)
			orientation = tf_conversions.transformations.quaternion_from_euler(0,0,theta)
			
			goal_pose = MoveBaseGoal()
			goal_pose.target_pose.header.frame_id = 'map'
			goal_pose.target_pose.pose.position.x = (float(x) * GlobalFrame.resolution) + GlobalFrame.pose.position.x
			goal_pose.target_pose.pose.position.y = (float(y) * GlobalFrame.resolution) + GlobalFrame.pose.position.y
			goal_pose.target_pose.pose.position.z = 0
			goal_pose.target_pose.pose.orientation.x = orientation[0]
			goal_pose.target_pose.pose.orientation.y = orientation[1]
			goal_pose.target_pose.pose.orientation.z = orientation[2]
			goal_pose.target_pose.pose.orientation.w = orientation[3]
			client.send_goal(goal_pose)
			ret = client.wait_for_result(timeout=rospy.Duration(timeout))
			print("success..." if ret else "timeout...")

	return None in Var.QRMsgs

override = True

def main():
	rospy.init_node('robot_navigation')
	sub1 = rospy.Subscriber('/move_base/global_costmap/costmap', OccupancyGrid, global_map_callback, queue_size=100)
	rospy.Subscriber('/move_base/global_costmap/costmap_updates', OccupancyGridUpdate, update_global_map_callback, queue_size=100)
	sub2 = rospy.Subscriber('/move_base/local_costmap/costmap', OccupancyGrid, local_map_callback, queue_size=100)
	rospy.Subscriber('/visp_auto_tracker/code_message', String, qr_message_callback, queue_size=100)
	rospy.Subscriber('/visp_auto_tracker/object_position', PoseStamped, qr_pos_callback, queue_size=100)

	client = actionlib.SimpleActionClient('move_base', MoveBaseAction) 
	client.wait_for_server()
	r = rospy.Rate(2)
	size = 15
	sigma = 5
	kernel = 1e3 * np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
	listener = tf.TransformListener()

	while not rospy.is_shutdown():
		if Var.explore_phase:
			goal = get_next_randomized_goal()
			if goal is not None and not override:
				client.send_goal(goal)
				ret = client.wait_for_result(timeout=rospy.Duration(10))
				print("success" if ret else "timeout")
				(trans, rots) = listener.lookupTransform('/map','/base_footprint', rospy.Time())
				increment_heatmap(trans[0], trans[1], kernel)
				execute_360_turn(client, listener)
			if has_hidden_loc():
				Var.explore_phase = False
				sub1.unregister()
				sub2.unregister()
				print("end of explore phase : ", Var.hidden_2_world)
		else:
			if not execute_round_search(client, 10, 20, 30):
				break

		r.sleep()

	print(Var.secret_code)


if __name__ == '__main__':
	main()



# wait in a active while loop through out checking if we receive a qr code.
# when qr code is detected exit random phase and start 
# to go to the next qr code and try all angles repeatedly.
	# planning motion 

# listener.wait_for_transform('/base_footprint', '/map', time=rospy.Time(), timeout=rospy.Duration(1))
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

# y_max = 3.076
# x_max = 7.2
# y_min = -4.066
# y_max_2 = 4.144
# x_min = -7.699













