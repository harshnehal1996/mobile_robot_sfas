#!/usr/bin/env python
import rospy, tf, random, actionlib, tf_conversions
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import *
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from threading import Lock, Thread
import numpy as np
import math
import re
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
import time

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
	job_mutex = Lock()
	tolerance = 1
	sleep_time = 0
	explore_phase = True
	adjust = True
	planning_radius = 20
	cost_threshold = 55
	minimum_error = 0.02
	heatmap = None
	hidden_2_world = None
	listener = None
	min_observation = 2
	y_max = 0
	x_max = 0
	y_min = 0
	x_min = 0
	br = None
	job = -1
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
        self.trans = 0
        self.orient = 0
        self.adjust = False
        self.num_observation = 0
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

def get_feasable_view_pos(position, vector, R=20, num_steps=30):
	offset = np.array([GlobalFrame.pose.position.x, GlobalFrame.pose.position.y])
	position =((position[:2] - offset) / GlobalFrame.resolution).astype(np.int32)
	vector = (vector[:2] / np.linalg.norm(vector[:2]))
	Var.global_mutex.acquire()

	for step in range(num_steps):
		new_position = position + R * vector
		if GlobalFrame.occupancy_grid[int(new_position[1]), int(new_position[0])] <= Var.cost_threshold:
			new_position = new_position * GlobalFrame.resolution + offset
			Var.global_mutex.release()
			return new_position, np.arctan2(-vector[1], -vector[0])
		R += 1

	Var.global_mutex.release()
	return None, None

def is_zero(quat_1, quat_2):
	return quat_1[0] == 0 and quat_1[1] == 0 and quat_1[2] == 0 and quat_2[0] == 0 and quat_2[1] == 0 and quat_2[2] == 0

# since no timestamp in msg is provided, assume that current time is msg timestamp
def qr_message_callback(message, client):
	if Var.listener is None:
		return

	if (len(message.data) > 1):
		Var.qr_mutex.acquire()
		qr_message = re.split("\r\n", message.data)
		
		try:
			N = int("".join(list(qr_message[4])[2:]))
		except:
			print("exception : ", message)
			Var.qr_mutex.release()
			return

		if Var.QRMsgs[N-1] is None and (Var.job == -1 or not Var.explore_phase):
			QR = QrMessage("QR")
			QR.x = float("".join(list(qr_message[0])[2:]))
			QR.y = float("".join(list(qr_message[1])[2:]))
			QR.x_target = float("".join(list(qr_message[2])[7:]))
			QR.y_target = float("".join(list(qr_message[3])[7:]))
			QR.N = N
			QR.L = "".join(list(qr_message[5])[2:])
			QR.num_observation = 1 if Var.explore_phase else Var.min_observation + 1
			QR.observation_complete = not Var.explore_phase
			Var.QRMsgs[QR.N-1] = QR
			change_code(QR.N, QR.L)			
			
			if Var.explore_phase:
				print("in-msg callback observing....")
				(trans, orient) = Var.listener.lookupTransform('/map', '/base_footprint', rospy.Time())
				goal_pose = MoveBaseGoal()
				goal_pose.target_pose.header.frame_id = 'map'
				goal_pose.target_pose.pose.position.x = trans[0]
				goal_pose.target_pose.pose.position.y = trans[1]
				goal_pose.target_pose.pose.position.z = 0
				goal_pose.target_pose.pose.orientation.x = orient[0]
				goal_pose.target_pose.pose.orientation.y = orient[1]
				goal_pose.target_pose.pose.orientation.z = orient[2]
				goal_pose.target_pose.pose.orientation.w = orient[3]
				Var.qr_mutex.release()
				ret, trans, orient = execute_goal(client, goal_pose, 10, block_threads=True)
				QR.timestamp = rospy.Time.now()
				Var.qr_mutex.acquire()
				
				if ret:
					QR.trans = trans
					QR.orient = orient
					Var.job = N-1
				else:
					Var.QRMsgs[N-1] = None

		Var.qr_mutex.release()

def qr_pos_callback(message, client): 
	if(Var.job != -1):
		Var.qr_mutex.acquire()
		QR = Var.QRMsgs[Var.job]
		print("timing : ", (message.header.stamp.secs, message.header.stamp.nsecs), (QR.timestamp.secs, QR.timestamp.nsecs))
		
		b1 = message.header.stamp.secs == QR.timestamp.secs
		b2 = message.header.stamp.secs > QR.timestamp.secs
		b3 = message.header.stamp.nsecs < QR.timestamp.nsecs
		b4 = message.header.stamp.secs - QR.timestamp.secs <= Var.tolerance

		if not(b1 or b2) or (b1 and b3):
			Var.qr_mutex.release()
			return
		elif (b1 and not b3) or (b2 and b4):				
			orient = QR.orient
		 	trans = QR.trans
			quat = np.array([message.pose.position.x, message.pose.position.y, message.pose.position.z, 0])
			code_quad = np.array([message.pose.orientation.x, message.pose.orientation.y, message.pose.orientation.z,  message.pose.orientation.w])
			
			if is_zero(quat, code_quad):
				Var.qr_mutex.release()
				return

			print('position vector of qr  : ', quat[:3])	
			iorient = tf.transformations.quaternion_inverse(orient)
			res = tf.transformations.quaternion_multiply(orient, quat)
			res = tf.transformations.quaternion_multiply(res, iorient)
			position = res[:3] + trans
			QR.x_map, QR.y_map, QR.z_map = position
			
			if not QR.adjust:
				print("adusting for: ", QR.N)
				lp = tf.transformations.quaternion_multiply(orient, code_quad)
				ilp = tf.transformations.quaternion_inverse(lp)
				quat = np.array([0, 0, 1, 0])
				res = tf.transformations.quaternion_multiply(lp, quat)
				res = tf.transformations.quaternion_multiply(res, ilp)
				position, angle = get_feasable_view_pos(position, res[:3])
				print(position)
				print(angle)
				
				if position is not None:
					goal_pose = MoveBaseGoal()
					orientation = tf.transformations.quaternion_from_euler(0, 0, angle)
					goal_pose.target_pose.header.frame_id = 'map'
					goal_pose.target_pose.pose.position.x = position[0]
					goal_pose.target_pose.pose.position.y = position[1]
					goal_pose.target_pose.pose.position.z = 0
					goal_pose.target_pose.pose.orientation.x = orientation[0]
					goal_pose.target_pose.pose.orientation.y = orientation[1]
					goal_pose.target_pose.pose.orientation.z = orientation[2]
					goal_pose.target_pose.pose.orientation.w = orientation[3]
					Var.qr_mutex.release()
					print('readjusting')
					ret, trans, orient = execute_goal(client, goal_pose, 30, block_threads=True)

					if ret:
						print('adjusted')
						QR.timestamp = rospy.Time.now()
						QR.adjust = True
						QR.trans = trans
						QR.orient = orient
						return
					Var.qr_mutex.acquire()

			QR.observation_complete = True

			i = 0
			while i < 10:
				br = tf.TransformBroadcaster()
				br.sendTransform((QR.x_map, QR.y_map, QR.z_map),
			                 	tf.transformations.quaternion_from_euler(0, 0, 0), # roll, pitch, yaw
			                 	rospy.Time.now(),
			                 	"code_" + str(QR.N),
			                 	"map")
				time.sleep(1)
				i += 1
			print("send transform")
		
		else:
			print("measurement dropped!")
			Var.QRMsgs[Var.job] = None

		Var.job = -1
		Var.qr_mutex.release()

def has_hidden_loc_3D():
	qr_msg = [msg for msg in Var.QRMsgs if msg is not None and msg.observation_complete]
	if len(qr_msg) <= 2:
		return False

	vec1 = np.array([qr_msg[0].x_map, qr_msg[0].y_map, qr_msg[0].z_map])
	vec2 = np.array([qr_msg[1].x_map, qr_msg[1].y_map, qr_msg[1].z_map])
	vec3 = np.array([qr_msg[2].x_map, qr_msg[2].y_map, qr_msg[2].z_map])

	p1m = vec2 - vec1
	p2m = vec3 - vec1
	p3m = np.cross(p1m, p2m)
	Pm = np.stack([p1m, p2m, p3m], axis=1)
	iPm = np.linalg.inv(Pm)

	mod_p1m = np.linalg.norm(p1m)
	mod_p2m = np.linalg.norm(p2m)
	vec1 = np.array([qr_msg[0].x, qr_msg[0].y, 0])
	vec2 = np.array([qr_msg[1].x, qr_msg[1].y, 0])
	vec3 = np.array([qr_msg[2].x, qr_msg[2].y, 0])

	p1h = vec2 - vec1
	p2h = vec3 - vec1

	mod_z1h = np.sqrt((mod_p1m ** 2) - (np.linalg.norm(p1h) ** 2))
	mod_z2h = np.sqrt((mod_p2m ** 2) - (np.linalg.norm(p2h) ** 2))
	R = np.zeros((3,3))

	for sign1 in [-1,1]:
		for sign2 in [-1,1]:
			p1h[2] = sign1 * mod_z1h
			p2h[2] = sign2 * mod_z2h
			p3h = np.cross(p1h, p2h)
			Ph = np.stack([p1h, p2h, p3h], axis=1)
			R = Ph.dot(iPm)

	return has_hidden_loc()

def has_hidden_loc():
	qr_msg = [msg for msg in Var.QRMsgs if msg is not None and msg.observation_complete]
	if len(qr_msg) <= 1:
		return False

	Var.qr_mutex.acquire()
	min_error = Var.INT_MAX
	pair = -1,-1
	
	for i in range(len(qr_msg)):
		for j in range(i+1, len(qr_msg)):
			vec1 = np.array([qr_msg[i].x_map, qr_msg[i].y_map])
			vec2 = np.array([qr_msg[j].x_map, qr_msg[j].y_map])
			v1 = np.linalg.norm(vec1 - vec2)
			vec1 = np.array([qr_msg[i].x, qr_msg[i].y])
			vec2 = np.array([qr_msg[j].x, qr_msg[j].y])
			v2 = np.linalg.norm(vec1 - vec2)
			print("(X-Y, x-y) : ", (v1, v2))
			error = 1 - (min(v1, v2) / max(v1, v2))
			if error < min_error:
				min_error = error
				pair = (i,j)

	if min_error > Var.minimum_error:
		# Var.qr_mutex.release()
		print("error too high : ", min_error)
		# return False

	qr_msg = [qr_msg[pair[0]], qr_msg[pair[1]]]
	X_1, Y_1 = qr_msg[0].x_map, qr_msg[0].y_map
	X_2, Y_2 = qr_msg[1].x_map, qr_msg[1].y_map
	x_1, y_1 = qr_msg[0].x, qr_msg[0].y
	x_2, y_2 = qr_msg[1].x, qr_msg[1].y

	A = np.array([[x_1, -y_1, 1, 0],\
		          [y_1,  x_1, 0, 1],\
		          [x_2, -y_2, 1, 0],\
		          [y_2,  x_2, 0, 1]])
	b = np.array([X_1, Y_1, X_2, Y_2])
	t1, t2, t_x, t_y = np.linalg.inv(A).dot(b)
	
	cos_t = t1 / np.sqrt(t1**2 + t2**2)
	sin_t = t2 / np.sqrt(t1**2 + t2**2)

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

def execute_goal(client, goal, timeout, lock_time=2, block_threads=False):
	Var.job_mutex.acquire()
	if block_threads:
		client.send_goal(goal)
		ret = client.wait_for_result(timeout=rospy.Duration(timeout))
		time.sleep(lock_time)
		Var.sleep_time = 2
		(trans, orient) = Var.listener.lookupTransform('/map', '/camera_optical_link', rospy.Time())
		Var.job_mutex.release()
		return ret, trans, orient
	else:
		if Var.sleep_time > 0:
			time.sleep(Var.sleep_time)
		Var.sleep_time = 0
		client.send_goal(goal)
		Var.job_mutex.release()
		ret = client.wait_for_result(timeout=rospy.Duration(timeout))

	return ret

def execute_360_turn(client):
	i = 0
	for angle in [0, np.pi, 2*np.pi]:
		(trans,rot) = Var.listener.lookupTransform('/map', '/base_footprint', rospy.Time())
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
		execute_goal(client, goal_pose, 10)
		# print("completed turn %d"%i)
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

location_ = None

def send_transform():
	global location_
	br = tf.TransformBroadcaster()
	while not Var.stop:
		br.sendTransform((location_[0], location_[1], 0),
				          tf.transformations.quaternion_from_euler(0, 0, 0), # roll, pitch, yaw
				          rospy.Time.now(),
				          "target",
				          "map")
		time.sleep(1)

def execute_round_search(client, inner_radius, outer_radius, timeout, br):
	for i in range(5):
		if Var.QRMsgs[i] is None:
			continue
		
		Var.qr_mutex.acquire()
		point = np.array([Var.QRMsgs[i].x_target, Var.QRMsgs[i].y_target, 0, 0])
		location = Var.hidden_2_world.dot(point)[:2]
		point = ((location - np.array([GlobalFrame.pose.position.x, GlobalFrame.pose.position.y])) / GlobalFrame.resolution).astype(np.int32)
		Var.qr_mutex.release()
		Var.stop = False
		global location_
		location_ = location.copy()
		thread = Thread(target=send_transform)
		thread.start()

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
			ret = execute_goal(client, goal_pose, timeout)
			print("success..." if ret else "timeout...")

		Var.stop = True
		thread.join()

	return None in Var.QRMsgs

override = False

def main():
	rospy.init_node('robot_navigation')
	client = actionlib.SimpleActionClient('move_base', MoveBaseAction) 
	client.wait_for_server()

	sub1 = rospy.Subscriber('/move_base/global_costmap/costmap', OccupancyGrid, global_map_callback, queue_size=100)
	rospy.Subscriber('/move_base/global_costmap/costmap_updates', OccupancyGridUpdate, update_global_map_callback, queue_size=100)
	sub2 = rospy.Subscriber('/move_base/local_costmap/costmap', OccupancyGrid, local_map_callback, queue_size=100)
	rospy.Subscriber('/visp_auto_tracker/code_message', String, qr_message_callback, callback_args=client, queue_size=100)
	sub3 = rospy.Subscriber('/visp_auto_tracker/object_position', PoseStamped, qr_pos_callback, callback_args=client, queue_size=100)

	r = rospy.Rate(2)
	size = 15
	sigma = 5
	kernel = 1e3 * np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
	listener = tf.TransformListener()
	Var.listener = listener
	br = tf.TransformBroadcaster()
	Var.br = br

	while not rospy.is_shutdown():
		if Var.explore_phase:
			goal = get_next_randomized_goal()
			
			if goal is not None and not override:
				execute_goal(client, goal, 10)
				(trans, rots) = listener.lookupTransform('/map','/base_footprint', rospy.Time())
				increment_heatmap(trans[0], trans[1], kernel)
				if has_hidden_loc():
					Var.explore_phase = False
					sub1.unregister()
					sub2.unregister()
					sub3.unregister()
					print("end of explore phase : ", Var.hidden_2_world)
					continue
				execute_360_turn(client)
			
			if has_hidden_loc():
				Var.explore_phase = False
				sub1.unregister()
				sub2.unregister()
				sub3.unregister()
				print("end of explore phase : ", Var.hidden_2_world)
		else:
			if not execute_round_search(client, 15, 36, 30, br):
				break

		r.sleep()

	print(Var.secret_code)


if __name__ == '__main__':
	main()



# # since no timestamp in msg is provided, assume that current time is msg timestamp
# def qr_message_callback(message, client):
# 	if Var.listener is None:
# 		return

# 	if (len(message.data) > 1):
# 		Var.qr_mutex.acquire()
# 		qr_message = re.split("\r\n", message.data)
		
# 		try:
# 			N = int("".join(list(qr_message[4])[2:]))
# 		except:
# 			print("exception : ", message)
# 			Var.qr_mutex.release()
# 			return

# 		if Var.QRMsgs[N-1] is None:
# 			QR = QrMessage("QR")
# 			QR.x = float("".join(list(qr_message[0])[2:]))
# 			QR.y = float("".join(list(qr_message[1])[2:]))
# 			QR.x_target = float("".join(list(qr_message[2])[7:]))
# 			QR.y_target = float("".join(list(qr_message[3])[7:]))
# 			QR.N = N
# 			QR.L = "".join(list(qr_message[5])[2:])
# 			QR.num_observation = 1 if Var.explore_phase else Var.min_observation + 1
# 			QR.observation_complete = not Var.explore_phase
# 			QR.timestamp = rospy.Time.now().secs
# 			Var.QRMsgs[QR.N-1] = QR
# 			change_code(QR.N, QR.L)
# 		elif Var.QRMsgs[N-1].num_observation < Var.min_observation:
# 			QR = Var.QRMsgs[N-1]
# 			current_time = rospy.Time.now().secs
# 			if (current_time - QR.timestamp) > 1:
# 				Var.QRMsgs[N-1] = None
# 			else:
# 				QR.num_observation += 1
# 				print('num_observation : ', QR.num_observation)
				
# 				QR.timestamp = current_time
# 				if QR.num_observation == Var.min_observation:
# 					(trans, orient) = Var.listener.lookupTransform('/map', '/base_footprint', rospy.Time())
# 					goal_pose = MoveBaseGoal()
# 					goal_pose.target_pose.header.frame_id = 'map'
# 					goal_pose.target_pose.pose.position.x = trans[0]
# 					goal_pose.target_pose.pose.position.y = trans[1]
# 					goal_pose.target_pose.pose.position.z = 0
# 					goal_pose.target_pose.pose.orientation.x = orient[0]
# 					goal_pose.target_pose.pose.orientation.y = orient[1]
# 					goal_pose.target_pose.pose.orientation.z = orient[2]
# 					goal_pose.target_pose.pose.orientation.w = orient[3]
# 					Var.qr_mutex.release()
# 					ret, trans, orient = execute_goal(client, goal_pose, 10, block_threads=True)
# 					QR.timestamp = rospy.Time.now().secs
# 					Var.qr_mutex.acquire()
					
# 					if ret:
# 						QR.trans = trans
# 						QR.orient = orient
# 						Var.jobs.append(N-1)
# 					else:
# 						Var.QRMsgs[N-1] = None

# 		Var.qr_mutex.release()

# def qr_pos_callback(message): 
# 	if(len(Var.jobs) > 0):
# 		Var.qr_mutex.acquire()
# 		while len(Var.jobs):
# 			QR = Var.QRMsgs[Var.jobs[0]]
# 			print("timing : ", message.header.stamp.secs, QR.timestamp)

# 			if message.header.stamp.secs < QR.timestamp:
# 				Var.qr_mutex.release()
# 				return
# 			elif message.header.stamp.secs - QR.timestamp <= Var.tolerance:				
# 				orient = QR.orient
# 			 	trans = QR.trans
# 				quat = np.array([message.pose.position.x, message.pose.position.y, message.pose.position.z, 0])				
# 				print('distance  : ', quat)

# 				iorient = tf.transformations.quaternion_inverse(orient)
# 				res = tf.transformations.quaternion_multiply(orient, quat)
# 				res = tf.transformations.quaternion_multiply(res, iorient)
# 				QR.x_map, QR.y_map, QR.z_map = res[:3] + trans
# 				QR.observation_complete = True

# 				i = 0
# 				while i < 10:
# 					br = tf.TransformBroadcaster()
# 					br.sendTransform((QR.x_map, QR.y_map, QR.z_map),
# 				                 	tf.transformations.quaternion_from_euler(0, 0, 0), # roll, pitch, yaw
# 				                 	rospy.Time.now(),
# 				                 	"code_" + str(QR.N),
# 				                 	"map")
# 					time.sleep(1)
# 					i += 1
				
# 				# print("send transform")
# 			else:
# 				Var.QRMsgs[Var.jobs[0]] = None

# 			Var.jobs.pop(0)

# 		Var.qr_mutex.release()





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
# 0.8 minimum view













