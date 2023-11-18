import math
import gymnasium as gym
from gym import spaces
import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
import time

# Open AI gym code
class GazeboEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()

        #Initialising ROS
        rospy.init_node('custom_gym', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.marker_publisher = rospy.Publisher('goal_marker', Marker, queue_size=10)

        #Generating velocity pairs
        linear_velocities = np.linspace(0, 1, 5)
        angular_velocities = np.linspace(-1, 1, 5)
        self.velocity_pairs = [(linear, angular) for linear in linear_velocities for angular in angular_velocities]

        #Defining action and observation space
        self.action_space = spaces.Discrete(len(self.velocity_pairs)) 
        self.observation_space = spaces.Box(low = np.array([-1, -1, -1, -1]), high = np.array([1, 1, 1, 1]), dtype = np.float32)

        #Defining initial goal
        self.goal_x = 1
        self.goal_y = 1
        self.ideal_dtheta = None

        #Defining goal and lost thresholds
        self.goal_threshold = 0.8
        self.lost_threshold = 3

        #Defining initial orientaions
        self.start_orientation_z = 0.000384
        self.start_orientation_w = 0.999

        #Updating distance and time at every time step
        self.prev_goal_distance = np.sqrt((self.goal_x)**2 + (self.goal_y)**2)
        self.prev_distance_from_line = 0
        self.episode_start_time = 0

        #Initialising Goal marker
        self.marker = Marker()
        self.marker.header.frame_id = "odom"
        self.marker.type = self.marker.SPHERE
        self.marker.action = self.marker.ADD
        self.marker.scale.x = 0.2
        self.marker.scale.y = 0.2
        self.marker.scale.z = 0.2
        self.marker.color.a = 1.0
        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0
        self.marker.pose.orientation.w = 1.0
        self.marker.pose.position.x = self.goal_x
        self.marker.pose.position.y = self.goal_y
        self.marker.pose.position.z = 0

    def step(self, action):
        #Picking the action
        linear, angular = self.velocity_pairs[action - 1]

        #Publishing goal on rviz
        self.marker_publisher.publish(self.marker) 

        #Initialising variables for odom callback
        pos_x, pos_y, linear_velocity, angular_velocity = None, None, None, None
        orientation_z, orientation_w = None, None

        def odom_callback(data):
            nonlocal pos_x, pos_y, linear_velocity, angular_velocity, orientation_z, orientation_w

            linear_velocity = data.twist.twist.linear.x
            angular_velocity = data.twist.twist.angular.z

            pos_x = data.pose.pose.position.x
            pos_y = data.pose.pose.position.y

            orientation_z = data.pose.pose.orientation.z
            orientation_w = data.pose.pose.orientation.w

        odom_subscriber = rospy.Subscriber('/odom', Odometry, odom_callback)

        rate = rospy.Rate(10)
        while pos_x is None or pos_y is None:
            rate.sleep()

        #Making the bot move for 0.01 seconds
        start_time = rospy.get_time()
        while (rospy.Time.now() - rospy.Time.from_sec(start_time)).to_sec() < 0.01:
            velocity_command = Twist()
            velocity_command.linear.x = linear
            velocity_command.angular.z = angular

            self.velocity_publisher.publish(velocity_command)
            rate.sleep()

        #Initialising new state
        dtheta = self.angle_from_goal(orientation_z, orientation_w, pos_x, pos_y)
        new_state = [np.sin(dtheta), np.cos(dtheta),linear_velocity, angular_velocity]

        #Calculating rewards
        reward = self.gen_reward(pos_x, pos_y, orientation_z, orientation_w)     

        #Checking if the episode has ended
        distance_from_goal = self.euclidean_distance(pos_x, pos_y, self.goal_x, self.goal_y)
        
        done = False
        if distance_from_goal < self.goal_threshold:
            done = True
        
        if distance_from_goal > self.lost_threshold:
            done = True

        #Setting new goal if episode done
        if(done == True):
            self.gen_goal(pos_x, pos_y, 1.5, (-np.pi, np.pi))
            self.prev_goal_distance = np.sqrt((self.goal_x)**2 + (self.goal_y)**2)
          
        #Returning the new state, reward and done
        return new_state, reward, done, {}

    def reset(self):
        try:
            reset = rospy.ServiceProxy("/gazebo/reset_world", Empty)
            reset()
            dx = self.goal_x
            dy = self.goal_y

            distance_from_goal = np.sqrt((dx)**2 + (dy)**2)
            bot_steer = np.arctan2(self.start_orientation_z, self.start_orientation_w) * 2
            goal_steer = np.arctan2(dy, dx)

            dtheta = goal_steer - bot_steer
            ini_state = [np.sin(dtheta), np.cos(dtheta), 0, 0]
            return ini_state, None

        except rospy.ServiceException as e:
            print(f"{type(e)}: {e}")
    
    def render(self):
        pass

    def gen_goal(self, pos_x, pos_y, distance, theta_range):
        theta = np.random.uniform(0, 2*math.pi)
        x = 0 + distance * math.cos(theta)
        y = 0 + distance * math.sin(theta)

        self.marker.pose.position.x = x
        self.marker.pose.position.y = y

        self.goal_x = x
        self.goal_y = y

    def distance_from_line(self, goal_x, goal_y, robo_pos_x, robo_pos_y):
        numerator = abs((goal_y)*(robo_pos_x) - (goal_x)*(robo_pos_y))
        denom = math.sqrt((goal_y)**2  + (goal_x)**2)

        distance = numerator / denom
        return distance

    def euclidean_distance(self, robo_pos_x, robo_pos_y, goal_x, goal_y):
        dx = goal_x - robo_pos_x
        dy = goal_y - robo_pos_y

        distance_from_goal = np.sqrt((dx)**2 + (dy)**2)
        return distance_from_goal

    def angle_from_goal(self, orientation_z, orientation_w, pos_x, pos_y):

        dx = self.goal_x - pos_x
        dy = self.goal_y - pos_y

        bot_steer = np.arctan2(orientation_z, orientation_w) * 2
        goal_steer = np.arctan2(dy, dx)

        dtheta = goal_steer - bot_steer

        return dtheta

    def gen_reward(self, robo_pos_x, robo_pos_y, orientation_z, orientation_w):

        #Calculating heuristics
        distance_from_goal = self.euclidean_distance(robo_pos_x, robo_pos_y, self.goal_x, self.goal_y)
        distance_from_shortest_path = self.distance_from_line(self.goal_x, self.goal_y, robo_pos_x, robo_pos_y)
        dtheta = self.angle_from_goal(orientation_z, orientation_w, robo_pos_x, robo_pos_y)

        #Checking if robot has reached or lost
        reached = False
        lost = False
        if distance_from_goal < self.goal_threshold:
            reached = True

        if distance_from_goal > self.lost_threshold:
            lost = True

        #Defining distance_from_goal reward function
        if reached:
            reward = 1
        elif distance_from_goal < self.prev_goal_distance:
            reward = 0.01 * (self.prev_goal_distance - distance_from_goal)
        elif distance_from_goal > self.prev_goal_distance:
            reward = -0.2 * (distance_from_goal- self.prev_goal_distance)
        else:
            reward = 0

        self.prev_goal_distance = distance_from_goal

        #Defining angle_from_goal reward function
        # if (self.ideal_dtheta is None):
        #         self.ideal_dtheta = dtheta
        # if abs(dtheta) < abs(self.ideal_dtheta):
        #     reward += 10
        # elif abs(dtheta) > abs(self.ideal_dtheta):
        #     reward -= 5
        # else:
        #     reward = 0

        #Defining distance_from_shortest_path reward function
        # if reached:
        #     reward = 1
        # elif distance_from_shortest_path < self.prev_distance_from_line:
        #     reward = 0.01 * (self.prev_distance_from_line - distance_from_goal)
        # elif distance_from_shortest_path > self.prev_distance_from_line:
        #     reward = -0.2 * (distance_from_goal- self.prev_distance_from_line)
        # else:
        #     reward = 0

        # self.prev_distance_from_line = distance_from_shortest_path

        return reward