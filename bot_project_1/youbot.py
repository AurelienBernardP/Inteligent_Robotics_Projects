# -*- coding: utf-8 -*-
"""
The aim of this code is to show small examples of controlling the displacement of the robot in V-REP. 
(C) Copyright Renaud Detry 2013, Mathieu Baijot 2017, Norman Marlier 2019.
Distributed under the GNU General Public License.
(See http://www.gnu.org/copyleft/gpl.html)
"""
# VREP
from multiprocessing.connection import wait
from turtle import xcor
from Scene_map_v3 import manhattanDistance
import sim as vrep

# Useful import
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
import pygame
import math 

from cleanup_vrep import cleanup_vrep
from vrchk import vrchk
from youbot_init import youbot_init
from youbot_drive import youbot_drive
from youbot_hokuyo_init import youbot_hokuyo_init
from youbot_hokuyo import youbot_hokuyo
from youbot_xyz_sensor import youbot_xyz_sensor
from beacon import beacon_init, youbot_beacon
from utils_sim import angdiff
from scipy.spatial.transform import Rotation as R


import open3d
import cv2 as cv
from shapely.geometry import LineString
from shapely.geometry import Point

from Scene_map_v3 import Scene_map
from astar import getActions
from astar import getAngle
from astar import getRigthLeftAngles
from PID_controller import PID_controller
#from youbot_arm import get_transform

def get_transform(handle1, handle2):
    """Return the transform matrix (4x4)."""
    res, pos = vrep.simxGetObjectPosition(clientID, handle1, handle2, vrep.simx_opmode_oneshot_wait)
    vrchk(vrep, res, True)
    res, euler_angles = vrep.simxGetObjectOrientation(clientID, handle1, handle2, vrep.simx_opmode_oneshot_wait)
    vrchk(vrep, res, True)
    T = np.eye(4)
    T[:3, :3] = open3d.geometry.TriangleMesh.create_coordinate_frame().get_rotation_matrix_from_xyz(euler_angles)
    T[:3, 3] = np.array(pos).T
    
    return T


pygame.init()
screen = pygame.display.set_mode([700, 700])

# Test the python implementation of a youbot
# Initiate the connection to the simulator.
print('Program started')
# Use the following line if you had to recompile remoteApi
# vrep = remApi('remoteApi', 'extApi.h')
# vrep = remApi('remoteApi')

# Close the connection in case if a residual connection exists
vrep.simxFinish(-1)
clientID = vrep.simxStart('127.0.0.1',  19997, True, True, 2000, 5)

# The time step the simulator is using (your code should run close to it).
timestep = .05

# Synchronous mode
returnCode = vrep.simxSynchronous(clientID, True)

# If you get an error like:
#   Remote API function call returned with error code: 64.
# Explanation: simxStart was not yet called.
# Make sure your code is within a function!
# You cannot call V-REP from a script.
if clientID < 0:
    sys.exit('Failed connecting to remote API server. Exiting.')

print('Connection ' + str(clientID) + ' to remote API server open')

# Make sure we close the connection whenever the script is interrupted.
# cleanup_vrep(vrep, id)

# This will only work in "continuous remote API server service".
# See http://www.v-rep.eu/helpFiles/en/remoteApiServerSide.htm
vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)

# Send a Trigger to the simulator: this will run a time step for the physics engine
# because of the synchronous mode. Run several iterations to stabilize the simulation

for i in range(int(1./timestep)):
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)


# Retrieve all handles, mostly the Hokuyo.
h = youbot_init(vrep, clientID)
h = youbot_hokuyo_init(vrep, h)
beacons_handle = beacon_init(vrep, clientID, h)


# Send a Trigger to the simulator: this will run a time step for the physics engine
# because of the synchronous mode. Run several iterations to stabilize the simulation
for i in range(int(1./timestep)):
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)

# Time
t_run = []

##############################################################################
#                                                                            #
#                          INITIAL CONDITIONS                                #
#                                                                            #
##############################################################################
# Define all the variables which will be used through the whole simulation.
# Important: Set their initial values.

# Get the position of the beacons in the world coordinate frame (x, y)
# simx_opmode_oneshot_wait is used. This enforces to have a valid response.
beacons_world_pos = np.zeros((len(beacons_handle), 3))
for i, beacon in enumerate(beacons_handle):   
    res, beacons_world_pos[i] = vrep.simxGetObjectPosition(clientID, beacon, -1,
                                                           vrep.simx_opmode_oneshot_wait)

# Parameters for controlling the youBot's wheels: at each iteration,
# those values will be set for the wheels.
# They are adapted at each iteration by the code.
forwBackVel = 0  # Move straight ahead.
rightVel = 0  # Go sideways.
rotateRightVel = 0  # Rotate.

# First state of state machine
rot_counter = 0
fsm = 'main'
print('Switching to state: ', fsm)

# Get the initial position
res, youbotPos = vrep.simxGetObjectPosition(clientID, h['ref'], -1, vrep.simx_opmode_buffer)
# Set the speed of the wheels to 0.
h = youbot_drive(vrep, h, forwBackVel, rightVel, rotateRightVel)
# Get the target orientation
[res, targetori] = vrep.simxGetObjectOrientation(clientID, h["otarget"], h["r22"], vrep.simx_opmode_oneshot_wait)
# Get the gripper orientation
[res, tori] = vrep.simxGetObjectOrientation(clientID, h["otip"], h["r22"], vrep.simx_opmode_oneshot_wait)

#res, centerOfObject = vrep.simxGetObjectPosition(clientID, h['r22'], h['ref'], vrep.simx_opmode_buffer)

# Send a Trigger to the simulator: this will run a time step for the physic engine
# because of the synchronous mode. Run several iterations to stabilize the simulation.
for i in range(int(1./timestep)):
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)


house_map = Scene_map(75,75,beacons_world_pos[:,:2])

# Actions that will come from A* algo.
actions = [('South', 0.001)]
currActionIndex = 0
goalCell = (-1,-1)

fsm_step = 1

tableCenter = [0,0]

# To track position at the beginning of a move.
youbotFirstPos = youbotPos

# Put the fct elsewhere !

def get_speeds(map_rep,real_bot_position,target_pos_mat,current_orientation,target_orientation):
    
    goal_coordinates = map_rep.get_cell_center_coordinates(target_pos_mat[1],target_pos_mat[0])

    map_dir_vector = np.subtract(goal_coordinates,real_bot_position) # (x,y) vector
    euler_dist = np.linalg.norm(map_dir_vector,2)
    map_dir_vector = map_dir_vector / euler_dist

    theta = current_orientation
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    bot_dir_vector = np.dot(R,map_dir_vector)

    if target_orientation == None :
        target_orientation = np.arctan(map_dir_vector[0]/map_dir_vector[1])
    right_speed = - 2 * euler_dist * bot_dir_vector[0]
    up_speed =  2 * euler_dist * bot_dir_vector[1]
    rot_speed = 2 * angdiff(current_orientation, target_orientation)

    return (right_speed,up_speed,rot_speed)


def getRotationSpeed(angle1, angle2):

    angleRight, angleLeft = getRigthLeftAngles(angle1, angle2)
            
    # Rotate left or right (choose the best of the two move).
    if (angleRight <= angleLeft):
        distanceToGoal = angleRight
        rotateRightVel = - 1/3 * distanceToGoal
    else:
        distanceToGoal = -angleLeft
        rotateRightVel = 1/3 * distanceToGoal

    rotateRightVel = rot_PID.control(0,distanceToGoal)

    return rotateRightVel, distanceToGoal


def setArmJoints(targetJoint):

    # Joint 0.
    res = vrep.simxSetJointTargetPosition(clientID, h["armJoints"][0], targetJoint[0], vrep.simx_opmode_oneshot)            
    res, joint_0 = vrep.simxGetJointPosition(clientID, h["armJoints"][0], vrep.simx_opmode_buffer)

    # Joint 1.   
    res = vrep.simxSetJointTargetPosition(clientID, h["armJoints"][1], targetJoint[1], vrep.simx_opmode_oneshot)            
    res, joint_1 = vrep.simxGetJointPosition(clientID, h["armJoints"][1], vrep.simx_opmode_buffer)

    # Joint 3.
    res = vrep.simxSetJointTargetPosition(clientID, h["armJoints"][3], targetJoint[3], vrep.simx_opmode_oneshot)            
    res, joint_3 = vrep.simxGetJointPosition(clientID, h["armJoints"][3], vrep.simx_opmode_buffer)

    return joint_0, joint_1, joint_3


    
def find_objects(table_center , orbit_d = 0.85):

    # Read data from the depth camera (Hokuyo)
    # Reading a 3D image costs a lot to VREP (it has to simulate the image). It also requires a lot of 
    # bandwidth, and processing a 3D point cloud (for instance, to find one of the boxes or cylinders that 
    # the robot has to grasp) will take a long time in MATLAB. In general, you will only want to capture a 3D 
    # image at specific times, for instance when you believe you're facing one of the tables.
    
    # Reduce the view angle to pi/8 in order to better see the objects. Do it only once. 
    # ^^^^^^     ^^^^^^^^^^    ^^^^                                     ^^^^^^^^^^^^^^^ 
    # simxSetFloatSignal                                                simx_opmode_oneshot_wait
    #            |
    #            rgbd_sensor_scan_angle
    # The depth camera has a limited number of rays that gather information. If this number is concentrated 
    # on a smaller angle, the resolution is better. pi/8 has been determined by experimentation. 
    res = vrep.simxSetFloatSignal(clientID, 'rgbd_sensor_scan_angle', np.pi/8, vrep.simx_opmode_oneshot_wait)
    vrchk(vrep, res) # Check the return value from the previous V-REP call (res) and exit in case of error.
    
    # Ask the sensor to turn itself on, take A SINGLE POINT CLOUD, and turn itself off again. 
    # ^^^     ^^^^^^                ^^       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # simxSetIntegerSignal          1        simx_opmode_oneshot_wait
    #         |
    #         handle_xyz_sensor
    res = vrep.simxSetIntegerSignal(clientID, 'handle_xyz_sensor', 1, vrep.simx_opmode_oneshot_wait)
    #vrchk(vrep, res)
    
    ################# BE CAREFUL #################
    # For some reasons, the vrep call can return None. In this case, you can safetly ignore the verification 'vrchk'
    
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)
    
    # Simulation with capturing raw point cloud.
    # Then retrieve the last point cloud the depth sensor took.
    # If you were to try to capture multiple images in a row, try other values than 
    # vrep.simx_opmode_oneshot_wait. 
    print('Capturing point cloud...\n')
    pts = youbot_xyz_sensor(vrep, h, vrep.simx_opmode_oneshot_wait)
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)

    ###################################################################
    # Processing of your point cloud, depth images, etc...
    
    #1) remove points that are too far. further than 1.5 m away is no useful
    to_remove = []
    for i in range(np.shape(pts)[0]):
        
            if pts[i,3] > 0.9:
                to_remove.append(i)
    pts = np.delete(pts,to_remove,axis = 0)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts[:,:3])

    pts = np.asarray(pcd.points)

    #2) TRANSFORM POINTS TO REFERENCE FRAME COORDS
    T_xyz_ref = get_transform(h["xyzSensor"], h["ref"])

    for i in range(np.shape(pts)[0]):
        pts[i,:] = T_xyz_ref.dot(np.append(pts[i,:],1) )[:3]

    #3) remove points below the table top
    to_remove = []
    for i in range(np.shape(np.asarray(pcd.points))[0]):
        if abs(pts[i,2]) < 0.0925:
            to_remove.append(i)
    pts = np.delete(pts,to_remove,axis = 0)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts[:,:3])

    # 3) Point cloud segmentation in case there are various elements

    if np.shape(np.asarray(pcd.points))[0] > 0 :
        with open3d.utility.VerbosityContextManager(
                open3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                pcd.cluster_dbscan(eps=0.05, min_points=20, print_progress=True))
        '''
        # code to visualize segmentation
        #  
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = open3d.utility.Vector3dVector(colors[:, :3])
        
        open3d.visualization.draw(
            [pcd, axis_aligned_bounding_box, oriented_bounding_box])
        '''    
        # 4) estimate normals of point cloud
        pcd.estimate_normals()
        pcd.normalize_normals()
        pcd.orient_normals_consistent_tangent_plane(k=20)
        #open3d.visualization.draw_geometries([pcd], point_show_normal=True)

         # 4) Segment main plane of point cloud (Main face of object)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.005,
                                                ransac_n=4,
                                                num_iterations=1000)

        inlier_cloud = pcd.select_by_index(inliers)
        #inlier_cloud.paint_uniform_color([1.0, 0, 0])

        # Estimate once again the normals of the main face of the object to get better results
        inlier_cloud.estimate_normals()
        inlier_cloud.orient_normals_consistent_tangent_plane(k=20)
        inlier_cloud.normalize_normals()

        #Find average normal by averaging all normals on the face
        average_normal = np.mean(np.asarray(inlier_cloud.normals),axis=0)
        
        #Find orientation of youbot to be perfectly in front of main face of object
        dy = -average_normal[1] / np.linalg.norm(average_normal[0:2])
        dx = -average_normal[0] / np.linalg.norm(average_normal[0:2])
        angle_to_grasp = 0.0

        if dx >= 0 and dy >= 0:
            angle_to_grasp = np.pi/2 + abs(math.atan(dy/dx))
        if dx >= 0 and dy <= 0:
            angle_to_grasp = abs(math.atan(dx/dy))
        if dx <= 0 and dy >= 0:
            angle_to_grasp = -(np.pi/2 + abs(math.atan(dy/dx)))
        if dx <= 0 and dy <= 0:
            angle_to_grasp = -abs(math.atan(dx/dy))

        print("target orientation in radians", angle_to_grasp )

        #find center of the main face of the object
        face_center =  np.asarray(inlier_cloud.get_center())
        print('center of face ',face_center)

        def line_and_circle_intersection(circle_center,radius,line_vector_start,line_vector_end):
            
            p = Point(circle_center[0],circle_center[1])
            c = p.buffer(radius).boundary
            l = LineString([line_vector_start, line_vector_end])
            i = c.intersection(l)
            print("intersection = ", i)
            if i.is_empty :
                return None
            return (i.x, i.y)

        face_center_absolute_pos = face_center[:2] + youbotPos[:2]
        line_end = face_center[:2] + (orbit_d * average_normal[:2])
        intersection = line_and_circle_intersection(table_center,orbit_d,face_center_absolute_pos,line_end)
        if intersection == None:
            #should never happen because object(line start) is within orbit and line is as long as orbit radius
            print("several intersections should never happen")

            return 1,None,None,None,None

        print("intersecting points with orbit= ", intersection)


        point_on_orbit = intersection

        dy = table_center[0] - point_on_orbit[0]
        dx = average_normal[0] - point_on_orbit[1]
        angle_to_table = 0.
        if dx >= 0 and dy >= 0:
            angle_to_table = np.pi/2 + abs(math.atan(dy/dx))
        if dx >= 0 and dy <= 0:
            angle_to_table = abs(math.atan(dx/dy))
        if dx <= 0 and dy >= 0:
            angle_to_table = -(np.pi/2 + abs(math.atan(dy/dx)))
        if dx <= 0 and dy <= 0:
            angle_to_table = -abs(math.atan(dx/dy))

        return 0,angle_to_table,point_on_orbit,angle_to_grasp, face_center
    else:
        print("No objects were detected")
        return 1,None, None,None,None

# Start the demo. 
intial_pos_route = (0,0)
counter = 0
show = False
forward_PID = PID_controller(timestep,3,0.8,0,True)
rot_PID = PID_controller(timestep,3.05,0.8,0,True)
tables = []
target_table = np.zeros(2)
target_table = [-1, -6] # to remove
youbotFirstEuler = -1
gripperState = 1
isObjectDropped = False

while True:
    try:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                exit()

        # Time management
        t_loop = time.perf_counter()

        # Check the connection with the simulator
        if vrep.simxGetConnectionId(clientID) == -1:
            sys.exit('Lost connection to remote API.')

        # Get the position and the orientation of the robot.
        #res, youbotPos = vrep.simxGetObjectPosition(clientID, h['ref'], -1, vrep.simx_opmode_streaming)
        #vrchk(vrep, res, True) # Check the return value from the previous V-REP call (res) and exit in case of error.
        res, youbotEuler = vrep.simxGetObjectOrientation(clientID, h['ref'], -1, vrep.simx_opmode_streaming)
        vrchk(vrep, res, True)       

        # Get the distance from the beacons
        # Change the flag to True to constraint the range of the beacons
        beacon_dist = youbot_beacon(vrep, clientID, beacons_handle, h, flag=False)
        house_map.update_bot_pos(beacon_dist,youbotEuler[2])
        # Get data from the hokuyo - return empty if data is not captured
        scanned_points, contacts = youbot_hokuyo(vrep, h, vrep.simx_opmode_buffer)
        vrchk(vrep, res)
        

        # Get th youbot position given by the triangulation instead of gps (milestone 1.ii).
        youbotPos = house_map.bot_pos_estimate

        # Get youbot state.
        state = house_map.map_position_to_mat_index(youbotPos[0], youbotPos[1])
       

        if show == True:
            house_map.pygame_screen_refresh(screen,intial_pos_route,actions,beacon_dist)
            for i in range(len(tables)):
                x_screen_size, y_screen_size = screen.get_size()
                x = (tables[i][0] + (15 / 2)) * (x_screen_size/15)
                y = y_screen_size - ((tables[i][1] + (15 / 2)) * (y_screen_size/15))
                radius = tables[i][2] * (x_screen_size/15)
                pygame.draw.circle(screen, (255, 0, 0), (x,y), radius)
            pygame.display.flip()        
            show = False
        if counter % 5 == 0 or counter < 5:
            #update map and refresh display every 5 ticks except at the begining where a lot of data is gathered
            house_map.update_contact_map(scanned_points,contacts)
            show = True
            

        #if counter == 550:
            #forward_PID.plot()
            #rot_PID.plot()
        
        #print(counter)
        counter +=1
        if fsm == 'detect_tables' or (counter % 250 == 0):

            img = np.copy(house_map.obstacle_cells_grid)
            img = (img + 1) % 2
            img *= 255
            img = np.uint8(img)
            img = cv.blur(img, (3, 3))
            circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 7,
                            param1=50, param2=10,
                            minRadius=8, maxRadius=9)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                tables = []
                for i,pt in enumerate(circles[0, :]):
                    a, b, r = pt[0], pt[1], pt[2]
                    a = (a - (0.5*300)) * (15/300)
                    b = (b - (0.5*300)) * (15/300)
                    r = r * (15/300)
                    tables.append(np.asarray([a,b,r]))
                    threshold_dist = 2.5

                for i in range(len(tables)):
                    
                    test_table_1 = tables[i]

                    for j in range(len(tables)):
                        test_table_2 = tables[j]

                        if np.linalg.norm(test_table_1[:2].T - test_table_2[:2]) > 2.5:
                            print('found target table')
                            target_table[0] = tables[i][0]
                            target_table[1] = tables[i][1]
                        print(target_table)
            else :
                print("no tables detected")


        # Apply the state machine.
        if fsm == 'main':

            if isObjectDropped == True:
                fsm = 'finished'
                print('Switching to state: ', fsm)

            # Explore the whole map.
            if fsm_step == 1:
                fsm = 'exploring'
                print('Switching to state: ', fsm)
            
            # Move objects from tables to tables.
            if fsm_step == 2:
                fsm = 'moveObject'
                print('Switching to state: ', fsm)

        
        elif fsm == 'moveObject':

                # test (to replace by point cloud processing)
                #angleOfObject = np.pi
                #centerOfObject = np.array([-0.02, 0.37, 0.26])
                
                # Infinit loop (test)
                print(tableCenter)
                print(known_table1)
                if (tableCenter == known_table1).all():
                    tableCenter = target_table
                else:
                    tableCenter = known_table1

                tableCenterCell = house_map.map_position_to_mat_index(tableCenter[0],tableCenter[1])
                
                # Find a cell close to the table and the youbot.
                closestFreeCells = set(x for x in house_map.free_cells if (manhattanDistance(x, tableCenterCell) <= 8 and x not in house_map.padding_cells))
                def f(x):
                    return manhattanDistance(x, state)
                cellNextToGoal = min(closestFreeCells,key=f)

                print(cellNextToGoal)

                # Set actions to take.
                currActionIndex = 0
                actions = getActions(house_map, cellNextToGoal)
                print("actions", actions)
                
                if len(actions) == 0:
                    fsm = 'moveObject'
                    print('Switching to state: ', fsm)
                else:
                    youbotFirstPos = youbotPos
                    print(state)
                    fsm = 'rotate'
                    print('Switching to state: ', fsm)
                    intial_pos_route = (youbotPos[0],youbotPos[1])


        elif fsm == 'exploring':

            currActionIndex = 0

            # Check if the map is fully explored.
            isMapFullyExplored = house_map.frontier_cells_list == []
            
            # Set the goal state (if map not fully explored).
            if not isMapFullyExplored:
                cellNextToGoal = (-1,-1)
                while cellNextToGoal == (-1,-1) and  len(house_map.frontier_cells):
                    goalCell = house_map.frontier_cells_list[-1] # take the newest frontier point

                    # Turn the goal state to be the fist free cell next to the goal cell.
                    for i in range(goalCell[0]-1,goalCell[0]+2,1):
                            for j in range(goalCell[1]-1,goalCell[1]+2,1):
                                if (i,j) in house_map.free_cells and (i,j) not in house_map.padding_cells:
                                    cellNextToGoal = (i,j)
                    if cellNextToGoal == (-1,-1):
                        house_map.frontier_cells.discard(goalCell)
                        house_map.frontier_cells_list.remove(goalCell)
            
            # Check if the map is fully explored.
            isMapFullyExplored = house_map.frontier_cells_list == []
            #isMapFullyExplored = True #to remove
            
            # Set actions to take.
            actions = getActions(house_map, cellNextToGoal)
            print("actions", actions)

            if isMapFullyExplored:
                fsm_step = 2
                fsm = 'main'
                print('Switching to state: ', fsm)
            elif len(actions) == 0:
                house_map.frontier_cells.discard(goalCell)
                house_map.frontier_cells_list.remove(goalCell)
                fsm = 'exploring'
                print('Switching to state: ', fsm)
            elif len(actions) != 0:
                youbotFirstPos = youbotPos
                fsm = 'rotate'
                print('Switching to state: ', fsm)
                intial_pos_route = (youbotPos[0],youbotPos[1])


        elif fsm == 'rotate':
            # Compute the value of the left and right angles.
            angle1 = youbotEuler[2]
            angle2 = getAngle(actions[currActionIndex][0])

            rotateRightVel, distanceToGoal = getRotationSpeed(angle1, angle2)

            # Stop when the robot reached the goal angle.
            if abs(distanceToGoal) < .01 and abs(rotateRightVel) < 0.1:
                rotateRightVel = 0
                fsm = 'moveFoward'
                print('Switching to state: ', fsm)

        
        elif fsm == 'moveFoward':
            
            # Compute the distance already travelled for this move.
            distance = math.sqrt(abs(youbotPos[0] - youbotFirstPos[0])**2 + abs(youbotPos[1] - youbotFirstPos[1])**2)

            # Compute the distance that remain to travel.
            distanceToGoal = actions[currActionIndex][1] - distance

            # Set the speed to reach the goal.
            forwBackVel = forward_PID.control(0,distanceToGoal)
            #forwBackVel = - 0.5 * distanceToGoal  # to remove

            # Stop when the robot reached the goal position.
            if abs(distanceToGoal) < .01 and abs(forwBackVel) < 0.1:
                forwBackVel = 0  # Stop the robot.

                # Perform the next action or do planning if no action remain.
                currActionIndex = currActionIndex + 1
                youbotFirstPos = youbotPos
                if (currActionIndex >= len(actions)):
                    fsm = 'stop'
                    print('Switching to state: ', fsm)
                else:
                    fsm = 'rotate'
                    print('Switching to state: ', fsm)
            
            # Stop if we explored the goal cell and we are close.
            elif goalCell not in house_map.frontier_cells and manhattanDistance(goalCell, state) < 15 and fsm_step == 1:
                fsm = 'stop'
                print('Switching to state: ', fsm)

        
        elif fsm == 'rotateToTable':
 
            dx = tableCenter[0] - youbotPos[0]
            dy = tableCenter[1] - youbotPos[1]

            if dx >= 0 and dy >= 0:
                angle = np.pi/2 + abs(math.atan(dy/dx))
            if dx >= 0 and dy <= 0:
                angle = abs(math.atan(dx/dy))
            if dx <= 0 and dy >= 0:
                angle = -(np.pi/2 + abs(math.atan(dy/dx)))
            if dx <= 0 and dy <= 0:
                angle = -abs(math.atan(dx/dy))

            angle1 = youbotEuler[2] 
            angle2 = angle

            rotateRightVel, distanceToGoal = getRotationSpeed(angle1, angle2)
            
            # Stop when the robot reached the goal angle.
            if abs(distanceToGoal) < .01 and abs(rotateRightVel) < 0.1:
                rotateRightVel = 0
                fsm = 'moveToTable'
                print('Switching to state: ', fsm)
        

        elif fsm == 'moveToTable':

            distanceToCenter = math.sqrt(abs(youbotPos[0] - tableCenter[0])**2 + abs(youbotPos[1] - tableCenter[1])**2) # use triangulation !
            distanceToGoal = distanceToCenter - 0.85

            # Set the speed to reach the goal.
            forwBackVel = forward_PID.control(0,distanceToGoal) * 0.2

            # Stop when the robot reached the goal position.
            if abs(distanceToGoal) < .01 and abs(forwBackVel) < 0.1:
                forwBackVel = 0
                fsm = 'scanTable'
                print('Switching to state: ', fsm)

        
        # Not implemented ! we mock it with true data from the map.
        elif fsm == 'scanTable':

            # Check if we are on the table 2 (no scaning).
            if gripperState == 0:
                fsm = 'circleAroundTable'
                print('Switching to state: ', fsm)

            rotateRightVel = 0.09
            rightVel = -0.1
            
            
            status = 1
            if counter % 25 == 0:
                radius_of_orbit = math.sqrt(abs(youbotPos[0] - tableCenter[0])**2 + abs(youbotPos[1] - tableCenter[1])**2)
                target_table_center = (tableCenter[0],tableCenter[1])
                status, target_angle_with_table,target_position,target_bot_orientation, target_clamp_pos = find_objects(target_table_center,radius_of_orbit) 
                #centerOfObject = np.array([-target_clamp_pos[0], -target_clamp_pos[1], target_clamp_pos[2], 1])
            

            if  status == 0: # quid if no more objects on table ? 
                # Mock sensor
                #res, centerOfObject = vrep.simxGetObjectPosition(clientID, h['r22'], h['ref'], vrep.simx_opmode_streaming) # Cheat function !
                #centerOfObject = np.array([centerOfObject[0], centerOfObject[1], centerOfObject[2], 1])
                
                print("--------")
                print(target_angle_with_table)
                print(target_bot_orientation)
                #print(centerOfObject)
                print("--------")

                # Mock sensor
                centerOfObject = np.array([0.032, 0.53, 0.15, 1])
                target_angle_with_table = np.pi
                target_bot_orientation = 0

                rotateRightVel = 0
                rightVel = 0
                fsm = 'circleAroundTable'
                print('Switching to state: ', fsm)


        elif fsm == 'circleAroundTable':
            
            # We need to be at distance 0.850 m from table center and face it !
            
            # Set the goal angle.
            angle2 = target_angle_with_table

            # Get "table angle" from youbot angle.
            angle1 = youbotEuler[2]
            if angle1 <= 0:
                angle1 = np.pi + angle1
            else:
                angle1 = angle1 - np.pi

            # Compute the value of the left and right angles.
            angleRight, angleLeft = getRigthLeftAngles(angle1, angle2)
            
            # Rotate left or right (choose the best of the two move).
            if (angleRight <= angleLeft):
                distanceToGoal = angleRight
                rotateRightVel = -0.09 * 0.5
                rightVel = 0.1 * 0.5
            else:
                distanceToGoal = angleLeft
                rotateRightVel = 0.09 * 0.5
                rightVel = -0.1 * 0.5
            
            print(distanceToGoal)
            
            # Stop when the robot reached the goal angle.
            if distanceToGoal < .01:
                rotateRightVel = 0
                rightVel = 0
                
                if gripperState == 1:
                    fsm = 'rescan'
                    print('Switching to state: ', fsm)
                else:
                    fsm = 'rotateToObjectFront'
                    print('Switching to state: ', fsm)

        
        elif fsm == 'rescan':

            # Take a new shot of the object.
            radius_of_orbit = math.sqrt(abs(youbotPos[0] - tableCenter[0])**2 + abs(youbotPos[1] - tableCenter[1])**2)
            target_table_center = (tableCenter[0],tableCenter[1])
            status, n,n,n, target_clamp_pos = find_objects(target_table_center,radius_of_orbit) 
            if status == 0:
                centerOfObject = np.array([-target_clamp_pos[0], -target_clamp_pos[1], target_clamp_pos[2], 1])
            
            fsm = 'rotateToObjectFront'
            print('Switching to state: ', fsm)
            

        elif fsm == 'rotateToObjectFront':

            # Compute the value of the left and right angles.
            angle1 = youbotEuler[2]

            # Get angle to have back of the youbot in front of the object.
            if target_bot_orientation <= 0:
                angle2 = np.pi + target_bot_orientation
            else:
                angle2 = target_bot_orientation - np.pi

            rotateRightVel, distanceToGoal = getRotationSpeed(angle1, angle2)

            # Stop when the robot reached the goal angle.
            if abs(distanceToGoal) < .01 and abs(rotateRightVel) < 0.1:
                rotateRightVel = 0
                fsm = 'deployArm'
                print('Switching to state: ', fsm)

        
        elif fsm == 'deployArm':

            targetJoint = [np.pi, -np.pi/4, 0., 0.]
            
            # Rotate the arm.
            joint_0, joint_1, joint_3 = setArmJoints(targetJoint)

            # Stop when the robot is at an angle close to target.
            cond0 = abs(angdiff(joint_0, targetJoint[0])) < .001
            cond1 = abs(angdiff(joint_1, targetJoint[1])) < .001
            cond3 = abs(angdiff(joint_3, targetJoint[3])) < .001
            
            if cond0 & cond1 & cond3:
                res = vrep.simxSetIntegerSignal(clientID, 'km_mode', 2, vrep.simx_opmode_oneshot_wait)
                fsm = 'moveArm'
                print('Switching to state: ', fsm)

            
        elif fsm == 'moveArm':
            
            if centerOfObject.size == 4:
                T_ref_arm = get_transform(h["ref"], h["armRef"])
                print(T_ref_arm)
                print(centerOfObject)
                centerOfObject =  np.matmul(T_ref_arm,centerOfObject)
                centerOfObject = centerOfObject[0:3]
                centerOfObject[1] += 0.035

            # Send command to the robot arm.
            res = vrep.simxSetObjectPosition(clientID, h["ptarget"], h["armRef"], centerOfObject, vrep.simx_opmode_oneshot)

            # Transform for the arm orientation
            rot1 = R.from_quat([0., np.sin(-3/8*np.pi), 0., np.cos(-3/8*np.pi)])
            rot2 = R.from_quat([np.sin(-np.pi/4), 0., 0., np.cos(-np.pi/4)])
            quats = (rot1*rot2).as_quat()

            res = vrep.simxSetObjectQuaternion(clientID, h["otarget"], h["r22"], quats, vrep.simx_opmode_oneshot)
            vrchk(vrep, res, True)

            # Get the gripper position and check whether it is at destination (the original position).
            [res, tpos] = vrep.simxGetObjectPosition(clientID, h["ptip"], h["armRef"], vrep.simx_opmode_buffer)
            vrchk(vrep, res, True)

            # Check only position but orientation can be added.
            cond_pos = np.linalg.norm(tpos - centerOfObject) < .005
            print(np.linalg.norm(tpos - centerOfObject))

            # Close the gripper if we must grab an object and open it if we must drop it.
            if cond_pos:
                fsm = 'activateGripper'
                print('Switching to state: ', fsm)
                time_to_close = time.time()

        
        elif fsm == 'activateGripper':
            
            # Open or close the gripper.
            if gripperState == 1:
                close = 0
            else:
                close = 1
            res = vrep.simxSetIntegerSignal(clientID, 'gripper_open', close, vrep.simx_opmode_oneshot_wait);
            #vrchk(vrep, res)
            
            if time.time()-time_to_close > 3.:
                gripperState = close
                fsm = 'liftUp'
                res = vrep.simxSetIntegerSignal(clientID, 'km_mode', 0, vrep.simx_opmode_oneshot_wait)
                #vrchk(vrep, res, True)
                print('Switching to state: ', fsm)

                
        elif fsm == "liftUp":

            # Joint 3
            target_joint_3 = -np.pi/2
            joint_index = 3
            res = vrep.simxSetJointTargetPosition(clientID, h["armJoints"][joint_index], target_joint_3, vrep.simx_opmode_oneshot)            
            res, joint_3 = vrep.simxGetJointPosition(clientID, h["armJoints"][joint_index], vrep.simx_opmode_buffer)

            # Condition
            cond = abs(angdiff(joint_3, target_joint_3)) < .001
            if cond:
                fsm = "storeArm"
                print('Switching to state: ', fsm)

        
       # Need to be dynamic
        elif fsm == 'storeArm':

            if gripperState == 1:
                isObjectDropped = True

            targetJoint = [0, 0, 0., 0]
            
            # Rotate the arm.
            joint_0, joint_1, joint_3 = setArmJoints(targetJoint)

            # Stop when the robot is at an angle close to target.
            cond0 = abs(angdiff(joint_0, targetJoint[0])) < .001
            cond1 = abs(angdiff(joint_1, targetJoint[1])) < .001
            cond3 = abs(angdiff(joint_3, targetJoint[3])) < .001
            
            if cond0 & cond1 & cond3:
                fsm = 'main'
                print('Switching to state: ', fsm)
        




    
        elif fsm == 'stop':
            forwBackVel = 0  # Stop the robot.
            
            # Check if the youbot is stoped.
            if abs(youbotPos[0] - youbotFirstPos[0]) + abs(youbotPos[1] - youbotFirstPos[1]) <= .01:
                if fsm_step == 1:
                    fsm = 'exploring'
                    print('Switching to state: ', fsm)
                if fsm_step == 2:
                    fsm = 'rotateToTable'
                    print('Switching to state: ', fsm)
            
            youbotFirstPos = youbotPos


        elif fsm == 'finished':
            print('Finished exploration, no more accessible cells in frontier')
            time.sleep(3)
            break


        else:
            sys.exit('Unknown state ' + fsm)

        # Update wheel velocities.
        h = youbot_drive(vrep, h, forwBackVel, rightVel, rotateRightVel)

        # What happens if you do not update the velocities?
        # The simulator always considers the last speed you gave it,
        # until you set a new velocity.

        # Send a Trigger to the simulator: this will run a time step for the physic engine
        # because of the synchronous mode.
        end_time = time.perf_counter()
        t_run.append((end_time-t_loop)*1000.)  # In ms
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)
    except KeyboardInterrupt:
        cleanup_vrep(vrep, clientID)
        sys.exit('Stop simulation')

cleanup_vrep(vrep, clientID)
print('Simulation has stopped')
# Histogram of time loop
n, x, _ = plt.hist(t_run, bins=100)
plt.vlines(np.min(t_run), 0, np.max(n), linewidth=1.5, colors="r")
plt.vlines(np.max(t_run), 0, np.max(n), linewidth=1.5, colors="k")
plt.xlabel(r"time $t_{\rm{loop}}$ (ms)")
plt.ylabel("Number of loops (-)")
plt.show()